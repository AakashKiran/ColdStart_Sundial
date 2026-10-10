"""Train the notebook-aligned RAID model on Intel aggregate SKU demand."""
from __future__ import annotations

import json
import random
import sys
import time
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.raid import RAID

CFG_PATH = ROOT / "configs/intel_sku/raid_aggregate.json"


class EpisodeDataset(Dataset):
    """Origin-aware pseudo-cold-start episodes from training SKUs only.

    Target history is never an input. Donor histories are the H complete weeks
    before an origin; donor mean/std use all observed donor values before that
    origin. Demand and donor statistics are divided by a training-only scale.
    """
    def __init__(self, demand_df, metadata_df, embeddings_np, sku_to_idx,
                 train_sku_list, config, demand_scale, seed=42):
        self.cfg = config
        self.train_skus = list(train_sku_list)
        self.embeddings = embeddings_np
        self.sku_to_idx = sku_to_idx
        self.scale = float(demand_scale)
        d = config["data"]
        self.sku_col, self.week_col = d["sku_column"], d["week_column"]
        self.y_col, self.start_col = d["target_column"], d["launch_column"]
        self.h = int(config["experiment"]["forecast_window_weeks"])
        self.k = int(config["model"]["neighbors_k"])
        self.meta = metadata_df.set_index(self.sku_col, drop=False)
        self.series = {
            str(s): g.set_index(self.week_col)[self.y_col].sort_index()
            for s, g in demand_df.groupby(self.sku_col)
        }
        self.min_week = int(demand_df[self.week_col].min())
        self.max_week = int(demand_df[self.week_col].max())
        self.rng = random.Random(seed)
        self.episodes = {}
        print("Preparing origin-aware pseudo-cold-start episodes...")

        for sku in self.train_skus:
            if sku not in self.series or sku not in self.meta.index:
                continue
            launch = int(self.meta.loc[sku, self.start_col])
            candidates = []
            last_origin = self.max_week - self.h + 1
            for origin in range(max(launch, self.min_week + self.h), last_origin + 1):
                target = self.series[sku].reindex(pd.RangeIndex(origin, origin + self.h))
                if not target.notna().any():
                    continue
                hist_weeks = pd.RangeIndex(origin - self.h, origin)
                donors = []
                for donor in self.train_skus:
                    if donor == sku or donor not in self.series:
                        continue
                    if int(self.meta.loc[donor, self.start_col]) >= origin:
                        continue
                    donor_series = self.series[donor]
                    hist = donor_series.reindex(hist_weeks)
                    if not hist.notna().all():
                        continue
                    prior = donor_series.loc[donor_series.index < origin].dropna()
                    if len(prior) < 2:
                        continue
                    prior_scaled = prior.to_numpy(dtype=np.float32) / self.scale
                    mu = float(np.mean(prior_scaled))
                    sd = float(np.std(prior_scaled))
                    if not np.isfinite(sd) or sd < 1e-6:
                        sd = 1e-6
                    donors.append((
                        donor,
                        hist.to_numpy(dtype=np.float32) / self.scale,
                        mu,
                        sd,
                    ))
                if not donors:
                    continue
                mask = target.notna().to_numpy(dtype=np.float32)
                y = target.to_numpy(dtype=np.float32) / self.scale
                y = np.where(mask > 0, y, 0.0).astype(np.float32)
                candidates.append((origin, y, mask, donors))

            if candidates:
                self.episodes[sku] = candidates

        missing = set(self.train_skus) - set(self.episodes)
        if missing:
            print("Training SKUs without valid episodes (excluded from sampling):", sorted(missing))
        if not self.episodes:
            raise RuntimeError("No valid training episodes found.")
        self.active_skus = sorted(self.episodes)
        counts = [len(self.episodes[s]) for s in self.active_skus]
        print(f"Prepared {len(self.active_skus)} SKUs; episode range {min(counts)}–{max(counts)}.")

    def __len__(self):
        return len(self.active_skus)

    def __getitem__(self, ix):
        sku = self.active_skus[ix]
        _, y, ymask, donor_candidates = self.rng.choice(self.episodes[sku])
        target_idx = self.sku_to_idx[sku]
        target_emb = self.embeddings[target_idx]

        # Match the notebook's embedding-dot-product donor ranking.
        donor_ids = [self.sku_to_idx[x[0]] for x in donor_candidates]
        similarities = self.embeddings[donor_ids] @ target_emb
        chosen = np.argsort(-similarities)[:min(self.k, len(donor_candidates))]
        selected = [donor_candidates[j] for j in chosen]

        idx = np.zeros(self.k, dtype=np.int64)
        hist = np.zeros((self.k, self.h), dtype=np.float32)
        mu = np.zeros(self.k, dtype=np.float32)
        sd = np.ones(self.k, dtype=np.float32)
        donor_mask = np.zeros(self.k, dtype=np.bool_)
        for j, (donor, values, donor_mu, donor_sd) in enumerate(selected):
            idx[j] = self.sku_to_idx[donor]
            hist[j] = values
            mu[j], sd[j], donor_mask[j] = donor_mu, donor_sd, True

        return {
            "target_idx": np.int64(target_idx),
            "donor_idx": idx,
            "donor_history": hist,
            "donor_mean": mu,
            "donor_std": sd,
            "donor_mask": donor_mask,
            "target_values": y,
            "target_mask": ymask,
        }


def collate_episodes(batch):
    return {k: torch.as_tensor(np.stack([item[k] for item in batch])) for k in batch[0]}


def main():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    d, o, m, tcfg = cfg["data"], cfg["outputs"], cfg["model"], cfg["training"]
    seed = int(cfg["experiment"]["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_started = time.time()
    run_started_utc = datetime.now(timezone.utc).isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    split_dir = ROOT / o["split_dir"]
    train_skus = pd.read_csv(split_dir / "train_skus.csv")["sku"].astype(str).tolist()
    test_skus = pd.read_csv(split_dir / "test_skus.csv")["sku"].astype(str).tolist()
    metadata = pd.read_csv(ROOT / d["sku_metadata_path"])
    demand = pd.read_csv(ROOT / d["aggregate_demand_path"])
    embeddings = np.load(ROOT / d["embedding_path"]).astype(np.float32)
    embed_index = json.loads((ROOT / d["embedding_index_path"]).read_text(encoding="utf-8"))
    embed_skus = [str(s) for s in embed_index["skus"]]
    if embeddings.shape != (len(embed_skus), int(m["embedding_dim"])):
        raise ValueError(f"Unexpected embedding shape {embeddings.shape}")
    sku_to_idx = {sku: i for i, sku in enumerate(embed_skus)}
    if set(train_skus + test_skus) - set(sku_to_idx):
        raise ValueError("Train/test split contains SKUs missing from embedding index.")
    if set(train_skus) & set(test_skus):
        raise ValueError("Train/test SKU split overlaps.")

    series = {
        str(s): g.set_index(d["week_column"])[d["target_column"]].sort_index()
        for s, g in demand.groupby(d["sku_column"])
    }
    train_values = np.concatenate([
        series[s].dropna().to_numpy(dtype=np.float64)
        for s in train_skus if s in series
    ])
    demand_scale = float(np.std(train_values))
    if not np.isfinite(demand_scale) or demand_scale <= 0:
        raise ValueError(f"Invalid training-only demand scale: {demand_scale}")
    print(f"Training-only demand std scale: {demand_scale:.6f}")
    print("Raw train demand min/median/mean/max:",
          float(np.min(train_values)), float(np.median(train_values)),
          float(np.mean(train_values)), float(np.max(train_values)))

    dataset = EpisodeDataset(
        demand, metadata, embeddings, sku_to_idx, train_skus, cfg, demand_scale, seed=seed
    )
    loader = DataLoader(
        dataset, batch_size=min(int(tcfg["batch_size"]), len(dataset)),
        shuffle=True, num_workers=0, collate_fn=collate_episodes
    )
    batch0 = next(iter(loader))
    assert not any(
        batch0["target_idx"][i].item()
        in batch0["donor_idx"][i][batch0["donor_mask"][i]].tolist()
        for i in range(len(batch0["target_idx"]))
    ), "Self-donor leakage detected."

    model = RAID(
        horizon=int(cfg["experiment"]["forecast_window_weeks"]),
        embedding_dim=int(m["embedding_dim"]),
        bottleneck_dim=int(m["bottleneck_dim"]),
        k=int(m["neighbors_k"]),
        gat_heads=int(m["gat_heads"]),
        gat_layers=int(m["gat_layers"]),
        dit_hidden=int(m["dit_hidden_dim"]),
        dit_layers=int(m["dit_layers"]),
        dit_heads=int(m["dit_heads"]),
        dit_dropout=float(m["dit_dropout"]),
        time_dim=int(m["time_embedding_dim"]),
        diffusion_steps=int(m["diffusion_train_steps"]),
        beta_start=float(m["beta_start"]),
        beta_end=float(m["beta_end"]),
        shape_clamp=float(m["shape_clamp"]),
        epsilon=float(m["epsilon"]),
        base_loss_weight=float(m["base_loss_weight"]),
        gate_bias_init=float(m["gate_bias_init"]),
    ).to(device)

    emb_t = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"])
    )
    checkpoint_dir = ROOT / o["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / o["training_log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = checkpoint_dir / "config_snapshot.json"
    metadata_path = checkpoint_dir / "run_metadata.json"
    config_snapshot_path.write_text(json.dumps({
        **cfg,
        "demand_scale": demand_scale,
        "normalization_note": "divide demand by training-SKU standard deviation; donor mean/std use all observations before each origin",
        "objective_note": "diffusion noise loss + base_loss_weight * base MSE; no auxiliary final-forecast gate loss",
        "sampler_note": "respaced stochastic DDPM posterior update as implemented in the architecture notebook",
    }, indent=2) + "\n", encoding="utf-8")

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        git_commit = None
    try:
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip())
    except Exception:
        git_dirty = None

    run_metadata = {
        "experiment_name": cfg["experiment"]["name"],
        "run_started_utc": run_started_utc,
        "seed": seed,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit,
        "git_dirty_at_start": git_dirty,
        "training_sku_count": len(train_skus),
        "training_skus": train_skus,
        "test_sku_count": len(test_skus),
        "test_skus": test_skus,
        "embedding_shape": list(embeddings.shape),
        "embedding_model": embed_index.get("model", "unknown"),
        "demand_scale": demand_scale,
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "config": cfg,
    }
    metadata_path.write_text(json.dumps(run_metadata, indent=2, default=str) + "\n", encoding="utf-8")

    history = []
    best_loss, best_epoch, stale = float("inf"), None, 0
    epochs_completed = 0
    for epoch in range(1, int(tcfg["epochs"]) + 1):
        epoch_started = time.time()
        model.train()
        totals = {"loss": 0.0, "diffusion_loss": 0.0, "base_loss": 0.0}
        batches = 0
        for b in loader:
            target_idx = b["target_idx"].long().to(device)
            donor_idx = b["donor_idx"].long().to(device)
            donor_hist = b["donor_history"].float().to(device)
            donor_mu = b["donor_mean"].float().to(device).unsqueeze(-1)
            donor_sd = b["donor_std"].float().to(device).unsqueeze(-1)
            donor_mask = b["donor_mask"].bool().to(device)
            y = b["target_values"].float().to(device)
            ymask = b["target_mask"].float().to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(
                emb_t, target_idx, donor_idx, donor_hist, donor_mu, donor_sd, donor_mask,
                target_values=y, target_mask=ymask
            )
            if not torch.isfinite(out["loss"]):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(tcfg["gradient_clip_norm"])
            )
            optimizer.step()
            for key in totals:
                totals[key] += float(out[key].detach().cpu())
            batches += 1

        row = {
            "epoch": epoch,
            "epoch_seconds": round(time.time() - epoch_started, 4),
            "elapsed_seconds": round(time.time() - run_started, 4),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "batches": batches,
            **{k: v / max(batches, 1) for k, v in totals.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(log_path, index=False)
        epochs_completed = epoch
        if epoch == 1 or epoch % 5 == 0:
            print(" ".join(
                f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()
            ))

        if row["loss"] < best_loss:
            best_loss, best_epoch, stale = row["loss"], epoch, 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "embedding_skus": embed_skus,
                "best_epoch": epoch,
                "best_loss": best_loss,
                "demand_scale": demand_scale,
                "architecture_version": "notebook_aligned_v1",
            }
            torch.save(checkpoint, checkpoint_dir / "raid_best.pt")
        else:
            stale += 1
        if stale >= int(tcfg["patience"]):
            print(f"Early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(log_path, index=False)
    total_seconds = time.time() - run_started
    run_metadata.update({
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "total_training_seconds": round(total_seconds, 3),
        "total_training_minutes": round(total_seconds / 60, 3),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_training_loss": float(best_loss),
        "training_log": str(log_path.relative_to(ROOT)),
        "best_checkpoint": str((checkpoint_dir / "raid_best.pt").relative_to(ROOT)),
    })
    metadata_path.write_text(json.dumps(run_metadata, indent=2, default=str) + "\n", encoding="utf-8")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "embedding_skus": embed_skus,
        "last_epoch": epochs_completed,
        "run_metadata": run_metadata,
        "optimizer_state_dict": optimizer.state_dict(),
        "training_history": history,
        "demand_scale": demand_scale,
        "architecture_version": "notebook_aligned_v1",
    }, checkpoint_dir / "raid_last.pt")
    print(f"Training time: {total_seconds / 60:.2f} min; epochs={epochs_completed}; best_epoch={best_epoch}")
    print(f"Best training loss={best_loss:.6f}; checkpoint={checkpoint_dir / 'raid_best.pt'}")
    print(f"Run metadata: {metadata_path}")
    print(f"Epoch log: {log_path}")


if __name__ == "__main__":
    main()
