"""Evaluate notebook-aligned RAID on held-out launch-aligned Intel SKU cases."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.raid import RAID

CFG_PATH = ROOT / "configs/intel_sku/raid_aggregate.json"


def metrics(y, pred):
    """Regression metrics for one aligned actual/prediction vector.

    NSE and KGE are undefined for degenerate windows (for example, constant
    actuals for NSE, or zero variance/mean for KGE). Undefined values are None,
    not zero, so they cannot be mistaken for poor-but-valid scores.
    """
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if y.shape != pred.shape or y.size == 0:
        raise ValueError("Actual and prediction arrays must be non-empty and aligned.")
    err = pred - y
    denom = np.abs(y) + np.abs(pred)
    smape = np.mean(np.divide(
        2 * np.abs(err), denom, out=np.zeros_like(denom), where=denom > 0
    ))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    wape = float(np.sum(np.abs(err)) / np.sum(np.abs(y))) if np.sum(np.abs(y)) > 0 else None

    nse_denom = float(np.sum((y - np.mean(y)) ** 2))
    nse = (float(1.0 - np.sum(err ** 2) / nse_denom)
           if nse_denom > 1e-12 else None)

    y_std, pred_std = float(np.std(y)), float(np.std(pred))
    y_mean, pred_mean = float(np.mean(y)), float(np.mean(pred))
    if y_std > 1e-12 and pred_std > 1e-12 and abs(y_mean) > 1e-12:
        corr = float(np.corrcoef(y, pred)[0, 1])
        alpha = pred_std / y_std
        beta = pred_mean / y_mean
        kge = float(1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
        if not np.isfinite(kge):
            kge = None
    else:
        kge = None

    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": float(smape),
        "WAPE": wape,
        "NSE": nse,
        "KGE": kge,
    }


def macro_mean_defined(values):
    """Mean over defined per-SKU metric values; return None if none are defined."""
    valid = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(valid)) if valid else None


def main():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    d, o, m = cfg["data"], cfg["outputs"], cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(ROOT / d["sku_metadata_path"])
    demand = pd.read_csv(ROOT / d["aggregate_demand_path"])
    cases = pd.read_csv(ROOT / d["cases_path"])

    split_dir = ROOT / o["split_dir"]
    train_skus = set(pd.read_csv(split_dir / "train_skus.csv")["sku"].astype(str))
    test_skus = set(pd.read_csv(split_dir / "test_skus.csv")["sku"].astype(str))
    cases = cases[cases["target_sku"].astype(str).isin(test_skus)].copy()
    if set(cases["target_sku"].astype(str)) != test_skus:
        raise ValueError("Test split does not match eligible evaluation cases.")

    embeddings = np.load(ROOT / d["embedding_path"]).astype(np.float32)
    index = json.loads((ROOT / d["embedding_index_path"]).read_text(encoding="utf-8"))
    embed_skus = [str(s) for s in index["skus"]]
    sku_to_idx = {s: i for i, s in enumerate(embed_skus)}
    checkpoint_path = ROOT / o["checkpoint_dir"] / "raid_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("embedding_skus") != embed_skus:
        raise ValueError("Embedding SKU order differs from the order saved in the checkpoint.")

    demand_scale = float(checkpoint["demand_scale"])
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    emb_t = torch.as_tensor(embeddings, dtype=torch.float32, device=device)

    meta_by_sku = metadata.set_index(d["sku_column"], drop=False)
    series = {
        str(s): g.set_index(d["week_column"])[d["target_column"]].sort_index()
        for s, g in demand.groupby(d["sku_column"])
    }
    h = int(cfg["experiment"]["forecast_window_weeks"])
    preds, per_sku = [], []

    for _, case in cases.sort_values(["origin_week", "target_sku"]).iterrows():
        target, origin = str(case["target_sku"]), int(case["origin_week"])
        if target not in test_skus:
            raise ValueError(f"Unexpected non-test target: {target}")
        hist_weeks = pd.RangeIndex(origin - h, origin)
        donors = []
        for sku in sorted(train_skus):
            if sku == target:
                continue
            if int(meta_by_sku.loc[sku, d["launch_column"]]) >= origin:
                continue
            donor_series = series[sku]
            history = donor_series.reindex(hist_weeks)
            if not history.notna().all():
                continue
            prior = donor_series.loc[donor_series.index < origin].dropna()
            if len(prior) < 2:
                continue
            prior_scaled = prior.to_numpy(dtype=np.float32) / demand_scale
            donor_mean = float(np.mean(prior_scaled))
            donor_std = float(np.std(prior_scaled))
            if not np.isfinite(donor_std) or donor_std < 1e-6:
                donor_std = 1e-6
            donors.append((
                sku, history.to_numpy(dtype=np.float32) / demand_scale,
                donor_mean, donor_std
            ))

        if not donors:
            raise RuntimeError(
                f"No training-split donor has {h} complete weeks before {target}'s origin {origin}. "
                "Do not impute missing weeks or use future observations; inspect this case/split."
            )

        target_vec = embeddings[sku_to_idx[target]]
        # Match the architecture notebook's embedding-dot-product donor ranking.
        sims = np.array([embeddings[sku_to_idx[s]] @ target_vec for s, _, _, _ in donors])
        order = np.argsort(-sims)[:min(int(m["neighbors_k"]), len(donors))]
        selected = [donors[i] for i in order]
        donor_names = [x[0] for x in selected]
        if target in donor_names:
            raise AssertionError("Self-donor leakage detected.")

        donor_idx = [sku_to_idx[s] for s, _, _, _ in selected]
        donor_history = torch.as_tensor(
            np.stack([x[1] for x in selected]), dtype=torch.float32, device=device
        )
        donor_mean = torch.as_tensor(
            [x[2] for x in selected], dtype=torch.float32, device=device
        )
        donor_std = torch.as_tensor(
            [x[3] for x in selected], dtype=torch.float32, device=device
        )
        samples = model.predict_samples(
            emb_t, sku_to_idx[target], donor_idx, donor_history, donor_mean, donor_std,
            num_samples=int(cfg["evaluation"]["num_samples"]),
            inference_steps=int(m["diffusion_inference_steps"]),
        ).cpu().numpy() * demand_scale
        point = samples.mean(axis=0)

        actual = series[target].reindex(pd.RangeIndex(origin, origin + h))
        if actual.isna().any():
            raise ValueError(f"Test target {target} lacks a complete {h}-week target window.")
        actual = actual.to_numpy(dtype=float)
        for j in range(h):
            preds.append({
                "sku": target, "origin_week": origin, "forecast_week": origin + j,
                "step": j + 1, "actual": float(actual[j]), "prediction": float(point[j]),
            })
        per_sku.append({
            "sku": target, "origin_week": origin, "donor_count": len(donor_names),
            "donors": ";".join(donor_names), **metrics(actual, point),
        })

    outdir = ROOT / o["prediction_dir"]
    outdir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(preds)
    per_sku_df = pd.DataFrame(per_sku)
    pred_df.to_csv(outdir / "raid_predictions.csv", index=False)
    per_sku_df.to_csv(outdir / "raid_per_sku_metrics.csv", index=False)
    # Flattened/pooled metrics preserve the previous aggregate scoring convention.
    # For NSE/KGE, also report macro averages over target-SKU windows: pooled scores
    # can be dominated by between-SKU scale differences and should not be the only view.
    overall = metrics(pred_df["actual"].to_numpy(), pred_df["prediction"].to_numpy())
    per_sku_nse = per_sku_df["NSE"].tolist()
    per_sku_kge = per_sku_df["KGE"].tolist()
    overall.update({
        "NSE_macro_mean_per_sku": macro_mean_defined(per_sku_nse),
        "KGE_macro_mean_per_sku": macro_mean_defined(per_sku_kge),
        "NSE_defined_sku_windows": int(per_sku_df["NSE"].notna().sum()),
        "NSE_undefined_sku_windows": int(per_sku_df["NSE"].isna().sum()),
        "KGE_defined_sku_windows": int(per_sku_df["KGE"].notna().sum()),
        "KGE_undefined_sku_windows": int(per_sku_df["KGE"].isna().sum()),
        "test_skus": int(per_sku_df["sku"].nunique()),
        "forecast_window_weeks": h,
        "protocol": "SKU-held-out cold-start simulation; not strict historical backtesting",
        "demand_scale": demand_scale,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "metric_note": "Pooled NSE/KGE use concatenated forecasts; macro NSE/KGE average defined per-SKU windows. Prefer macro values for across-SKU comparison.",
    })
    (outdir / "raid_metrics.json").write_text(
        json.dumps(overall, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(overall, indent=2))
    print(f"Predictions and metrics saved to {outdir}")


if __name__ == "__main__":
    main()
