"""Make the requested SKU-held-out split. Run from repository root."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "configs/intel_sku/raid_aggregate.json").read_text())
D = CFG["data"]
OUT = ROOT / CFG["outputs"]["split_dir"]

def main():
    cases = pd.read_csv(ROOT / D["cases_path"])
    exclusions = pd.read_csv(ROOT / D["exclusions_path"])
    metadata = pd.read_csv(ROOT / D["sku_metadata_path"])
    all_skus = set(metadata[D["sku_column"]].astype(str))
    eligible = sorted(set(cases["target_sku"].astype(str)))
    excluded = sorted(set(exclusions["target_sku"].astype(str)))
    if set(eligible) & set(excluded):
        raise ValueError("Eligible and excluded SKU sets overlap.")
    if set(eligible) | set(excluded) != all_skus:
        raise ValueError("Cases plus exclusions do not cover the metadata SKU set.")
    n_test = int(CFG["experiment"]["test_eligible_skus"])
    if not 0 < n_test < len(eligible):
        raise ValueError(f"test_eligible_skus must be between 1 and {len(eligible)-1}.")
    rng = np.random.default_rng(int(CFG["experiment"]["random_seed"]))
    test = sorted(rng.choice(np.array(eligible, dtype=object), size=n_test, replace=False).tolist())
    train_eligible = sorted(set(eligible) - set(test))
    train = sorted(set(excluded) | set(train_eligible))
    rows = [{"sku": s, "split": "train",
             "split_reason": "evaluation_ineligible_training_pool" if s in set(excluded)
                             else "eligible_sku_selected_for_training"} for s in train]
    rows += [{"sku": s, "split": "test", "split_reason": "eligible_held_out_target"} for s in test]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["split", "sku"]).to_csv(OUT / "sku_split.csv", index=False)
    pd.DataFrame({"sku": train}).to_csv(OUT / "train_skus.csv", index=False)
    pd.DataFrame({"sku": test}).to_csv(OUT / "test_skus.csv", index=False)
    summary = {
      "seed": int(CFG["experiment"]["random_seed"]), "total_skus": len(all_skus),
      "evaluation_ineligible_training_skus": len(excluded), "eligible_training_skus": len(train_eligible),
      "training_skus_total": len(train), "test_skus": len(test), "train_skus": train, "test_skus_list": test,
      "protocol": "SKU-held-out cold-start simulation, not strict historical backtesting",
      "caveat": "Training labels can occur after some test launch origins. For a strict historical claim, use origin-specific training cutoffs."
    }
    (OUT / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"train_skus","test_skus_list"}}, indent=2))
    print(f"Saved split to {OUT}")

if __name__ == "__main__":
    main()
