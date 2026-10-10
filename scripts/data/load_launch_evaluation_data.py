"""Load launch-aligned aggregate SKU cold-start evaluation data.

The configuration controls dataset paths, forecast-window length, eligibility
rules, retrieval metadata, and output artifact names.

Run from the repository root:
    python scripts/data/load_launch_evaluation_data.py
    python scripts/data/load_launch_evaluation_data.py --config configs/intel_sku/aggregate_launch_evaluation.json

The script writes an evaluation-case table, an exclusion table, and a summary.
Target actuals are used only to decide whether a case has a complete forecast
window; they are not included in donor-history data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_config(config_path: Path) -> dict[str, Any]:
    """Read and validate the JSON configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required_sections = {"experiment", "data", "evaluation", "outputs"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    window = config["experiment"].get("forecast_window_weeks")
    if not isinstance(window, int) or window < 1:
        raise ValueError("experiment.forecast_window_weeks must be a positive integer")

    evaluation = config["evaluation"]
    supported_rules = {
        "donor_launch_rule": {"strictly_before_origin"},
        "history_rule": {"week_strictly_before_origin"},
    }
    for key, allowed in supported_rules.items():
        value = evaluation.get(key)
        if value not in allowed:
            raise ValueError(f"Unsupported evaluation rule {key}={value!r}; expected one of {sorted(allowed)}")

    if evaluation.get("use_lifecycle_end_as_donor_filter", False):
        raise ValueError(
            "Lifecycle end must not be used to filter donor SKUs because it can leak future information."
        )

    return config


def resolve_repo_path(repo_root: Path, configured_path: str) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(configured_path)
    return path if path.is_absolute() else repo_root / path


def load_inputs(config: dict[str, Any], repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate aggregate weekly demand and SKU metadata."""
    data_cfg = config["data"]
    demand_path = resolve_repo_path(repo_root, data_cfg["aggregate_demand_path"])
    metadata_path = resolve_repo_path(repo_root, data_cfg["sku_metadata_path"])

    if not demand_path.exists():
        raise FileNotFoundError(f"Aggregate demand file not found: {demand_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"SKU metadata file not found: {metadata_path}")

    demand = pd.read_csv(demand_path)
    metadata = pd.read_csv(metadata_path)

    sku_col = data_cfg["sku_column"]
    week_col = data_cfg["week_column"]
    target_col = config["experiment"]["target_column"]
    launch_col = data_cfg["launch_week_column"]
    feature_cols = data_cfg["retrieval_feature_columns"]

    required_demand = {sku_col, week_col, target_col}
    required_metadata = {sku_col, launch_col, *feature_cols}
    if missing := required_demand - set(demand.columns):
        raise ValueError(f"Aggregate demand file is missing columns: {sorted(missing)}")
    if missing := required_metadata - set(metadata.columns):
        raise ValueError(f"SKU metadata file is missing columns: {sorted(missing)}")

    demand[week_col] = pd.to_numeric(demand[week_col], errors="raise").astype(int)
    metadata[launch_col] = pd.to_numeric(metadata[launch_col], errors="raise").astype(int)

    if demand.duplicated([sku_col, week_col]).any():
        raise ValueError(f"Duplicate SKU-week rows found in {demand_path}")
    if metadata[sku_col].duplicated().any():
        raise ValueError(f"Duplicate SKU rows found in {metadata_path}")
    if set(demand[sku_col].dropna()) != set(metadata[sku_col].dropna()):
        raise ValueError("SKU sets differ between aggregate demand and SKU metadata")

    demand[target_col] = pd.to_numeric(demand[target_col], errors="raise")
    # Missing values are retained as missing; never convert them to zero here.
    return demand, metadata


def build_evaluation_cases(
    demand: pd.DataFrame,
    metadata: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create launch-aligned evaluation cases and explain excluded cases."""
    data_cfg = config["data"]
    exp_cfg = config["experiment"]
    eval_cfg = config["evaluation"]

    sku_col = data_cfg["sku_column"]
    week_col = data_cfg["week_column"]
    target_col = exp_cfg["target_column"]
    launch_col = data_cfg["launch_week_column"]
    window = exp_cfg["forecast_window_weeks"]
    min_week, max_week = int(demand[week_col].min()), int(demand[week_col].max())

    series_by_sku = {
        sku: group.set_index(week_col)[target_col].sort_index()
        for sku, group in demand.groupby(sku_col, sort=False)
    }
    metadata_by_sku = metadata.set_index(sku_col, drop=False)

    cases: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for target_sku in sorted(metadata[sku_col].tolist()):
        origin = int(metadata_by_sku.loc[target_sku, launch_col])
        forecast_end = origin + window - 1
        forecast_weeks = pd.RangeIndex(origin, forecast_end + 1)
        target_series = series_by_sku[target_sku]
        actuals = target_series.reindex(forecast_weeks)
        actual_weeks_available = int(actuals.notna().sum())

        reasons: list[str] = []
        if eval_cfg.get("require_prelaunch_history", True) and origin <= min_week:
            reasons.append("no_prelaunch_history")
        if forecast_end > max_week:
            reasons.append("forecast_window_exceeds_dataset")
        if (
            eval_cfg.get("require_complete_forecast_window", True)
            and actual_weeks_available != window
        ):
            reasons.append("incomplete_target_forecast_window")

        # Candidate eligibility uses launch metadata only as of the target origin.
        donor_metadata = metadata.loc[metadata[launch_col] < origin]
        if eval_cfg.get("exclude_target_from_donors", True):
            donor_metadata = donor_metadata.loc[donor_metadata[sku_col] != target_sku]

        donor_skus: list[str] = []
        min_donor_obs = int(eval_cfg.get("minimum_donor_history_observations", 1))
        for donor_sku in donor_metadata[sku_col].tolist():
            donor_history = series_by_sku[donor_sku]
            prior_history = donor_history.loc[donor_history.index < origin]
            if int(prior_history.notna().sum()) >= min_donor_obs:
                donor_skus.append(donor_sku)

        record = {
            "target_sku": target_sku,
            "origin_week": origin,
            "forecast_start_week": origin,
            "forecast_end_week": forecast_end,
            "forecast_window_weeks": window,
            "prior_launched_sku_count": int((metadata[launch_col] < origin).sum()),
            "candidate_sku_count": len(donor_skus),
            "target_actual_weeks_available": actual_weeks_available,
        }

        if reasons:
            exclusions.append({**record, "exclusion_reason": ";".join(reasons)})
        else:
            cases.append(record)

    return pd.DataFrame(cases), pd.DataFrame(exclusions)


def write_artifacts(
    cases: pd.DataFrame,
    exclusions: pd.DataFrame,
    demand: pd.DataFrame,
    metadata: pd.DataFrame,
    config: dict[str, Any],
    repo_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Write evaluation case tables and a machine-readable summary."""
    output_cfg = config["outputs"]
    output_dir = resolve_repo_path(repo_root, output_cfg["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cases.to_csv(output_dir / output_cfg["cases_file"], index=False)
    exclusions.to_csv(output_dir / output_cfg["exclusions_file"], index=False)

    week_col = config["data"]["week_column"]
    sku_col = config["data"]["sku_column"]
    reasons = (
        exclusions["exclusion_reason"].str.split(";").explode().value_counts().to_dict()
        if not exclusions.empty
        else {}
    )
    summary = {
        "protocol": config["experiment"]["name"],
        "setting": config["experiment"]["setting"],
        "forecast_window_weeks": config["experiment"]["forecast_window_weeks"],
        "dataset_week_range": [
            int(demand[week_col].min()),
            int(demand[week_col].max()),
        ],
        "total_skus": int(metadata[sku_col].nunique()),
        "eligible_target_skus": int(len(cases)),
        "excluded_target_skus": int(len(exclusions)),
        "unique_eligible_origins": int(cases["origin_week"].nunique()) if not cases.empty else 0,
        "exclusion_reason_counts": reasons,
        "config_path": str(config_path.relative_to(repo_root)) if config_path.is_relative_to(repo_root) else str(config_path),
        "artifacts": {
            "cases": output_cfg["cases_file"],
            "exclusions": output_cfg["exclusions_file"],
        },
        "notes": [
            "A target SKU's own demand is not used as model input.",
            "Donor SKUs must have launched strictly before the target origin.",
            "Donor histories include only weeks strictly before the target origin.",
            "Missing demand remains missing; no missing values are imputed to zero.",
            "Lifecycle end is not used to filter donors.",
            "Cases require a complete observed target forecast window when configured.",
            "Exclusion reason counts can sum to more than excluded_target_skus because a case can have multiple reasons.",
        ],
    }
    with (output_dir / output_cfg["summary_file"]).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    return summary


def load_launch_evaluation_data(
    config_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Reusable entry point for experiment code.

    Returns demand and metadata plus eligible cases/exclusions. Use
    ``get_donor_history`` below to retrieve leakage-safe histories for a case.
    """
    config_path = Path(config_path).resolve()
    if repo_root is None:
        # This script lives at <repo>/scripts/data/<script>.py.
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root).resolve()

    config = load_config(config_path)
    demand, metadata = load_inputs(config, repo_root)
    cases, exclusions = build_evaluation_cases(demand, metadata, config)
    summary = write_artifacts(cases, exclusions, demand, metadata, config, repo_root, config_path)
    return {
        "config": config,
        "demand": demand,
        "metadata": metadata,
        "cases": cases,
        "exclusions": exclusions,
        "summary": summary,
    }


def get_donor_history(
    target_sku: str,
    origin_week: int,
    data: dict[str, Any],
) -> pd.DataFrame:
    """Return donor-SKU observations strictly before origin, never target demand."""
    config = data["config"]
    data_cfg = config["data"]
    exp_cfg = config["experiment"]
    sku_col = data_cfg["sku_column"]
    week_col = data_cfg["week_column"]
    target_col = exp_cfg["target_column"]
    launch_col = data_cfg["launch_week_column"]
    features = data_cfg["retrieval_feature_columns"]

    metadata = data["metadata"]
    demand = data["demand"]
    eligible = metadata.loc[metadata[launch_col] < origin_week].copy()
    if config["evaluation"].get("exclude_target_from_donors", True):
        eligible = eligible.loc[eligible[sku_col] != target_sku]

    # Inner join with observations, then enforce the historical cutoff.
    history = demand.loc[demand[week_col] < origin_week].merge(
        eligible[[sku_col, *features]],
        on=sku_col,
        how="inner",
        validate="many_to_one",
    )
    history = history.loc[history[target_col].notna()].copy()
    return history.sort_values([sku_col, week_col]).reset_index(drop=True)


def get_target_actuals(
    target_sku: str,
    origin_week: int,
    data: dict[str, Any],
) -> pd.DataFrame:
    """Return target actuals for scoring only; do not pass these to model inputs."""
    config = data["config"]
    sku_col = config["data"]["sku_column"]
    week_col = config["data"]["week_column"]
    target_col = config["experiment"]["target_column"]
    window = int(config["experiment"]["forecast_window_weeks"])

    demand = data["demand"]
    mask = (
        demand[sku_col].eq(target_sku)
        & demand[week_col].between(origin_week, origin_week + window - 1)
    )
    return demand.loc[mask, [sku_col, week_col, target_col]].sort_values(week_col).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/intel_sku/aggregate_launch_evaluation.json",
        help="Path to the JSON evaluation configuration (relative to current working directory or absolute).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    # Script is expected to be run from repository root; allow config paths relative to root.
    repo_root = Path(__file__).resolve().parents[2]
    if not config_path.is_absolute():
        config_path = repo_root / args.config

    data = load_launch_evaluation_data(config_path, repo_root=repo_root)
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
