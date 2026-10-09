
"""Preprocess and validate the Intel SKU demand dataset."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths and schema
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "Intel_SKU" / "raw"
PROCESSED_DIR = ROOT / "data" / "Intel_SKU" / "processed"

AGGREGATE_DIR = PROCESSED_DIR / "aggregate"
CENTER_DIR = PROCESSED_DIR / "by_distribution_center"
REPORT_DIR = ROOT / "artifacts" / "Intel_SKU" / "preprocessing"

SOURCE_FILE = RAW_DIR / "msom.2020.0933.csv"
DEMAND_FILE = RAW_DIR / "Intel_actual_demand.csv"
LIFECYCLE_FILE = RAW_DIR / "startend_date.csv"

STATIC_ATTRIBUTES = ["Product Offering", "Generation", "ASP Group"]
TARGET = "Customer Orders"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sku_key(value):
    """Create a punctuation-insensitive key for matching SKU IDs."""
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def standardize_sku(value):
    """Standardize display IDs while preserving their readable format."""
    return str(value).strip().replace(".", "-").upper()


def require_columns(df, columns, name):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def check_unique(df, columns, name):
    duplicates = df.duplicated(columns, keep=False)
    if duplicates.any():
        examples = df.loc[duplicates, columns].head(10)
        raise ValueError(
            f"{name} contains duplicate keys {columns}:\n{examples}"
        )


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------
# Load raw datasets
# ---------------------------------------------------------------------

def main():
    for path in [SOURCE_FILE, DEMAND_FILE, LIFECYCLE_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required raw file not found: {path}")

    source = pd.read_csv(SOURCE_FILE)
    demand_wide = pd.read_csv(DEMAND_FILE)
    lifecycle = pd.read_csv(LIFECYCLE_FILE)

    require_columns(
        source,
        [
            "Distribution Center",
            "Product Offering",
            "Generation",
            "SKU",
            "ASP Group",
            "Week",
            "Forecasted Demand",
            TARGET,
        ],
        "Original source",
    )
    require_columns(demand_wide, ["Week"], "Aggregate demand")
    require_columns(lifecycle, ["sku", "start", "end"], "Lifecycle metadata")

    # Use numeric week indices consistently.
    for df, col, name in [
        (source, "Week", "Original source"),
        (demand_wide, "Week", "Aggregate demand"),
        (lifecycle, "start", "Lifecycle metadata"),
        (lifecycle, "end", "Lifecycle metadata"),
    ]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if (
        source["Week"].isna().any()
        or demand_wide["Week"].isna().any()
        or lifecycle[["start", "end"]].isna().any().any()
    ):
        raise ValueError("Week or lifecycle boundaries contain missing values.")

    if not np.all(source["Week"] == source["Week"].astype(int)):
        raise ValueError("Source Week contains non-integer values.")
    if not np.all(demand_wide["Week"] == demand_wide["Week"].astype(int)):
        raise ValueError("Aggregate Week contains non-integer values.")
    if not np.all(lifecycle[["start", "end"]].to_numpy()
                  == lifecycle[["start", "end"]].to_numpy().astype(int)):
        raise ValueError("Lifecycle boundaries must be integer week indices.")

    source["Week"] = source["Week"].astype(int)
    demand_wide["Week"] = demand_wide["Week"].astype(int)
    lifecycle[["start", "end"]] = lifecycle[["start", "end"]].astype(int)

    if source[["Distribution Center", "SKU", "Week"]].isna().any().any():
        raise ValueError("Source keys contain missing values.")

    if demand_wide["Week"].duplicated().any():
        raise ValueError("Aggregate demand contains duplicate weeks.")

    if lifecycle["sku"].isna().any():
        raise ValueError("Lifecycle metadata contains missing SKU IDs.")

    # -----------------------------------------------------------------
    # Standardize identifiers and verify SKU mappings
    # -----------------------------------------------------------------

    demand_skus = [c for c in demand_wide.columns if c != "Week"]

    demand_sku_map = {
        sku_key(sku): standardize_sku(sku) for sku in demand_skus
    }
    if len(demand_sku_map) != len(demand_skus):
        raise ValueError("Aggregate demand has colliding normalized SKU IDs.")

    source["sku_key"] = source["SKU"].map(sku_key)
    lifecycle["sku_key"] = lifecycle["sku"].map(sku_key)

    if lifecycle["sku_key"].duplicated().any():
        raise ValueError("Lifecycle metadata has duplicate normalized SKUs.")

    demand_keys = set(demand_sku_map)
    source_keys = set(source["sku_key"])
    lifecycle_keys = set(lifecycle["sku_key"])

    if demand_keys != source_keys:
        raise ValueError(
            "SKU mismatch between aggregate demand and source. "
            f"Only in demand: {sorted(demand_keys - source_keys)}; "
            f"only in source: {sorted(source_keys - demand_keys)}"
        )

    if demand_keys != lifecycle_keys:
        raise ValueError(
            "SKU mismatch between demand and lifecycle metadata. "
            f"Only in demand: {sorted(demand_keys - lifecycle_keys)}; "
            f"only in lifecycle: {sorted(lifecycle_keys - demand_keys)}"
        )

    source["sku"] = source["sku_key"].map(demand_sku_map)
    lifecycle["sku"] = lifecycle["sku_key"].map(demand_sku_map)

    # -----------------------------------------------------------------
    # Validate source records and static metadata
    # -----------------------------------------------------------------

    source_key_columns = ["Distribution Center", "sku", "Week"]
    check_unique(source, source_key_columns, "Original source")

    if source[[TARGET, "Forecasted Demand"]].isna().any().any():
        raise ValueError("Source demand/forecast columns contain missing values.")

    if (source[[TARGET, "Forecasted Demand"]] < 0).any().any():
        raise ValueError("Source demand/forecast contains negative values.")

    if source[STATIC_ATTRIBUTES].isna().any().any():
        raise ValueError("Static product attributes contain missing values.")

    attribute_counts = source.groupby("sku")[STATIC_ATTRIBUTES].nunique()
    if (attribute_counts > 1).any().any():
        raise ValueError("Static attributes are inconsistent within a SKU.")

    metadata = (
        source[["sku", *STATIC_ATTRIBUTES]]
        .drop_duplicates()
        .merge(
            lifecycle[["sku", "start", "end"]],
            on="sku",
            how="left",
            validate="one_to_one",
        )
        .sort_values("sku")
        .reset_index(drop=True)
    )

    if metadata[["start", "end"]].isna().any().any():
        raise ValueError("Some SKUs have missing lifecycle boundaries.")

    if (metadata["start"] > metadata["end"]).any():
        raise ValueError("Some lifecycle start weeks exceed end weeks.")

    # -----------------------------------------------------------------
    # Prepare aggregate demand in long format
    # -----------------------------------------------------------------

    aggregate_long = demand_wide.melt(
        id_vars="Week",
        var_name="sku_original",
        value_name="customer_orders",
    )

    aggregate_long["sku"] = aggregate_long["sku_original"].map(
        lambda x: demand_sku_map[sku_key(x)]
    )
    aggregate_long = aggregate_long.rename(columns={"Week": "week"})
    aggregate_long = aggregate_long[["sku", "week", "customer_orders"]]
    aggregate_long = aggregate_long.sort_values(
        ["sku", "week"]
    ).reset_index(drop=True)

    if aggregate_long.duplicated(["sku", "week"]).any():
        raise ValueError("Aggregate long table has duplicate SKU-week keys.")

    # Missing aggregate values should not occur inside the recorded
    # lifecycle, according to the exploratory checks.
    aggregate_long = aggregate_long.merge(
        metadata[["sku", "start", "end"]],
        on="sku",
        how="left",
        validate="many_to_one",
    )

    inside_lifecycle = aggregate_long["week"].between(
        aggregate_long["start"], aggregate_long["end"]
    )
    missing_inside_lifecycle = (
        inside_lifecycle & aggregate_long["customer_orders"].isna()
    )

    if missing_inside_lifecycle.any():
        examples = aggregate_long.loc[
            missing_inside_lifecycle, ["sku", "week"]
        ].head(10)
        raise ValueError(
            "Missing aggregate demand found inside recorded lifecycle:\n"
            f"{examples}"
        )

    aggregate_long = aggregate_long.drop(columns=["start", "end"])

    # -----------------------------------------------------------------
    # Verify aggregate Customer Orders against original source
    # -----------------------------------------------------------------

    source_aggregate = (
        source.groupby(["sku", "Week"], as_index=False)[TARGET]
        .sum()
        .rename(columns={"Week": "week", TARGET: "source_customer_orders"})
    )

    observed_aggregate = aggregate_long.loc[
        aggregate_long["customer_orders"].notna()
    ].copy()

    comparison = observed_aggregate.merge(
        source_aggregate,
        on=["sku", "week"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    missing_source_record_count = int(
        (comparison["_merge"] == "left_only").sum()
    )

    # This is a comparison convention only. It does NOT impute the
    # aggregate dataset or the center-level dataset.
    comparison["source_customer_orders"] = (
        comparison["source_customer_orders"].fillna(0)
    )
    comparison["match"] = np.isclose(
        comparison["customer_orders"],
        comparison["source_customer_orders"],
    )

    mismatch_count = int((~comparison["match"]).sum())
    if mismatch_count:
        examples = comparison.loc[
            ~comparison["match"],
            [
                "sku",
                "week",
                "customer_orders",
                "source_customer_orders",
            ],
        ].head(20)
        raise ValueError(
            f"Found {mismatch_count} aggregate/source mismatches:\n{examples}"
        )

    # -----------------------------------------------------------------
    # Prepare distribution-center-level observations
    # -----------------------------------------------------------------

    observed = source.rename(
        columns={
            "Distribution Center": "distribution_center",
            "Week": "week",
            "Forecasted Demand": "forecasted_demand",
            TARGET: "customer_orders",
            "Product Offering": "product_offering",
            "Generation": "generation",
            "ASP Group": "asp_group",
        }
    ).copy()

    observed["source_record_present"] = True

    observed = observed[
        [
            "distribution_center",
            "sku",
            "week",
            "customer_orders",
            "forecasted_demand",
            "product_offering",
            "generation",
            "asp_group",
            "source_record_present",
        ]
    ].sort_values(
        ["distribution_center", "sku", "week"]
    ).reset_index(drop=True)

    # -----------------------------------------------------------------
    # Build a complete center × SKU × week panel without imputing
    # -----------------------------------------------------------------

    centers = sorted(source["Distribution Center"].unique())
    skus = sorted(metadata["sku"].unique())
    weeks = sorted(demand_wide["Week"].unique())

    full_panel = pd.MultiIndex.from_product(
        [centers, skus, weeks],
        names=["distribution_center", "sku", "week"],
    ).to_frame(index=False)

    panel = full_panel.merge(
        observed,
        on=["distribution_center", "sku", "week"],
        how="left",
        validate="one_to_one",
    )

    panel = panel.merge(
        metadata.rename(
            columns={
                "Product Offering": "product_offering",
                "Generation": "generation",
                "ASP Group": "asp_group",
            }
        ),
        on="sku",
        how="left",
        suffixes=("", "_metadata"),
        validate="many_to_one",
    )

    # Populate attributes from metadata for absent source rows.
    for col in ["product_offering", "generation", "asp_group"]:
        metadata_col = f"{col}_metadata"
        panel[col] = panel[col].fillna(panel[metadata_col])
        panel = panel.drop(columns=metadata_col)

    panel["source_record_present"] = (
        panel["source_record_present"].eq(True)
    )

    panel["start"] = panel["start"].astype(int)
    panel["end"] = panel["end"].astype(int)

    inside_lifecycle = panel["week"].between(panel["start"], panel["end"])

    panel["record_status"] = np.select(
        [
            panel["source_record_present"],
            ~inside_lifecycle,
        ],
        [
            "observed",
            "outside_lifecycle",
        ],
        default="missing_within_lifecycle",
    )

    panel = panel.sort_values(
        ["distribution_center", "sku", "week"]
    ).reset_index(drop=True)

    
# -----------------------------------------------------------------
# Save processed datasets
# -----------------------------------------------------------------

    save_csv(
        aggregate_long,
        AGGREGATE_DIR / "aggregate_sku_weekly_demand.csv",
    )
    save_csv(
        metadata,
        AGGREGATE_DIR / "sku_metadata.csv",
    )
    save_csv(
        observed,
        CENTER_DIR / "distribution_center_observations.csv",
    )
    save_csv(
        panel,
        CENTER_DIR / "distribution_center_weekly_panel.csv",
    )


    # -----------------------------------------------------------------
    # Save a reproducible validation summary
    # -----------------------------------------------------------------

    summary = {
        "source_rows": int(len(source)),
        "source_skus": int(source["sku"].nunique()),
        "distribution_centers": centers,
        "aggregate_weeks": int(demand_wide["Week"].nunique()),
        "aggregate_cells": int(len(aggregate_long)),
        "aggregate_non_missing_cells": int(
            aggregate_long["customer_orders"].notna().sum()
        ),
        "aggregate_missing_cells": int(
            aggregate_long["customer_orders"].isna().sum()
        ),
        "aggregate_zero_cells": int(
            (aggregate_long["customer_orders"] == 0).sum()
        ),
        "aggregate_source_records_matched": int(
            (comparison["_merge"] == "both").sum()
        ),
        "aggregate_cells_without_source_record": (
            missing_source_record_count
        ),
        "aggregate_mismatches": mismatch_count,
        "center_observed_rows": int(len(observed)),
        "center_panel_rows": int(len(panel)),
        "center_panel_observed_rows": int(
            panel["source_record_present"].sum()
        ),
        "center_panel_missing_within_lifecycle": int(
            (panel["record_status"] == "missing_within_lifecycle").sum()
        ),
        "center_panel_outside_lifecycle": int(
            (panel["record_status"] == "outside_lifecycle").sum()
        ),
        "static_attributes_consistent": True,
        "negative_demand_or_forecast_values": 0,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "validation_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nPreprocessing completed successfully.")
    print(json.dumps(summary, indent=2))
    print(f"\nProcessed data saved under: {PROCESSED_DIR}")
    print(f"Validation report saved under: {REPORT_DIR}")


if __name__ == "__main__":
    main()