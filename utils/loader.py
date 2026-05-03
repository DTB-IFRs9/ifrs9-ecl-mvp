"""
Loan‑level dataset assembly utilities.

This module assembles clean, one‑row‑per‑loan tables from
long‑format behavioral data and default events.
"""

import pandas as pd
import numpy as np


def build_loan_summary_table(
    df_long: pd.DataFrame,
    df_default: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a loan‑level summary table combining latest behavior
    and default / censoring information.

    Parameters
    ----------
    df_long : pd.DataFrame
        Long‑format DPD dataframe with columns:
        [
            'loan_id', 'customer_id', 'product_type',
            'snapshot_date', 'dpd', 'origination_date', 'mob'
        ]

    df_default : pd.DataFrame
        Loan‑level default table from compute_default_events() with columns:
        [
            'loan_id', 'origination_date', 'first_default_date',
            'default_flag', 'censored', 'time_to_default'
        ]

    Returns
    -------
    pd.DataFrame
        One‑row‑per‑loan summary table with:
        [
          'loan_id',
          'customer_id',
          'product_type',
          'origination_date',
          'last_snapshot_date',
          'current_dpd',
          'mob',
          'default_flag',
          'censored',
          'time_to_default',
          'is_active'
        ]
    """

    # ---------------------------------------------------------
    # 1. Identify latest snapshot per loan
    # ---------------------------------------------------------
    last_snapshot = (
        df_long
        .groupby("loan_id", as_index=False)["snapshot_date"]
        .max()
        .rename(columns={"snapshot_date": "last_snapshot_date"})
    )

    # ---------------------------------------------------------
    # 2. Extract latest DPD and MOB at that snapshot
    # ---------------------------------------------------------
    latest_behavior = (
        df_long
        .merge(last_snapshot, on="loan_id", how="inner")
        .loc[lambda x: x["snapshot_date"] == x["last_snapshot_date"]]
        [
            [
                "loan_id",
                "customer_id",
                "product_type",
                "origination_date",
                "last_snapshot_date",
                "dpd",
                "mob",
            ]
        ]
        .rename(columns={"dpd": "current_dpd"})
    )

    # Defensive: ensure one row per loan
    latest_behavior = (
        latest_behavior
        .drop_duplicates(subset=["loan_id"])
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 3. Merge default / censoring information
    # ---------------------------------------------------------
    loan_summary = latest_behavior.merge(
        df_default[
            [
                "loan_id",
                "default_flag",
                "censored",
                "time_to_default",
            ]
        ],
        on="loan_id",
        how="left",
    )

    # ---------------------------------------------------------
    # 4. Active / inactive flag
    # ---------------------------------------------------------
    # A loan is active if it has NOT defaulted
    loan_summary["is_active"] = (
        loan_summary["default_flag"] == 0
    ).astype(int)

    # ---------------------------------------------------------
    # 5. Final column ordering
    # ---------------------------------------------------------
    final_cols = [
        "loan_id",
        "customer_id",
        "product_type",
        "origination_date",
        "last_snapshot_date",
        "current_dpd",
        "mob",
        "default_flag",
        "censored",
        "time_to_default",
        "is_active",
    ]

    return loan_summary[final_cols].reset_index(drop=True)