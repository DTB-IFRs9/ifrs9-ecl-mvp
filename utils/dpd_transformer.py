"""
DPD Wide → Long Transformer

This module converts TTC PD datasets provided in wide format
(one column per month) into a clean, analysis‑ready long format.

This is the foundational transformer for PD, staging, and ECL models.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime


def transform_dpd_wide_to_long(file_path: str) -> pd.DataFrame:
    """
    Transform a wide-format DPD matrix into long-format loan performance data.

    Parameters
    ----------
    file_path : str
        Path to CSV or Excel TTC PD dataset.

    Returns
    -------
    pd.DataFrame
        Clean long-format dataframe with:
        [
            'loan_id',
            'customer_id',
            'product_type',
            'snapshot_date',
            'dpd',
            'origination_date',
            'mob'
        ]
    """

    # ------------------------------------------------------------------
    # 1. Load dataset (CSV or Excel)
    # ------------------------------------------------------------------
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Use CSV or Excel.")

    df.columns = df.columns.str.strip()

    # ------------------------------------------------------------------
    # 2. Identify identifier & month columns
    # ------------------------------------------------------------------
    # Treat first three columns as identifiers
    id_cols = list(df.columns[:3])
    id_mapping = {
        id_cols[0]: "customer_id",
        id_cols[1]: "loan_id",
        id_cols[2]: "product_type",
    }
    df = df.rename(columns=id_mapping)

    identifier_cols = list(id_mapping.values())

    # Month columns: JAN_18, OCT_21, 2019-01, etc.
    month_pattern = re.compile(
        r"^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\-_]?\d{2,4}$",
        re.IGNORECASE,
    )

    month_cols = [
        col for col in df.columns
        if col not in identifier_cols and month_pattern.match(col)
    ]

    if not month_cols:
        raise ValueError("No month columns detected in dataset.")

    # ------------------------------------------------------------------
    # 3. Melt wide → long
    # ------------------------------------------------------------------
    long_df = df.melt(
        id_vars=identifier_cols,
        value_vars=month_cols,
        var_name="snapshot_date",
        value_name="dpd",
    )

    # ------------------------------------------------------------------
    # 4. Clean DPD values
    # ------------------------------------------------------------------
    long_df["dpd"] = pd.to_numeric(
        long_df["dpd"].replace(r"^\s*$", np.nan, regex=True),
        errors="coerce"
    )

    # ------------------------------------------------------------------
    # 5. Convert snapshot_date → actual datetime (end of month)
    # ------------------------------------------------------------------
    def parse_snapshot_date(x: str) -> pd.Timestamp:
        """
        Convert strings like 'OCT_18' or 'JAN_21' → end-of-month datetime.
        """
        x = x.replace("-", "_").upper()
        try:
            dt = datetime.strptime(x, "%b_%y")
        except ValueError:
            try:
                dt = datetime.strptime(x, "%b_%Y")
            except ValueError:
                return pd.NaT
        return pd.Timestamp(dt) + pd.offsets.MonthEnd(0)

    long_df["snapshot_date"] = long_df["snapshot_date"].apply(parse_snapshot_date)

    # Drop rows with invalid snapshot dates
    long_df = long_df.dropna(subset=["snapshot_date"])

    # ------------------------------------------------------------------
    # 6. Identify origination month (first non-null DPD per loan)
    # ------------------------------------------------------------------
    long_df = long_df.sort_values(["loan_id", "snapshot_date"])

    origination_df = (
        long_df[long_df["dpd"].notna()]
        .groupby("loan_id", as_index=False)["snapshot_date"]
        .min()
        .rename(columns={"snapshot_date": "origination_date"})
    )

    long_df = long_df.merge(origination_df, on="loan_id", how="left")

    # ------------------------------------------------------------------
    # 7. Remove rows before origination (loan not yet booked)
    # ------------------------------------------------------------------
    long_df = long_df[
        (long_df["origination_date"].notna()) &
        (long_df["snapshot_date"] >= long_df["origination_date"])
    ]

    # ------------------------------------------------------------------
    # 8. Compute MOB (Months on Book)
    # ------------------------------------------------------------------
    long_df["mob"] = (
        (long_df["snapshot_date"].dt.year - long_df["origination_date"].dt.year) * 12
        + (long_df["snapshot_date"].dt.month - long_df["origination_date"].dt.month)
    )

    long_df["mob"] = long_df["mob"].astype(int)

    # ------------------------------------------------------------------
    # 9. Final column ordering
    # ------------------------------------------------------------------
    final_cols = [
        "loan_id",
        "customer_id",
        "product_type",
        "snapshot_date",
        "dpd",
        "origination_date",
        "mob",
    ]

    return long_df[final_cols].reset_index(drop=True)


def compute_default_events(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute loan-level default events and censoring indicators.

    Parameters
    ----------
    df_long : pd.DataFrame
        Output of transform_dpd_wide_to_long() with columns:
        [
            'loan_id', 'customer_id', 'product_type',
            'snapshot_date', 'dpd', 'origination_date', 'mob'
        ]

    Returns
    -------
    pd.DataFrame
        Loan-level default table with:
        [
            'loan_id',
            'origination_date',
            'first_default_date',
            'default_flag',
            'censored',
            'time_to_default'
        ]
    """

    # -------------------------------------------------------------
    # 1. Define default condition
    # -------------------------------------------------------------
    df = df_long.copy()
    df["is_default"] = df["dpd"] >= 90

    # -------------------------------------------------------------
    # 2. First default date per loan (if any)
    # -------------------------------------------------------------
    first_default = (
        df[df["is_default"]]
        .groupby("loan_id", as_index=False)["snapshot_date"]
        .min()
        .rename(columns={"snapshot_date": "first_default_date"})
    )

    # -------------------------------------------------------------
    # 3. Origination date per loan
    # -------------------------------------------------------------
    origination = (
        df.groupby("loan_id", as_index=False)["origination_date"]
        .first()
    )

    # -------------------------------------------------------------
    # 4. Last observed snapshot per loan (for censored loans)
    # -------------------------------------------------------------
    last_snapshot = (
        df.groupby("loan_id", as_index=False)["snapshot_date"]
        .max()
        .rename(columns={"snapshot_date": "last_snapshot_date"})
    )

    # -------------------------------------------------------------
    # 5. Merge loan-level information
    # -------------------------------------------------------------
    loans = (
        origination
        .merge(first_default, on="loan_id", how="left")
        .merge(last_snapshot, on="loan_id", how="left")
    )

    # -------------------------------------------------------------
    # 6. Default flag & censoring
    # -------------------------------------------------------------
    loans["default_flag"] = loans["first_default_date"].notna().astype(int)
    loans["censored"] = (1 - loans["default_flag"]).astype(int)

    # -------------------------------------------------------------
    # 7. Time to default (or censoring)
    #    Measured in whole months
    # -------------------------------------------------------------
    def months_diff(start, end):
        return (
            (end.dt.year - start.dt.year) * 12
            + (end.dt.month - start.dt.month)
        )

    loans["time_to_default"] = np.where(
        loans["default_flag"] == 1,
        months_diff(loans["origination_date"], loans["first_default_date"]),
        months_diff(loans["origination_date"], loans["last_snapshot_date"]),
    ).astype(int)

    # -------------------------------------------------------------
    # 8. Final column selection
    # -------------------------------------------------------------
    return loans[
        [
            "loan_id",
            "origination_date",
            "first_default_date",
            "default_flag",
            "censored",
            "time_to_default",
        ]
    ].reset_index(drop=True)