"""
EAD (Exposure at Default) Placeholder Module

This module implements a deliberately simple Exposure at Default (EAD)
component for MVP and demo purposes.

Design principles:
- Works even when no EAD file is provided
- Uses conservative, easy-to-explain assumptions
- Designed to be replaced later by a real EAD / CCF model
"""

import pandas as pd

# ---------------------------------------------------------
# Fixed MVP assumptions
# ---------------------------------------------------------
DEFAULT_EAD = 100_000          # Fixed exposure per loan (BDT)
DEFAULT_CCF = 0.50             # Credit Conversion Factor for undisbursed amount


def get_ead(loan_summary_df: pd.DataFrame, ead_file_path: str = None) -> pd.DataFrame:
    """
    Compute Exposure at Default (EAD).

    Parameters
    ----------
    loan_summary_df : pd.DataFrame
        Loan-level summary dataframe (Task 5 output).
        Must contain at least ['loan_id'].

    ead_file_path : str, optional
        Path to an Excel or CSV file containing EAD-related data.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly:
        ['loan_id', 'ead']

    Notes
    -----
    This is a placeholder MVP implementation.

    Case A — No EAD file:
        EAD = DEFAULT_EAD for all loans

    Case B — EAD file provided:
        EAD = Principal Outstanding
              + Interest Due
              + DEFAULT_CCF * Undisbursed Amount

        Missing values are treated as zero.
        Loans missing in the EAD file fall back to DEFAULT_EAD.
    """

    # -----------------------------------------------------
    # Case A: No EAD file provided
    # -----------------------------------------------------
    if ead_file_path is None:
        return pd.DataFrame({
            "loan_id": loan_summary_df["loan_id"],
            "ead": DEFAULT_EAD,
        })

    # -----------------------------------------------------
    # Case B: EAD file provided
    # -----------------------------------------------------
    if ead_file_path.lower().endswith(".csv"):
        ead_raw = pd.read_csv(ead_file_path)
    elif ead_file_path.lower().endswith((".xlsx", ".xls")):
        ead_raw = pd.read_excel(ead_file_path)
    else:
        raise ValueError("Unsupported EAD file type. Use CSV or Excel.")

    # -----------------------------------------------------
    # Standardize column names
    # -----------------------------------------------------
    ead_raw = ead_raw.rename(columns={
        "LAN": "loan_id",
        "Principal Outstanding": "principal_outstanding",
        "Interest Due": "interest_due",
        "Undibursed Amount": "undisbursed_amount",
    })

    # -----------------------------------------------------
    # Replace missing numeric values with zero
    # -----------------------------------------------------
    numeric_cols = [
        "principal_outstanding",
        "interest_due",
        "undisbursed_amount",
    ]

    for col in numeric_cols:
        if col in ead_raw.columns:
            ead_raw[col] = ead_raw[col].fillna(0)
        else:
            ead_raw[col] = 0

    # -----------------------------------------------------
    # Compute EAD using simple MVP formula
    # -----------------------------------------------------
    ead_raw["ead"] = (
        ead_raw["principal_outstanding"]
        + ead_raw["interest_due"]
        + DEFAULT_CCF * ead_raw["undisbursed_amount"]
    )

    ead_calculated = ead_raw[["loan_id", "ead"]]

    # -----------------------------------------------------
    # Merge with loan summary
    # Loans missing in EAD file → DEFAULT_EAD
    # -----------------------------------------------------
    ead_final = loan_summary_df[["loan_id"]].merge(
        ead_calculated,
        on="loan_id",
        how="left",
    )

    ead_final["ead"] = ead_final["ead"].fillna(DEFAULT_EAD)

    return ead_final