"""
LGD (Loss Given Default) Placeholder Module

This module implements a deliberately simple LGD component
for MVP and demo purposes.

Design choice:
- LGD is assumed to be a fixed constant (60%) for all loans.

This is:
- Common in early-stage IFRS 9 prototypes
- Easy to explain ("conservative assumption")
- Designed to be replaced later by a real LGD model
"""

import pandas as pd

# ---------------------------------------------------------
# Fixed LGD assumption for MVP
# ---------------------------------------------------------
DEFAULT_LGD = 0.60


def get_lgd(loan_summary_df: pd.DataFrame = None):
    """
    Return Loss Given Default (LGD).

    Parameters
    ----------
    loan_summary_df : pd.DataFrame, optional
        Loan-level summary dataframe (Task 5 output).
        Must contain 'loan_id' column if provided.

    Returns
    -------
    float or pd.Series
        - If loan_summary_df is None:
            returns DEFAULT_LGD as a scalar
        - If loan_summary_df is provided:
            returns a pandas Series of LGD values
            (all equal to DEFAULT_LGD) indexed by loan_id
    """

    # -----------------------------------------------------
    # Portfolio-level LGD (scalar)
    # -----------------------------------------------------
    if loan_summary_df is None:
        return DEFAULT_LGD

    # -----------------------------------------------------
    # Loan-level LGD (same value for all loans)
    # -----------------------------------------------------
    return pd.Series(
        DEFAULT_LGD,
        index=loan_summary_df["loan_id"],
        name="lgd",
    )