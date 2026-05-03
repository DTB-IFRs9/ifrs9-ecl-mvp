"""
ECL Calculator Module

This module combines PD, LGD, and EAD to compute Expected Credit Loss (ECL)
using simple IFRS‑9‑style staging logic.

Design principles:
- Clear and explainable calculations
- Placeholder‑friendly (PD engine + flat LGD + simple EAD)
- Modular and easy to extend later
"""

import pandas as pd
import numpy as np


class ECLCalculator:
    """
    Expected Credit Loss (ECL) Calculator.
    """

    def __init__(
        self,
        loan_summary_df: pd.DataFrame,
        pd_results: dict,
        lgd_df_or_scalar,
        ead_df: pd.DataFrame,
    ):
        """
        Parameters
        ----------
        loan_summary_df : pd.DataFrame
            Must contain at least:
            ['loan_id', 'current_dpd', 'product_type']

        pd_results : dict
            Output from PDEngine.run(), containing:
            {
              'pd_12m': float,
              'pd_lifetime': float,
              'loan_level_pd': DataFrame or None
            }

        lgd_df_or_scalar :
            Either:
              - scalar LGD value, or
              - DataFrame with ['loan_id', 'lgd']

        ead_df : pd.DataFrame
            DataFrame with ['loan_id', 'ead']
        """
        self.loan_summary = loan_summary_df.copy()
        self.pd_results = pd_results
        self.lgd_input = lgd_df_or_scalar
        self.ead_df = ead_df.copy()

        self._loan_level_ecl = None

    # ---------------------------------------------------------
    # Loan-level ECL calculation
    # ---------------------------------------------------------
    def calculate_loan_level_ecl(self) -> pd.DataFrame:
        """
        Calculate loan-level ECL using simple IFRS-9 logic.

        Returns
        -------
        pd.DataFrame
            [
                'loan_id',
                'stage',
                'pd_used',
                'lgd',
                'ead',
                'ecl'
            ]
        """

        df = self.loan_summary.copy()

        # -----------------------------------------------------
        # 1. Stage assignment based on current DPD
        # -----------------------------------------------------
        df["stage"] = np.where(
            df["current_dpd"] >= 90,
            "Stage 3",
            np.where(
                df["current_dpd"] >= 30,
                "Stage 2",
                "Stage 1",
            ),
        )

        # -----------------------------------------------------
        # 2. PD assignment
        # -----------------------------------------------------
        # If loan-level PD exists (logistic model), merge it
        if self.pd_results.get("loan_level_pd") is not None:
            loan_pd = self.pd_results["loan_level_pd"]
            df = df.merge(loan_pd, on="loan_id", how="left")
            df["pd_12m_used"] = df["pd_12m"]
        else:
            # Use portfolio-level PDs
            df["pd_12m_used"] = self.pd_results["pd_12m"]

        portfolio_pd_lifetime = self.pd_results["pd_lifetime"]

        # -----------------------------------------------------
        # 3. LGD assignment
        # -----------------------------------------------------
        if isinstance(self.lgd_input, pd.DataFrame):
            lgd_df = self.lgd_input.copy()
            df = df.merge(lgd_df, on="loan_id", how="left")
        else:
            df["lgd"] = float(self.lgd_input)

        # -----------------------------------------------------
        # 4. Merge EAD
        # -----------------------------------------------------
        df = df.merge(self.ead_df, on="loan_id", how="left")

        # Defensive fills
        df["lgd"] = df["lgd"].fillna(0)
        df["ead"] = df["ead"].fillna(0)

        # -----------------------------------------------------
        # 5. ECL calculation by stage
        # -----------------------------------------------------
        df["pd_used"] = np.where(
            df["stage"] == "Stage 1",
            df["pd_12m_used"],
            np.where(
                df["stage"] == "Stage 2",
                portfolio_pd_lifetime,
                1.0,  # Stage 3: defaulted
            ),
        )

        df["ecl"] = np.where(
            df["stage"] == "Stage 1",
            df["pd_used"] * df["lgd"] * df["ead"],
            np.where(
                df["stage"] == "Stage 2",
                df["pd_used"] * df["lgd"] * df["ead"],
                df["lgd"] * df["ead"],
            ),
        )

        loan_ecl = df[
            ["loan_id", "stage", "pd_used", "lgd", "ead", "ecl"]
        ].reset_index(drop=True)

        self._loan_level_ecl = loan_ecl
        return loan_ecl

    # ---------------------------------------------------------
    # Portfolio summaries
    # ---------------------------------------------------------
    def get_portfolio_summary(self) -> dict:
        """
        Compute portfolio-level ECL summaries.

        Returns
        -------
        dict
            {
              'total_ecl': float,
              'total_ecl_by_stage': pd.Series
            }
        """

        if self._loan_level_ecl is None:
            raise RuntimeError(
                "Run calculate_loan_level_ecl() first."
            )

        total_ecl = float(self._loan_level_ecl["ecl"].sum())

        total_by_stage = (
            self._loan_level_ecl
            .groupby("stage")["ecl"]
            .sum()
        )

        return {
            "total_ecl": total_ecl,
            "total_ecl_by_stage": total_by_stage,
        }