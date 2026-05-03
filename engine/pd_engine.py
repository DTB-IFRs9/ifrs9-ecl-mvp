"""
PD Engine Wrapper

This module provides a single, unified interface to run
different Probability of Default (PD) models.

Supported models:
- Vintage PD
- Roll-Rate (Transition Matrix) PD
- Logistic Regression PD
- Survival (Kaplan–Meier) PD

The engine returns standardized outputs for downstream
ECL calculation and reporting.
"""

from typing import Dict, Optional
import pandas as pd

from engine.pd_vintage import VintagePD
from engine.pd_rollrate import RollRatePD
from engine.pd_logistic import LogisticPD
from engine.pd_survival import SurvivalPD


class PDEngine:
    """
    Unified PD Engine that orchestrates multiple PD models.
    """

    SUPPORTED_MODELS = ["vintage", "rollrate", "logistic", "survival"]

    def __init__(
        self,
        df_long: pd.DataFrame,
        df_default: pd.DataFrame,
        loan_summary: pd.DataFrame,
    ):
        """
        Parameters
        ----------
        df_long : pd.DataFrame
            Long-format DPD dataframe (Task 3 output)

        df_default : pd.DataFrame
            Loan-level default dataframe (Task 4 output)

        loan_summary : pd.DataFrame
            Loan-level summary dataframe (Task 5 output)
        """
        self.df_long = df_long
        self.df_default = df_default
        self.loan_summary = loan_summary

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    def validate_model_type(self, model_type: str):
        """
        Validate supported PD model type.
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Choose from {self.SUPPORTED_MODELS}."
            )

    # ---------------------------------------------------------
    # Main execution method
    # ---------------------------------------------------------
    def run(self, model_type: str) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Run a selected PD model and return standardized outputs.

        Parameters
        ----------
        model_type : str
            One of ['vintage', 'rollrate', 'logistic', 'survival']

        Returns
        -------
        dict
            {
              'model_name': str,
              'pd_12m': float,
              'pd_lifetime': float,
              'pd_curve': DataFrame or None,
              'loan_level_pd': DataFrame or None
            }
        """

        self.validate_model_type(model_type)

        # -----------------------------------------------------
        # Vintage PD
        # -----------------------------------------------------
        if model_type == "vintage":
            model = VintagePD(self.df_long, self.df_default)
            model.fit()

            return {
                "model_name": "Vintage PD",
                "pd_12m": model.get_pd_12m(),
                "pd_lifetime": model.get_pd_lifetime(),
                "pd_curve": model.get_pd_curve(),
                "loan_level_pd": None,
            }

        # -----------------------------------------------------
        # Roll-Rate PD
        # -----------------------------------------------------
        if model_type == "rollrate":
            model = RollRatePD(self.df_long)
            model.fit()

            return {
                "model_name": "Roll-Rate PD",
                "pd_12m": model.get_pd_12m(),
                "pd_lifetime": model.get_pd_lifetime(),
                "pd_curve": model.get_transition_matrix(),
                "loan_level_pd": None,
            }

        # -----------------------------------------------------
        # Logistic PD
        # -----------------------------------------------------
        if model_type == "logistic":
            model = LogisticPD(self.df_long, self.df_default)
            model.fit()

            loan_pd = model.predict_pd_12m()

            return {
                "model_name": "Logistic PD",
                "pd_12m": model.get_portfolio_pd_12m(),
                "pd_lifetime": None,  # Logistic model is 12-month only
                "pd_curve": None,
                "loan_level_pd": loan_pd,
            }

        # -----------------------------------------------------
        # Survival PD
        # -----------------------------------------------------
        if model_type == "survival":
            model = SurvivalPD(self.df_default)
            model.fit()

            return {
                "model_name": "Survival PD",
                "pd_12m": model.get_pd_12m(),
                "pd_lifetime": model.get_pd_lifetime(),
                "pd_curve": model.get_pd_curve(),
                "loan_level_pd": None,
            }