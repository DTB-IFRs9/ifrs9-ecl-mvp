"""
Survival-Based PD Model (Kaplan–Meier)

This module implements a portfolio-level Probability of Default (PD)
model using survival analysis.

Key characteristics:
- Uses time-to-default with censoring
- Fully non-parametric (Kaplan–Meier)
- Auditor- and regulator-friendly
- Naturally produces lifetime PD
"""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter


class SurvivalPD:
    """
    Survival-based Probability of Default model using Kaplan–Meier.
    """

    def __init__(self, df_default: pd.DataFrame):
        """
        Parameters
        ----------
        df_default : pd.DataFrame
            Loan-level default dataframe with columns:
            [
                'loan_id',
                'time_to_default',
                'default_flag',
                'censored'
            ]
        """
        self.df_default = df_default.copy()
        self.kmf = KaplanMeierFitter()

        self._survival_curve = None
        self._pd_curve = None

    # ---------------------------------------------------------
    # Model fitting
    # ---------------------------------------------------------
    def fit(self):
        """
        Fit the Kaplan-Meier survival model on the full loan portfolio.
        """

        # -----------------------------------------------------
        # 1. Prepare duration and event indicator
        # -----------------------------------------------------
        durations = self.df_default["time_to_default"]
        events = self.df_default["default_flag"]

        # -----------------------------------------------------
        # 2. Fit Kaplan–Meier model
        # -----------------------------------------------------
        self.kmf.fit(
            durations=durations,
            event_observed=events,
        )

        # -----------------------------------------------------
        # 3. Extract survival function S(t)
        # -----------------------------------------------------
        survival_df = self.kmf.survival_function_.reset_index()
        survival_df.columns = ["mob", "survival_prob"]

        # -----------------------------------------------------
        # 4. Convert to cumulative PD: PD(t) = 1 - S(t)
        # -----------------------------------------------------
        survival_df["pd"] = 1.0 - survival_df["survival_prob"]

        self._survival_curve = survival_df[["mob", "survival_prob"]]
        self._pd_curve = survival_df[["mob", "pd"]]

    # ---------------------------------------------------------
    # Accessors
    # ---------------------------------------------------------
    def get_pd_curve(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            Cumulative PD curve with columns:
            ['mob', 'pd']
        """
        return self._pd_curve.copy()

    def get_pd_12m(self) -> float:
        """
        Returns
        -------
        float
            PD at 12 months (interpolated if necessary)
        """

        if self._pd_curve is None:
            raise RuntimeError("Model must be fitted first.")

        pd_curve = self._pd_curve.set_index("mob").sort_index()

        if 12 in pd_curve.index:
            return float(pd_curve.loc[12, "pd"])

        # Linear interpolation if exact 12m not observed
        return float(
            np.interp(
                12,
                pd_curve.index.values,
                pd_curve["pd"].values,
            )
        )

    def get_pd_lifetime(self) -> float:
        """
        Returns
        -------
        float
            Lifetime PD at maximum observed time
        """

        if self._pd_curve is None:
            raise RuntimeError("Model must be fitted first.")

        return float(
            self._pd_curve
            .sort_values("mob")
            .tail(1)["pd"]
            .iloc[0]
        )