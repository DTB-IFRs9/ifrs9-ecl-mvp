"""
Logistic Regression PD (12-Month)

This module implements a simple, explainable 12‑month PD model using
logistic regression and behavioral features only.

Key properties:
- Target: default within next 12 months (DPD >= 90)
- Features: current DPD, recent delinquency, MOB
- Transparent, auditor‑friendly, demo‑ready
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticPD:
    """
    12‑month Probability of Default model using logistic regression.
    """

    def __init__(self, df_long: pd.DataFrame, df_default: pd.DataFrame):
        """
        Parameters
        ----------
        df_long : pd.DataFrame
            Long-format DPD dataframe with columns:
            [
                'loan_id', 'snapshot_date', 'dpd',
                'origination_date', 'mob'
            ]

        df_default : pd.DataFrame
            Loan-level default table with columns:
            [
                'loan_id', 'default_flag', 'time_to_default'
            ]
        """
        self.df_long = df_long.copy()
        self.df_default = df_default.copy()

        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
        )

        self._modeling_df = None
        self._pd_12m_per_loan = None

    # -------------------------------------------------------------
    # Model fitting
    # -------------------------------------------------------------
    def fit(self):
        """
        Build the modeling dataset and train the logistic regression.
        """

        # ---------------------------------------------------------
        # 1. Merge default timing onto long-format data
        # ---------------------------------------------------------
        df = self.df_long.merge(
            self.df_default[["loan_id", "default_flag", "time_to_default"]],
            on="loan_id",
            how="left",
        )

        # ---------------------------------------------------------
        # 2. Determine target: default within next 12 months
        # ---------------------------------------------------------
        # A default occurs within 12 months if:
        #   time_to_default ∈ (mob, mob + 12]
        df["target"] = np.where(
            (df["default_flag"] == 1) &
            (df["time_to_default"] > df["mob"]) &
            (df["time_to_default"] <= df["mob"] + 12),
            1,
            0,
        )

        # ---------------------------------------------------------
        # 3. Exclude observations without 12‑month look‑ahead
        # ---------------------------------------------------------
        max_mob_per_loan = (
            df.groupby("loan_id")["mob"]
            .max()
            .rename("max_mob")
            .reset_index()
        )

        df = df.merge(max_mob_per_loan, on="loan_id", how="left")

        df = df[df["mob"] + 12 <= df["max_mob"]]

        # ---------------------------------------------------------
        # 4. Feature engineering
        # ---------------------------------------------------------
        # current_dpd
        df["current_dpd"] = df["dpd"].fillna(0)

        # max_dpd_last_6m (rolling over MOB)
        df = df.sort_values(["loan_id", "mob"])

        df["max_dpd_last_6m"] = (
            df
            .groupby("loan_id")["dpd"]
            .rolling(window=6, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

        # mob is already available

        # ---------------------------------------------------------
        # 5. Final modeling dataset
        # ---------------------------------------------------------
        features = ["current_dpd", "max_dpd_last_6m", "mob"]

        modeling_df = df[
            ["loan_id", "snapshot_date", "target"] + features
        ].dropna()

        X = modeling_df[features]
        y = modeling_df["target"]

        # ---------------------------------------------------------
        # 6. Train logistic regression
        # ---------------------------------------------------------
        self.model.fit(X, y)

        self._modeling_df = modeling_df

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------
    def predict_pd_12m(self) -> pd.DataFrame:
        """
        Predict 12‑month PD per loan using the latest observation.

        Returns
        -------
        pd.DataFrame
            ['loan_id', 'pd_12m']
        """

        if self._modeling_df is None:
            raise RuntimeError("Model must be fitted before prediction.")

        # ---------------------------------------------------------
        # 1. Use latest available observation per loan
        # ---------------------------------------------------------
        latest_obs = (
            self._modeling_df
            .sort_values(["loan_id", "snapshot_date"])
            .groupby("loan_id", as_index=False)
            .tail(1)
        )

        X_latest = latest_obs[
            ["current_dpd", "max_dpd_last_6m", "mob"]
        ]

        pd_12m = self.model.predict_proba(X_latest)[:, 1]

        results = pd.DataFrame({
            "loan_id": latest_obs["loan_id"].values,
            "pd_12m": pd_12m,
        })

        self._pd_12m_per_loan = results

        return results

    # -------------------------------------------------------------
    # Portfolio metric
    # -------------------------------------------------------------
    def get_portfolio_pd_12m(self) -> float:
        """
        Returns
        -------
        float
            Average portfolio 12‑month PD
        """

        if self._pd_12m_per_loan is None:
            raise RuntimeError("Run predict_pd_12m() first.")

        return float(self._pd_12m_per_loan["pd_12m"].mean())
