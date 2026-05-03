"""
Roll‑Rate / Transition Matrix PD Model

This module implements a Markov (migration-based) PD model using
month‑to‑month DPD state transitions.

Key features:
- Industry‑standard roll‑rate logic
- Default treated as absorbing state
- Produces 12‑month and lifetime PD
- Fully explainable and IFRS‑9 friendly
"""

import pandas as pd
import numpy as np


class RollRatePD:
    """
    Roll‑Rate / Transition Matrix Probability of Default model.
    """

    STATES = ["S0", "S30", "S60", "S90"]

    def __init__(self, df_long: pd.DataFrame):
        """
        Parameters
        ----------
        df_long : pd.DataFrame
            Long-format DPD dataframe with columns:
            ['loan_id', 'snapshot_date', 'dpd', 'mob']
        """
        self.df = df_long.copy()
        self.transition_matrix = None

    # ---------------------------------------------------------
    # Utility: map DPD to delinquency state
    # ---------------------------------------------------------
    @staticmethod
    def map_state(dpd: float) -> str:
        if pd.isna(dpd):
            return None
        if dpd == 0:
            return "S0"
        if 1 <= dpd < 30:
            return "S30"
        if 30 <= dpd < 90:
            return "S60"
        return "S90"

    # ---------------------------------------------------------
    # Model fitting
    # ---------------------------------------------------------
    def fit(self):
        """
        Build the transition probability matrix.
        """

        df = self.df.copy()

        # -----------------------------------------------------
        # 1. Assign delinquency states
        # -----------------------------------------------------
        df["state"] = df["dpd"].apply(self.map_state)

        # Sort for transition construction
        df = df.sort_values(["loan_id", "snapshot_date"])

        # -----------------------------------------------------
        # 2. Build next-period states via groupby shift
        # -----------------------------------------------------
        df["next_state"] = (
            df.groupby("loan_id")["state"]
            .shift(-1)
        )

        # Drop last observation per loan (no next state)
        transitions = df.dropna(subset=["state", "next_state"])

        # -----------------------------------------------------
        # 3. Count transitions
        # -----------------------------------------------------
        transition_counts = (
            transitions
            .groupby(["state", "next_state"])
            .size()
            .reset_index(name="count")
        )

        # -----------------------------------------------------
        # 4. Pivot to matrix form
        # -----------------------------------------------------
        matrix = (
            transition_counts
            .pivot(index="state", columns="next_state", values="count")
            .reindex(index=self.STATES, columns=self.STATES, fill_value=0)
        )

        # -----------------------------------------------------
        # 5. Normalize rows to probabilities
        # -----------------------------------------------------
        matrix = matrix.div(matrix.sum(axis=1), axis=0)

        # -----------------------------------------------------
        # 6. Enforce absorbing default state (S90)
        # -----------------------------------------------------
        matrix.loc["S90"] = 0.0
        matrix.loc["S90", "S90"] = 1.0

        self.transition_matrix = matrix

    # ---------------------------------------------------------
    # Accessors
    # ---------------------------------------------------------
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            Transition probability matrix (rows sum to 1)
        """
        return self.transition_matrix.copy()

    # ---------------------------------------------------------
    # PD calculations
    # ---------------------------------------------------------
    def _compute_pd(self, months: int) -> float:
        """
        Compute probability of reaching default (S90)
        within a given horizon assuming start in S0.
        """

        P = self.transition_matrix.values

        # Matrix exponentiation
        P_n = np.linalg.matrix_power(P, months)

        # Start at S0 → probability of being in S90
        start_index = self.STATES.index("S0")
        default_index = self.STATES.index("S90")

        return float(P_n[start_index, default_index])

    def get_pd_12m(self) -> float:
        """
        Returns
        -------
        float
            12-month probability of default
        """
        return self._compute_pd(months=12)

    def get_pd_lifetime(self) -> float:
        """
        Returns
        -------
        float
            Lifetime probability of default
            (approximated using long horizon)
        """
        return self._compute_pd(months=60)
