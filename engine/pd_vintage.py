"""
Vintage PD Model

This module implements a simple, explainable Vintage PD model.
It produces cumulative default rates by Months-on-Book (MOB)
and returns PD at 12 months and lifetime PD.

This model is:
- Auditor-friendly
- Non-parametric
- IFRS-9 compatible
"""

import pandas as pd
import numpy as np


class VintagePD:
    """
    Vintage Probability of Default (PD) model.
    """

    def __init__(self, df_long: pd.DataFrame, df_default: pd.DataFrame):
        """
        Parameters
        ----------
        df_long : pd.DataFrame
            Long-format DPD data from transform_dpd_wide_to_long()

        df_default : pd.DataFrame
            Loan-level default data from compute_default_events()
        """
        self.df_long = df_long.copy()
        self.df_default = df_default.copy()

        self._pd_curve = None
        self._pd_12m = None
        self._pd_lifetime = None

    # ---------------------------------------------------------
    # Model fitting
    # ---------------------------------------------------------
    def fit(self):
        """
        Build vintage-level cumulative PD curves and aggregate them
        into a single portfolio-level Vintage PD curve.
        """

        # -----------------------------------------------------
        # 1. Build loan-level default timing table
        # -----------------------------------------------------
        loans = (
            self.df_long[
                ["loan_id", "origination_date", "mob"]
            ]
            .drop_duplicates("loan_id")
            .merge(
                self.df_default[
                    ["loan_id", "default_flag", "time_to_default"]
                ],
                on="loan_id",
                how="left",
            )
        )

        # Loans that never default are treated as having infinite default time
        loans["default_mob"] = np.where(
            loans["default_flag"] == 1,
            loans["time_to_default"],
            np.inf,
        )

        loans["vintage"] = loans["origination_date"]

        # -----------------------------------------------------
        # 2. Create MOB grid
        # -----------------------------------------------------
        max_mob = int(self.df_long["mob"].max())
        mob_grid = pd.DataFrame({"mob": range(0, max_mob + 1)})

        # -----------------------------------------------------
        # 3. Cross loans with MOB grid
        #    → evaluate whether default has occurred by each MOB
        # -----------------------------------------------------
        loans_mob = loans.merge(mob_grid, how="cross")

        # Only consider MOBs where the loan has actually existed
        loans_mob = loans_mob[loans_mob["mob_y"] >= 0]

        loans_mob["default_by_mob"] = (
            loans_mob["default_mob"] <= loans_mob["mob_y"]
        ).astype(int)

        loans_mob = loans_mob.rename(columns={"mob_y": "mob"})

        # -----------------------------------------------------
        # 4. Vintage-level cumulative default rates
        # -----------------------------------------------------
        vintage_curve = (
            loans_mob
            .groupby(["vintage", "mob"], as_index=False)
            .agg(
                loans_originated=("loan_id", "count"),
                cumulative_defaults=("default_by_mob", "sum"),
            )
        )

        vintage_curve["cumulative_pd"] = (
            vintage_curve["cumulative_defaults"]
            / vintage_curve["loans_originated"]
        )

        # -----------------------------------------------------
        # 5. Aggregate across vintages (simple average)
        # -----------------------------------------------------
        pd_curve = (
            vintage_curve
            .groupby("mob", as_index=False)["cumulative_pd"]
            .mean()
            .sort_values("mob")
        )

        self._pd_curve = pd_curve.reset_index(drop=True)

        # -----------------------------------------------------
        # 6. PD at 12 months
        # -----------------------------------------------------
        pd_12m = (
            self._pd_curve[self._pd_curve["mob"] <= 12]
            .sort_values("mob")
            .tail(1)["cumulative_pd"]
        )

        self._pd_12m = float(pd_12m.iloc[0]) if not pd_12m.empty else np.nan

        # -----------------------------------------------------
        # 7. Lifetime PD (max observed MOB)
        # -----------------------------------------------------
        self._pd_lifetime = float(
            self._pd_curve
            .sort_values("mob")
            .tail(1)["cumulative_pd"]
            .iloc[0]
        )

    # ---------------------------------------------------------
    # Accessors
    # ---------------------------------------------------------
    def get_pd_curve(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            Columns: ['mob', 'cumulative_pd']
        """
        return self._pd_curve.copy()

    def get_pd_12m(self) -> float:
        """
        Returns
        -------
        float
            12-month cumulative PD
        """
        return self._pd_12m

    def get_pd_lifetime(self) -> float:
        """
        Returns
        -------
        float
            Lifetime cumulative PD
        """
        return self._pd_lifetime