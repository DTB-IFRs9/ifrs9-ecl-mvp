"""
Streamlit UI for IFRS 9 ECL MVP

IMPORTANT:
This application includes DEMO-ONLY authentication.
It is intentionally simplified and NOT suitable for production use.

The focus of this MVP is the IFRS 9 ECL engine, not security.
Enterprise authentication (SSO, OAuth, AD, etc.) can be added later.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

# Backend imports
from utils.dpd_transformer import (
    transform_dpd_wide_to_long,
    compute_default_events,
)
from utils.loader import build_loan_summary_table
from engine.pd_engine import PDEngine
from engine.lgd import get_lgd
from engine.ead import get_ead
from engine.ecl_calculator import ECLCalculator


# =========================================================
# DEMO-ONLY AUTHENTICATION — NOT FOR PRODUCTION USE
# =========================================================

# Hard-coded demo credentials
DEMO_USERS = {
    "admin": "ecldemo2026",
    "viewer": "ecl_view_2026",
}

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None


# ---------------------------------------------------------
# Login Screen (shown before any backend logic runs)
# ---------------------------------------------------------
if not st.session_state["authenticated"]:

    st.title("IFRS 9 ECL MVP — Demo Login")

    st.info(
        "This is a demo-only login.\n\n"
        "Security is intentionally simplified for this prototype."
    )

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in DEMO_USERS and DEMO_USERS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success("Login successful ✅")
            st.rerun()
        else:
            st.error("Invalid username or password")

    # Stop the app here if not authenticated
    st.stop()


# ---------------------------------------------------------
# Sidebar (shown only after login)
# ---------------------------------------------------------
with st.sidebar:
    st.write(f"✅ Logged in as: **{st.session_state['username']}**")

    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.rerun()


# =========================================================
# MAIN APPLICATION (PD + LGD + EAD + ECL)
# =========================================================

st.set_page_config(page_title="IFRS 9 ECL MVP", layout="wide")
st.title("IFRS 9 ECL MVP – End-to-End Demo")


# ---------------------------------------------------------
# Step 1: Upload TTC PD File
# ---------------------------------------------------------
st.subheader("Step 1: Upload TTC PD File")

uploaded_file = st.file_uploader(
    "Upload TTC PD file (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.warning("Please upload a TTC PD file to proceed.")
    st.stop()


# ---------------------------------------------------------
# Step 2: Data Preparation
# ---------------------------------------------------------
st.subheader("Step 2: Data Preparation")

with st.spinner("Running data preparation pipeline..."):
    try:
        df_long = transform_dpd_wide_to_long(uploaded_file)
        df_default = compute_default_events(df_long)
        loan_summary = build_loan_summary_table(df_long, df_default)
    except Exception as e:
        st.error(f"Data preparation failed: {e}")
        st.stop()

st.success("Data pipeline completed successfully ✅")

col1, col2 = st.columns(2)
with col1:
    st.metric("Number of Loans", loan_summary["loan_id"].nunique())
with col2:
    st.metric(
        "Date Range",
        f"{df_long['snapshot_date'].min().date()} → "
        f"{df_long['snapshot_date'].max().date()}",
    )

st.write("Loan Summary (sample)")
st.dataframe(loan_summary.head(10))


# ---------------------------------------------------------
# Step 3: Select PD Model
# ---------------------------------------------------------
st.subheader("Step 3: Select PD Model")

pd_model_map = {
    "Vintage (Cohort PD)": "vintage",
    "Roll-Rate (Transition Matrix)": "rollrate",
    "Logistic Regression (12-month PD)": "logistic",
    "Survival (Kaplan-Meier Lifetime PD)": "survival",
}

model_label = st.selectbox(
    "Choose a PD model",
    options=list(pd_model_map.keys()),
)
model_type = pd_model_map[model_label]


# ---------------------------------------------------------
# Step 4: Run PD Engine
# ---------------------------------------------------------
st.subheader("Step 4: Run PD Engine")

pd_results = None

if st.button("Run PD Model"):
    with st.spinner("Running PD Engine..."):
        try:
            engine = PDEngine(df_long, df_default, loan_summary)
            pd_results = engine.run(model_type)
        except Exception:
            if model_type == "logistic":
                st.error(
                    "Logistic PD model failed due to insufficient "
                    "12‑month look‑ahead history."
                )
            else:
                st.error("PD model execution failed.")
            st.stop()

    st.success("PD Model Run Completed ✅")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "12‑month PD",
            f"{pd_results['pd_12m']:.2%}"
            if pd_results["pd_12m"] is not None else "N/A",
        )
    with col2:
        st.metric(
            "Lifetime PD",
            f"{pd_results['pd_lifetime']:.2%}"
            if pd_results["pd_lifetime"] is not None else "N/A",
        )

    if pd_results["pd_curve"] is not None:
        pd_curve_df = pd_results["pd_curve"].copy()
        st.write("PD Curve (Table)")
        st.dataframe(pd_curve_df)

        y_col = "pd" if "pd" in pd_curve_df.columns else "cumulative_pd"
        fig_pd = px.line(
            pd_curve_df,
            x="mob",
            y=y_col,
            title="PD Curve (Cumulative)",
            labels={"mob": "Months on Book", y_col: "Probability of Default"},
        )
        st.plotly_chart(fig_pd, use_container_width=True)

    if pd_results["loan_level_pd"] is not None:
        st.write("Loan‑level PD (Preview)")
        st.dataframe(pd_results["loan_level_pd"].head())
    else:
        st.info("This PD model does not produce loan‑level PDs.")


# ---------------------------------------------------------
# Step 5: ECL Calculation
# ---------------------------------------------------------
st.subheader("Step 5: ECL Calculation")

ead_file = st.file_uploader(
    "Upload EAD File (Optional)",
    type=["csv", "xlsx", "xls"],
)

loan_level_ecl_df = None

if st.button("Run ECL Calculation"):

    if pd_results is None:
        st.warning("Please run a PD model first.")
        st.stop()

    with st.spinner("Calculating Expected Credit Loss..."):
        lgd_value = get_lgd()
        ead_df = get_ead(loan_summary, ead_file)

        ecl_calc = ECLCalculator(
            loan_summary_df=loan_summary[
                ["loan_id", "current_dpd", "product_type"]
            ],
            pd_results=pd_results,
            lgd_df_or_scalar=lgd_value,
            ead_df=ead_df,
        )

        loan_level_ecl_df = ecl_calc.calculate_loan_level_ecl()
        portfolio_summary = ecl_calc.get_portfolio_summary()

    st.success("ECL Calculation Completed ✅")

    st.metric(
        "Total ECL",
        f"{portfolio_summary['total_ecl']:,.0f} BDT",
    )

    ecl_by_stage_df = (
        portfolio_summary["total_ecl_by_stage"]
        .reset_index()
        .rename(columns={"ecl": "Total ECL"})
    )

    st.write("ECL by Stage")
    st.dataframe(ecl_by_stage_df)

    fig_ecl = px.bar(
        ecl_by_stage_df,
        x="stage",
        y="Total ECL",
        title="ECL by Stage",
        labels={"stage": "Stage"},
    )
    st.plotly_chart(fig_ecl, use_container_width=True)

    st.subheader("Loan‑level ECL (Preview)")
    st.dataframe(loan_level_ecl_df.head())


# ---------------------------------------------------------
# Step 6: Download Results
# ---------------------------------------------------------
if loan_level_ecl_df is not None:

    st.subheader("Download Results")

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        loan_level_ecl_df.to_excel(
            writer,
            index=False,
            sheet_name="Loan_Level_ECL",
        )

    buffer.seek(0)

    st.download_button(
        label="Download Loan-Level ECL (Excel)",
        data=buffer,
        file_name="loan_level_ecl.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
