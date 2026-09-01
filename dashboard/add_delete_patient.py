"""
Patient Management Module (Add / Remove Patient View)
----------------------------------------------------
Provides forms for adding new patient admissions with instant AI risk scoring 
& intervention mapping, and deleting completed patients with CSV persistence.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.intervention_engine import get_patient_interventions


# Persistent data file paths:
# We write active modifications to 'patients_current.csv' while preserving 'patient_admissions_sample.csv' untouched.
ACTIVE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "patients_current.csv")
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "patient_admissions_sample.csv")


def get_data_filepath() -> str:
    """
    Returns working dataset path, initializing patients_current.csv from sample data if it doesn't exist yet.
    """
    if not os.path.exists(ACTIVE_DATA_PATH):
        if os.path.exists(SAMPLE_DATA_PATH):
            df_init = pd.read_csv(SAMPLE_DATA_PATH)
            df_init.to_csv(ACTIVE_DATA_PATH, index=False)
        else:
            raise FileNotFoundError(f"Neither {ACTIVE_DATA_PATH} nor {SAMPLE_DATA_PATH} exists.")
    return ACTIVE_DATA_PATH


def save_patient_dataset(df_to_save: pd.DataFrame):
    """
    Persists working DataFrame to disk and clears Streamlit cache so all dashboard views update immediately.
    """
    raw_cols = [
        "patient_id", "age", "primary_diagnosis", "length_of_stay_days",
        "prior_admissions_12mo", "discharge_disposition", "follow_up_scheduled",
        "num_medications", "has_case_manager", "insurance_type", "readmitted_within_30_days"
    ]
    # Ensure only valid schema columns are saved
    df_clean = df_to_save[raw_cols].copy()
    data_path = get_data_filepath()
    df_clean.to_csv(data_path, index=False)
    # Invalidate cached dataset so Executive Overview & Worklist views reload updated file
    st.cache_data.clear()


def generate_next_patient_id(df: pd.DataFrame) -> str:
    """Generates sequential Patient ID (e.g. P05001)."""
    p_ids = df["patient_id"].astype(str).str.extract(r'(\d+)')[0].dropna().astype(int)
    max_id = p_ids.max() if not p_ids.empty else 0
    return f"P{max_id + 1:05d}"


def render_add_remove_patient_page(df_scored: pd.DataFrame, model_payload: dict):
    """Renders the Add / Remove Patient management interface."""
    st.markdown("""
    <div class="header-container">
        <div class="header-title">Patient Management Center</div>
        <div class="header-subtitle">Add newly admitted patients for instant AI risk scoring or remove completed cases</div>
    </div>
    """, unsafe_allow_html=True)

    tab_add, tab_remove = st.tabs(["➕ Add New Patient Admission", "🗑️ Remove Patient from List"])

    # --------------------------------------------------------------------------
    # TAB 1: ADD PATIENT
    # --------------------------------------------------------------------------
    with tab_add:
        st.subheader("Add New Patient Admission")
        st.caption("Enter clinical details below to calculate instant 30-day readmission risk score and care plan.")

        with st.form("add_patient_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Patient Age", min_value=18, max_value=105, value=65, step=1)
                primary_diag = st.selectbox("Primary Diagnosis", [
                    "Heart Failure", "COPD", "Diabetes", "Pneumonia",
                    "Sepsis", "Kidney Disease", "Stroke", "Hip/Knee Surgery"
                ])
                los = st.number_input("Length of Stay (Days)", min_value=1, max_value=60, value=5, step=1)

            with c2:
                prior_adm = st.number_input("Prior Admissions (Past 12Mo)", min_value=0, max_value=20, value=1, step=1)
                discharge_disp = st.selectbox("Discharge Disposition", [
                    "Home", "Home Health Care", "Skilled Nursing Facility", "Against Medical Advice"
                ])
                follow_up = st.selectbox("Follow-up Appointment Scheduled?", ["Y", "N"], index=0)

            with c3:
                num_meds = st.number_input("Active Medications Count", min_value=0, max_value=40, value=8, step=1)
                case_mgr = st.selectbox("Has Case Manager Assigned?", ["Y", "N"], index=1)
                insurance = st.selectbox("Insurance Type", ["Medicare", "Medicaid", "Private", "Uninsured"])

            submitted = st.form_submit_button("⚡ Score Patient & Save Admission", type="primary", width="stretch")

        if submitted:
            new_id = generate_next_patient_id(df_scored)
            new_row = {
                "patient_id": new_id,
                "age": age,
                "primary_diagnosis": primary_diag,
                "length_of_stay_days": los,
                "prior_admissions_12mo": prior_adm,
                "discharge_disposition": discharge_disp,
                "follow_up_scheduled": follow_up,
                "num_medications": num_meds,
                "has_case_manager": case_mgr,
                "insurance_type": insurance,
                "readmitted_within_30_days": 0
            }

            # Predict probability using trained model pipeline
            if model_payload and "pipeline" in model_payload:
                pipeline = model_payload["pipeline"]
                cat_cols = model_payload["categorical_cols"]
                num_cols = model_payload["numerical_cols"]
                df_single = pd.DataFrame([new_row])[cat_cols + num_cols]
                risk_prob = float(pipeline.predict_proba(df_single)[0, 1])
            else:
                risk_prob = 0.22

            thresh = model_payload["optimal_threshold"] if model_payload else 0.18
            interventions_info = get_patient_interventions(pd.Series(new_row), risk_prob=risk_prob, threshold=thresh)

            # Persist new record to CSV
            df_updated = pd.concat([df_scored, pd.DataFrame([new_row])], ignore_index=True)
            save_patient_dataset(df_updated)

            # Instant UI Feedback
            st.success(f"✅ Patient **{new_id}** added and saved to active dataset successfully!")

            m_col1, m_col2 = st.columns([1, 2])
            with m_col1:
                score_pct = risk_prob * 100
                st.metric("Predicted 30-Day Readmission Risk", f"{score_pct:.1f}%")
                if interventions_info["risk_level"] == "High Risk":
                    st.error("Risk Tier: HIGH RISK")
                elif interventions_info["risk_level"] == "Moderate Risk":
                    st.warning("Risk Tier: MODERATE RISK")
                else:
                    st.success("Risk Tier: LOW RISK")

            with m_col2:
                st.markdown("### 📋 Generated Clinical Care Plan")
                st.markdown("**Top Identified Risk Drivers:**")
                for d in interventions_info["top_drivers"]:
                    st.markdown(f"- ⚠️ {d}")
                st.markdown("**Recommended Care Team Interventions:**")
                for i_item in interventions_info["recommended_interventions"]:
                    st.markdown(f"- 🩺 {i_item}")

    # --------------------------------------------------------------------------
    # TAB 2: REMOVE PATIENT
    # --------------------------------------------------------------------------
    with tab_remove:
        st.subheader("Remove Patient Record")
        st.caption("Select a patient to remove from the active working dataset (e.g. completed care plan or discharged).")

        if df_scored.empty:
            st.warning("No patients currently in active working dataset.")
        else:
            # Build clean dropdown choices
            options = df_scored.apply(
                lambda r: f"{r['patient_id']} | Age {r['age']} | {r['primary_diagnosis']} | Risk: {r['risk_score']*100:.1f}%",
                axis=1
            ).tolist()

            selected_option = st.selectbox("Select Patient to Remove:", options)
            selected_pid = selected_option.split(" | ")[0]

            st.warning(f"⚠️ Are you sure you want to permanently delete patient **{selected_pid}** from active records?")
            if st.button("🗑️ Confirm & Delete Patient Record", type="secondary"):
                df_updated = df_scored[df_scored["patient_id"] != selected_pid].copy()
                save_patient_dataset(df_updated)
                st.success(f"✅ Patient **{selected_pid}** has been removed and changes persisted!")
                st.rerun()
