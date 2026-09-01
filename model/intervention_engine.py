"""
Intervention Engine for Patient Readmission Reduction
------------------------------------------------------
Maps patient risk drivers and clinical features to concrete, actionable 
care-team interventions.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np


def get_patient_interventions(patient: pd.Series, risk_prob: float = None, threshold: float = 0.25) -> Dict[str, Any]:
    """
    Analyzes a single patient record and determines risk drivers and personalized interventions.

    Parameters:
        patient: pd.Series containing patient demographic and clinical attributes.
        risk_prob: float (optional), predicted probability of 30-day readmission.
        threshold: float (default 0.25), classification threshold for High Risk classification.

    Returns:
        Dict containing:
            - risk_level: str ("High Risk", "Moderate Risk", "Low Risk")
            - top_drivers: List[str]
            - recommended_interventions: List[str]
    """
    drivers = []
    interventions = []

    # Rule 1: Follow-up Appointment Status
    if str(patient.get("follow_up_scheduled", "")).upper() in ["N", "NO"]:
        drivers.append("No post-discharge follow-up appointment scheduled")
        interventions.append("Flag for care coordinator outreach to schedule PCP/specialist visit within 48 hrs of discharge")

    # Rule 2: Case Manager & High Utilization
    prior_adm = int(patient.get("prior_admissions_12mo", 0))
    has_cm = str(patient.get("has_case_manager", "")).upper() in ["Y", "YES"]
    if not has_cm and prior_adm >= 2:
        drivers.append(f"High prior admissions ({prior_adm} in 12mo) without assigned Case Manager")
        interventions.append("Assign dedicated Case Manager prior to discharge for transition support")
    elif prior_adm >= 3:
        drivers.append(f"Frequent readmitter history ({prior_adm} prior admissions in 12mo)")
        interventions.append("Enroll in High-Utilizer Multidisciplinary Care Coordination Program")

    # Rule 3: Discharge Disposition (AMA / SNF)
    discharge = str(patient.get("discharge_disposition", ""))
    if discharge == "Against Medical Advice":
        drivers.append("Discharged Against Medical Advice (AMA)")
        interventions.append("Conduct urgent AMA risk counseling, safety plan review, & social work consultation")
    elif discharge == "Skilled Nursing Facility":
        drivers.append("Discharged to Skilled Nursing Facility (SNF)")
        interventions.append("Establish warm handoff & 72-hour clinical check-in with SNF medical director")

    # Rule 4: High-Risk Primary Diagnoses
    diag = str(patient.get("primary_diagnosis", ""))
    high_risk_diagnoses = ["Heart Failure", "COPD", "Sepsis", "Kidney Disease"]
    if diag in high_risk_diagnoses:
        drivers.append(f"High-acuity primary diagnosis: {diag}")
        interventions.append(f"Enroll in disease-specific care protocol & disease management clinic for {diag}")

    # Rule 5: Polypharmacy / Medication Burden
    num_meds = int(patient.get("num_medications", 0))
    if num_meds >= 10:
        drivers.append(f"Polypharmacy regimen ({num_meds} active medications)")
        interventions.append("Schedule inpatient clinical pharmacist medication reconciliation & teach-back prior to discharge")

    # Rule 6: Uninsured / Socioeconomic Barriers
    insurance = str(patient.get("insurance_type", ""))
    if insurance == "Uninsured":
        drivers.append("Uninsured patient status (financial/access barrier)")
        interventions.append("Connect with financial counselor for Medicaid eligibility screening & community prescription assistance")
    elif insurance == "Medicaid":
        drivers.append("Medicaid insurance status (social drivers of health risk)")
        interventions.append("Assess Social Determinants of Health (SDOH) & provide community resource navigator support")

    # Rule 7: Extended Length of Stay
    los = int(patient.get("length_of_stay_days", 0))
    if los >= 7:
        drivers.append(f"Extended inpatient stay ({los} days)")
        interventions.append("Arrange post-discharge home health assessment & mobility safety evaluation")

    # Rule 8: Advanced Age
    age = int(patient.get("age", 0))
    if age >= 75:
        drivers.append(f"Elderly patient age ({age} years)")
        interventions.append("Provide caregiver instruction packet & initiate 7-day post-discharge welfare call")

    # Risk level assignment based on predicted probability or driver count
    if risk_prob is not None:
        if risk_prob >= threshold:
            risk_level = "High Risk"
        elif risk_prob >= (threshold * 0.6):
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"
    else:
        # Fallback if probability not supplied
        if len(drivers) >= 3:
            risk_level = "High Risk"
        elif len(drivers) >= 1:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

    # Default fallback interventions if none triggered
    if not interventions:
        interventions.append("Standard discharge instructions and routine follow-up care in 7-14 days")

    return {
        "risk_level": risk_level,
        "top_drivers": drivers[:3],  # Return top 2-3 drivers
        "recommended_interventions": list(dict.fromkeys(interventions))  # Deduplicate while preserving order
    }


def evaluate_batch_interventions(df: pd.DataFrame, probs: np.ndarray = None, threshold: float = 0.25) -> pd.DataFrame:
    """
    Evaluates risk drivers and interventions for an entire DataFrame of patients.
    Returns DataFrame augmented with 'risk_score', 'risk_level', 'top_drivers', and 'interventions'.
    """
    df_out = df.copy()
    
    if probs is not None:
        df_out["risk_score"] = probs
    else:
        df_out["risk_score"] = 0.0

    risk_levels = []
    top_drivers_list = []
    interventions_list = []

    for idx, row in df_out.iterrows():
        p_prob = row["risk_score"] if probs is not None else None
        res = get_patient_interventions(row, risk_prob=p_prob, threshold=threshold)
        risk_levels.append(res["risk_level"])
        top_drivers_list.append(" | ".join(res["top_drivers"]) if res["top_drivers"] else "None identified")
        interventions_list.append(" | ".join(res["recommended_interventions"]))

    df_out["risk_level"] = risk_levels
    df_out["top_drivers"] = top_drivers_list
    df_out["recommended_interventions"] = interventions_list

    return df_out
