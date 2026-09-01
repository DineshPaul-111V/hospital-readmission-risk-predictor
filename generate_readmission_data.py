"""
Synthetic Patient Readmission Dataset Generator
------------------------------------------------
Generates a realistic synthetic dataset of hospital admissions with
plausible correlations between patient/clinical features and 30-day
readmission risk. Intended as a stand-in when a real (de-identified)
dataset isn't available yet.

Usage:
    python generate_readmission_data.py --n 5000 --out patient_admissions_sample.csv
"""

import argparse
import numpy as np
import pandas as pd


def generate_dataset(n_records: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Core demographic / stay features ---
    age = rng.integers(18, 95, size=n_records)

    diagnosis_options = [
        "Heart Failure", "COPD", "Diabetes", "Pneumonia",
        "Sepsis", "Kidney Disease", "Stroke", "Hip/Knee Surgery"
    ]
    # Some diagnoses are inherently higher-risk for readmission
    diagnosis_risk = {
        "Heart Failure": 0.35, "COPD": 0.28, "Diabetes": 0.20,
        "Pneumonia": 0.22, "Sepsis": 0.30, "Kidney Disease": 0.27,
        "Stroke": 0.24, "Hip/Knee Surgery": 0.12
    }
    primary_diagnosis = rng.choice(diagnosis_options, size=n_records)

    length_of_stay = np.clip(
        rng.normal(loc=5, scale=3, size=n_records), 1, 30
    ).round().astype(int)

    prior_admissions_12mo = rng.poisson(lam=1.1, size=n_records)

    discharge_options = ["Home", "Home Health Care", "Skilled Nursing Facility", "Against Medical Advice"]
    discharge_probs = [0.55, 0.20, 0.20, 0.05]
    discharge_disposition = rng.choice(discharge_options, size=n_records, p=discharge_probs)

    follow_up_scheduled = rng.choice(["Y", "N"], size=n_records, p=[0.65, 0.35])

    num_medications = np.clip(rng.normal(loc=8, scale=4, size=n_records), 0, 30).round().astype(int)

    has_case_manager = rng.choice(["Y", "N"], size=n_records, p=[0.4, 0.6])

    insurance_options = ["Medicare", "Medicaid", "Private", "Uninsured"]
    insurance = rng.choice(insurance_options, size=n_records, p=[0.45, 0.20, 0.30, 0.05])

    # --- Build readmission probability from a weighted combination of factors ---
    base_risk = np.array([diagnosis_risk[d] for d in primary_diagnosis])

    risk_score = (
        base_risk
        + 0.10 * (age > 75)
        + 0.015 * length_of_stay
        + 0.06 * prior_admissions_12mo
        + np.where(discharge_disposition == "Against Medical Advice", 0.20, 0)
        + np.where(discharge_disposition == "Skilled Nursing Facility", 0.05, 0)
        - np.where(follow_up_scheduled == "Y", 0.15, 0)
        - np.where(has_case_manager == "Y", 0.08, 0)
        + 0.01 * np.clip(num_medications - 8, 0, None)
        + np.where(insurance == "Uninsured", 0.07, 0)
    )

    # Add noise, then squash to a valid probability.
    # Centered/scaled so the overall base rate lands near a realistic
    # 15-18% 30-day readmission rate, with clear separation between
    # low- and high-risk patients.
    risk_score += rng.normal(0, 0.05, size=n_records)
    readmit_prob = 1 / (1 + np.exp(-4 * (risk_score - 0.72)))

    readmitted_within_30_days = rng.binomial(1, readmit_prob)

    df = pd.DataFrame({
        "patient_id": [f"P{i:05d}" for i in range(1, n_records + 1)],
        "age": age,
        "primary_diagnosis": primary_diagnosis,
        "length_of_stay_days": length_of_stay,
        "prior_admissions_12mo": prior_admissions_12mo,
        "discharge_disposition": discharge_disposition,
        "follow_up_scheduled": follow_up_scheduled,
        "num_medications": num_medications,
        "has_case_manager": has_case_manager,
        "insurance_type": insurance,
        "readmitted_within_30_days": readmitted_within_30_days,
    })

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic patient readmission dataset")
    parser.add_argument("--n", type=int, default=5000, help="Number of patient records to generate")
    parser.add_argument("--out", type=str, default="patient_admissions_sample.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    df = generate_dataset(n_records=args.n, seed=args.seed)
    df.to_csv(args.out, index=False)

    print(f"Generated {len(df):,} records -> {args.out}")
    print(f"Overall readmission rate: {df['readmitted_within_30_days'].mean():.1%}")
    print("\nReadmission rate by diagnosis:")
    print(
        df.groupby("primary_diagnosis")["readmitted_within_30_days"]
        .mean()
        .sort_values(ascending=False)
        .apply(lambda x: f"{x:.1%}")
    )


if __name__ == "__main__":
    main()
