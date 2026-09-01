# 🏥 30-Day Patient Readmission Reduction Platform

An end-to-end clinical decision support platform designed to reduce 30-day hospital readmissions by **15% within 12 months**. Built using synthetic patient admission data (5,000 records), interpretable machine learning models, clinical intervention mapping rules, and an interactive Streamlit dashboard.

---

## 🎯 Strategic Goal & Context

- **Baseline 30-Day Readmission Rate**: **18.7%**
- **12-Month Target Readmission Rate**: **15.9%** (15% relative reduction goal)
- **Clinical & Financial Rationale**:
  - **False Negative (FN) Cost**: A high-risk patient discharged without intervention returns as an emergency readmission ($\$15,000–\$20,000$ in un-reimbursed costs & penalties).
  - **False Positive (FP) Cost**: A low-risk patient receives a care coordinator call or case manager visit ($\$50–\$150$ operational cost).
  - **Decision Strategy**: Calibrate classification thresholds to **prioritize Recall (sensitivity)** over precision to maximize patient capture.

---

## 📁 Repository Structure

```
c:\Users\abish\data analysis\
├── generate_readmission_data.py   # Dataset generator script (5,000 records)
├── patient_admissions_sample.csv  # Synthetic patient admission dataset
├── analysis/
│   ├── eda_feature_importance.py  # Script for univariate analysis & feature importances
│   ├── feature_importance.png     # Saved bar chart of top readmission drivers
│   └── eda_summary.txt            # Written executive summary of EDA findings
├── model/
│   ├── train_model.py             # Preprocessing pipeline, model training, & threshold tuning
│   ├── evaluation_report.md       # Comprehensive evaluation & confusion matrix report
│   ├── readmission_model.joblib   # Serialized scikit-learn pipeline & metadata
│   └── intervention_engine.py     # Rule-based clinical intervention mapping function
├── dashboard/
│   └── app.py                     # Streamlit 3-page interactive clinical dashboard
└── README.md                      # Project documentation & run guide
```

---

## 💡 Key Clinical Drivers & EDA Findings

From univariate analysis and interpretable model feature rankings:

1. **Prior Admissions (12 Mo)**: The single strongest driver of 30-day readmission risk. Patients with $\ge 3$ prior admissions readmit at **$>28\%$**.
2. **Post-Discharge Follow-Up**: Absence of a scheduled follow-up appointment (`Follow-up == N`) increases readmission rate from **15.3% to 24.6%**.
3. **High-Acuity Diagnoses**: Heart Failure (**22.7%**), Stroke (**21.6%**), and Sepsis (**21.4%**) present significantly higher risk than elective surgical cases (**13.0%**).
4. **Discharge Disposition**: Discharges Against Medical Advice (AMA) escalate readmission rate to **35.7%**.
5. **Care Coordination Gap**: Unassigned case management for frequent utilizers and polypharmacy ($\ge 10$ medications) markedly increase risk.

---

## 🤖 Predictive Modeling & Threshold Tuning

Two models were evaluated using an **80/20 stratified train/test split**:
- **Logistic Regression (Interpretable Baseline)**: Test ROC-AUC = **0.6769**
- **Gradient Boosting Classifier**: Test ROC-AUC = **0.6388**

### Threshold Calibration Table (Logistic Regression Pipeline)

| Threshold | Recall (Sensitivity) | Precision | F1 Score | Clinical & Financial Impact |
| :---: | :---: | :---: | :---: | :--- |
| `0.50` (Default) | **5.88%** | 64.71% | 0.1078 | Misses 176 out of 187 high-risk patients ❌ |
| `0.30` | **31.55%** | 37.82% | 0.3440 | Improved capture, conservative worklist |
| `0.25` | **46.52%** | 32.71% | 0.3841 | Balanced trade-off |
| **`0.18` (Tuned)** | **66.84%** | **27.23%** | **0.3870** | **Captures 125 of 187 readmissions (Recommended)** ✅ |
| `0.15` | **78.07%** | 24.01% | 0.3673 | Maximum sensitivity for high-capacity teams |

> **Recommendation**: Deploy the interpretable **Logistic Regression pipeline** at threshold **`0.18`**, capturing **66.8%–78% of readmissions** while keeping care team worklists manageable.

---

## 🩺 Intervention Mapping Rules

The `intervention_engine.py` module evaluates patient rows dynamically and attaches targeted interventions:

- `follow_up_scheduled == 'N'` $\rightarrow$ **"Flag for care coordinator outreach within 48 hrs of discharge"**
- `has_case_manager == 'N'` & `prior_admissions_12mo >= 2` $\rightarrow$ **"Assign dedicated Case Manager prior to discharge"**
- `discharge_disposition == 'Against Medical Advice'` $\rightarrow$ **"Conduct urgent AMA risk counseling & social work consultation"**
- `primary_diagnosis in ['Heart Failure', 'COPD', 'Sepsis', 'Kidney Disease']` $\rightarrow$ **"Enroll in disease management protocol"**
- `num_medications >= 10` $\rightarrow$ **"Schedule inpatient pharmacist medication reconciliation & teach-back"**
- `insurance_type == 'Uninsured'` $\rightarrow$ **"Connect with financial counselor for Medicaid screening & financial aid"**
- `length_of_stay_days >= 7` $\rightarrow$ **"Arrange post-discharge home health assessment & mobility safety check"**

---

## 🚀 How to Run

### 1. Requirements & Setup
Ensure Python 3.10+ is installed along with required packages:
```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn streamlit
```

### 2. Dataset Generation (Optional - Pre-generated)
```bash
python generate_readmission_data.py --n 5000 --out patient_admissions_sample.csv
```

### 3. Run EDA & Feature Importance
```bash
python analysis/eda_feature_importance.py
```
*Outputs `analysis/feature_importance.png` and `analysis/eda_summary.txt`.*

### 4. Train Model & Calibrate Threshold
```bash
python model/train_model.py
```
*Outputs `model/readmission_model.joblib` and `model/evaluation_report.md`.*

### 5. Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501` to explore:
- 📊 **Executive Overview**: Baseline vs 15% target progress, 12-month trajectory, feature importances.
- 📋 **Care Team Worklist**: Filterable/sortable patient risk scores, top drivers, and interactive intervention checklist.
- ⚙️ **Threshold Simulator**: Real-time slider to simulate recall, precision, worklist size, and net cost savings.
