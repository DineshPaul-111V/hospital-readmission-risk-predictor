# Model Evaluation & Threshold Tuning Report

## Executive Summary
This document summarizes the performance evaluation and clinical threshold calibration for the 30-day patient readmission predictive model trained on 5,000 synthetic patient records (80/20 stratified split).

---

## 1. Model Performance Summary

| Model Architecture | Test ROC-AUC | Description |
| :--- | :---: | :--- |
| **Logistic Regression Baseline** | **0.6769** | Primary Interpretable Model |
| **Gradient Boosting Classifier** | **0.6388** | Non-linear Ensemble Model |

---

## 2. Threshold Calibration & Recall Prioritization

In 30-day hospital readmission prevention:
- **False Negative (FN) Cost**: A high-risk patient is misclassified as low-risk and discharged without intervention $\rightarrow$ High likelihood of emergency readmission within 30 days (estimated clinical/financial penalty cost: **\$15,000 - \$20,000** per un-reimbursed readmission).
- **False Positive (FP) Cost**: A low-risk patient receives a preventive follow-up call or case manager visit $\rightarrow$ Low financial cost (estimated operational cost: **\$50 - \$150** per outreach).

Consequently, **Recall must be prioritized over Precision** to ensure maximal patient capture.

### Precision-Recall Threshold Evaluation Curve

| Threshold | Recall (Sensitivity) | Precision | F1 Score | True Positives (TP) | False Negatives (FN) | False Positives (FP) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.50` | 5.9% | 64.7% | 0.1078 | 11 | 176 | 6 |
| `0.30` | 25.1% | 35.9% | 0.2956 | 47 | 140 | 84 |
| `0.25` | 39.0% | 33.8% | 0.3623 | 73 | 114 | 143 |
| `0.20` | 56.7% | 28.8% | 0.3820 | 106 | 81 | 262 |
| `0.18` | 66.8% | 27.2% | 0.3870 | 125 | 62 | 334 |
| `0.15` | 79.1% | 23.9% | 0.3677 | 148 | 39 | 470 |


---

## 3. Recommended Threshold Selection (`0.18`)

| Metric | Default Threshold (`0.50`) | Tuned Threshold (`0.18`) | Clinical Impact |
| :--- | :---: | :---: | :--- |
| **Recall (Sensitivity)** | **5.88%** | **66.84%** | **+61.0% patient capture** |
| **Precision** | **64.71%** | **27.23%** | Manageable care team worklist |
| **F1 Score** | **0.1078** | **0.3870** | Maximum clinical utility |
| **True Positives (TP)** | 11 | 125 | High-risk patients caught |
| **False Negatives (FN)** | 176 | 62 | **114 fewer missed readmissions** |

---

## 4. Confusion Matrix Comparison

### Default Threshold (`0.50`)
```
               Predicted Negative    Predicted Positive
Actual Negative      807                  6
Actual Positive      176                  11
```

### Tuned Threshold (`0.18`)
```
               Predicted Negative    Predicted Positive
Actual Negative      479                  334
Actual Positive      62                   125
```

---

## 5. Clinical Deployment Guidance
- Deploy the **Interpretable Logistic Regression Pipeline** paired with the calibrated **0.18 risk threshold**.
- Flag all patients scoring $\ge 0.18$ for mandatory care team intervention.
- The model captures **66.8%** of all 30-day readmissions, providing care coordination teams with maximum opportunity to deploy targeted interventions prior to discharge.
