
# Credit Pre-Screening Engine
### Risk-Aware Loan Intake Optimisation Using Machine Learning

An end-to-end machine learning project focused on improving personal loan application screening through predictive modelling, risk-based decision thresholds, and fairness evaluation.

**Domain:** Banking, Financial Services and Insurance (BFSI)  
**Project Type:** Supervised Learning | Classification  
**Status:** In Development  
**Origin:** Calibo AI Academy — Phase 2 Mini Use Case 02

---

## 1. Business Problem

A mid-sized Indian private bank processes approximately 3,000 personal loan applications every month.

Currently, each application undergoes manual credit assessment, requiring approximately 25 minutes per application.

Nearly half of the applications are declined, with many straightforward cases identifiable early in the review process.

This creates:
- Unnecessary analyst workload
- Operational inefficiency
- Potential delays in loan decisions

The objective is to reduce full manual reviews while controlling the risk of incorrect automated decisions.

## 2. Proposed Solution

Develop a machine learning-based credit pre-screening engine with three decision lanes.

| Decision Lane | Action |
|---|---|
| Auto-Decline | Automatically decline applications with sufficiently low predicted approval probability |
| Manual Review | Route uncertain applications to credit analysts |
| Fast-Track | Route applications with sufficiently high predicted approval probability for a light-touch approval check |

Decision thresholds will be selected based on the bank's specified risk limits.

### Business Constraints

- Wrong auto-decline rate must not exceed 5%.
- Wrong fast-track rate must not exceed 5%.
- Maximise applications removed from full manual review while satisfying both constraints.
- Evaluate fairness across applicant groups.

The error limits are measured against historical analyst decisions, not actual repayment outcomes.

## 3. Dataset

The assignment provides a simulated dataset containing 12,000 personal loan applications collected between April and September 2026.

### Important Features

| Feature | Description |
|---|---|
| credit_score | Applicant's credit bureau score |
| monthly_income | Declared gross monthly income |
| existing_emi | Existing monthly loan repayments |
| proposed_emi | Estimated EMI for the requested loan |
| employment_type | Applicant's employment category |
| delinquencies_24m | Late or missed repayments in the past 24 months |
| loan_amount | Requested loan amount |
| credit_history_months | Length of credit history |
| decision | Historical analyst decision: Approved or Declined |

### Engineered Features

**Fixed Obligation to Income Ratio (FOIR)**

FOIR = (Existing EMI + Proposed EMI) / Monthly Income

**Loan-to-Income Ratio**

Loan-to-Income = Loan Amount / (Monthly Income × 12)

**New-to-Credit Indicator**

is_ntc = 1 when credit score is missing, otherwise 0.

The provided dataset is kept outside Git until redistribution permissions are confirmed.

## 4. Project Architecture

```text
credit-prescreening-ml/
|
|-- data/
|   |-- raw/
|   |-- processed/
|
|-- notebooks/
|   |-- 01_credit_prescreening.ipynb
|
|-- src/
|   |-- data/
|   |-- features/
|   |-- models/
|   |-- evaluation/
|
|-- tests/
|-- models/
|-- reports/
|   |-- figures/
|
|-- .gitignore
|-- pyproject.toml
|-- requirements.txt
|-- README.md
```

The notebook follows the assignment workflow, while reusable logic will be implemented in Python modules under `src/`.

## 5. Machine Learning Methodology

### Phase 1 — Business Understanding
- Complete the CBIM Problem Canvas
- Identify stakeholders and success criteria
- Define business risks and limitations

### Phase 2 — Data Preparation
- Data quality assessment
- Missing value treatment
- Exploratory data analysis
- Feature engineering

### Phase 3 — Baseline Modelling
- Logistic Regression baseline
- Classification metrics
- Confusion matrix analysis
- Baseline auto-decline threshold selection

### Phase 4 — Model Comparison
Evaluate:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Use 5-fold Stratified Cross-Validation on the training set.

Primary evaluation measures:
- ROC-AUC
- Accuracy
- Precision
- Recall

### Phase 5 — Risk-Based Decision Optimisation
- Auto-decline threshold optimisation
- Fast-track threshold optimisation
- Three-lane application routing
- Model comparison against the baseline
- Analyst workload impact calculation

### Phase 6 — Fairness and Business Evaluation
- Historical approval disparities
- Group-wise automated decline rates
- Potential proxy-variable bias
- Operational savings
- Credit policy recommendation

### Optional Analysis
K-Means segmentation to identify applicant groups where model performance requires additional human oversight.

## 6. Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | scikit-learn |
| Model Persistence | joblib |
| Experimentation | Jupyter Notebook |
| Testing | pytest |
| Version Control | Git, GitHub |

## 7. Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd credit-prescreening-ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Place the course-provided dataset in `data/raw/`.

Launch Jupyter:

```bash
jupyter lab
```

Open `notebooks/01_credit_prescreening.ipynb`.

## 8. Results

Model evaluation results, selected thresholds, fairness findings and estimated analyst hours saved will be added after experiments are completed.

No model is currently claimed to be production-ready.

## 9. Limitations

- Historical credit decisions are used as reference labels.
- Actual repayment and default outcomes are unavailable.
- Historical decisions may contain policy biases.
- Model agreement with historical decisions does not guarantee lending safety.
- Independent validation, monitoring and compliance review are necessary before real-world deployment.

## 10. Project Deliverables

- CBIM Problem Canvas
- Reproducible Jupyter Notebook
- Model comparison and threshold analysis
- Fairness evaluation
- Business insight summary
- Credit policy recommendation

---

**Project Status:** Repository setup and business understanding.
