# Credit Pre-Screening Engine — Loan Intake Optimisation

An end-to-end machine learning project for analysing historical personal-loan decisions and designing a selective credit pre-screening system.

**Domain:** Banking & Financial Services (BFSI)  
**Project:** Calibo AI Academy — Phase 2, Mini Use Case 02  
**Tech Stack:** Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

## 1. Business Problem

A retail lender processes approximately **3,000 personal-loan applications per month**, with each manual review taking around **25 minutes**.

The objective is to reduce manual-review workload by routing applications into three lanes:

- **Auto-decline:** High-confidence historical decline predictions.
- **Fast-track:** High-confidence historical approval predictions, subject to a light-touch check.
- **Manual review:** Applications that require human judgement.

Each automated lane must have **no more than 5% disagreement with historical credit-team decisions**.

## 2. Dataset

The project uses a simulated dataset containing **12,000 loan applications** from April–September 2026.

- Historically approved: **52.94%**
- Historically declined: **47.06%**
- Target: `decision` — Approved (0), Declined (1)

**Important:** The target represents historical analyst decisions, not actual loan defaults or repayment outcomes.

## 3. Project Workflow

1. **Data Quality:** Investigated missing income, new-to-credit applicants and unusual income values.
2. **Exploratory Analysis:** Studied approval patterns across credit score, employment, affordability and other applicant characteristics.
3. **Feature Engineering:** Created FOIR, loan-to-income ratio and new-to-credit indicator.
4. **Baseline Modelling:** Trained and evaluated Logistic Regression using a reusable preprocessing pipeline.
5. **Fairness Audit:** Compared historical and model outcomes across gender groups.
6. **Model Comparison:** Evaluated four classifiers using five-fold stratified cross-validation.
7. **Two-Lane Optimisation:** Searched approval-probability thresholds under separate 5% disagreement constraints.
8. **Applicant Segmentation:** Used K-Means to profile seven applicant groups and analyse classification errors.
9. **Validation Audit:** Re-selected thresholds using training-only out-of-fold predictions and evaluated the fixed rules on the test split.

## 4. Model Comparison

Five-fold stratified cross-validation on training data:

| Model | ROC-AUC | Accuracy |
|---|---:|---:|
| Gradient Boosting | **0.8897** | **80.03%** |
| Random Forest | 0.8779 | 79.42% |
| Logistic Regression | 0.8676 | 77.46% |
| Decision Tree | 0.7232 | 72.42% |

**Selected candidate:** Gradient Boosting, based on its stronger cross-validated discrimination.

On the test split, Gradient Boosting achieved **80.29% accuracy** and **0.8853 ROC-AUC**.

Its principal features included recent delinquencies, credit score and FOIR.

## 5. Final Threshold Validation

Thresholds were selected using five-fold out-of-fold predictions from training data. They were then applied without adjustment to the test split.

All thresholds refer to the **predicted probability of approval**.

| Metric | Gradient Boosting | Logistic Regression |
|---|---:|---:|
| Auto-decline below | 0.15 | 0.11 |
| Fast-track above | 0.88 | 0.91 |
| Auto-decline disagreement | 4.21% | 5.44% |
| Fast-track disagreement | 5.14% | 4.15% |
| Automation coverage | 43.00% | 30.12% |
| Both lanes satisfy 5%? | **No** | **No** |

Gradient Boosting routed 546 test applications to auto-decline, 486 to fast-track and 1,368 to manual review.

Its **43% potential automation coverage** corresponds to approximately **1,290 full reviews or 537.5 gross analyst-hours avoided per month**, assuming comparable application volumes. This estimate is conditional and does not account for light-touch review time.

**Neither complete two-lane configuration is ready for deployment**, because each model exceeds the disagreement limit in one automated lane.

## 6. Fairness and Segmentation

The historical approval rate was **50.43% for female applicants** and **54.10% for male applicants**.

At the earlier exploratory 0.90 decline-probability threshold, auto-decline rates were 15.73% for females and 14.67% for males. These descriptive results do not establish model fairness; the final two-lane configuration requires a separate fairness audit.

K-Means segmentation identified seven applicant groups. Applicants with unusual or unverified income values were highlighted for human verification, while higher-income and younger segments showed opportunities for further model-error analysis.

## 7. Repository Structure

```text
credit-prescreening-ml/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_credit_prescreening.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   │   └── imputation.py
│   ├── features/
│   │   └── engineering.py
│   └── evaluation/
├── tests/
│   ├── test_imputation.py
│   └── test_feature_engineering.py
├── README.md
├── requirements.txt
└── pyproject.toml
```

## 8. Setup and Execution

Clone the repository:

```bash
git clone https://github.com/harshithkatta04-blip/credit-prescreening-ml.git
cd credit-prescreening-ml
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Open `notebooks/01_credit_prescreening.ipynb` in VS Code or Jupyter and run the cells sequentially.

The dataset is expected at:

`data/raw/MUC02_Credit_PreScreening_Dataset.csv`

Run automated tests:

```bash
python -m pytest tests/ -v
```

## 9. Limitations and Future Work

- Historical approval decisions are not ground-truth repayment outcomes.
- Earlier exploratory threshold searches used the test labels; the later audit used training-only out-of-fold threshold selection.
- The test split was previously inspected, so fresh independent validation remains necessary.
- Automated-lane error rates require further stability, fairness and calibration assessments.
- Extreme-income records require additional data-quality verification.
- A controlled, human-supervised pilot is recommended before any operational deployment.

## 10. Conclusion

Gradient Boosting demonstrated stronger prediction performance and greater potential automation coverage than Logistic Regression.

However, the final audit showed that **neither proposed two-lane configuration fully satisfies the bank's 5% historical-disagreement requirement**.

The project therefore recommends further validation and a human-supervised pilot rather than immediate automated lending decisions.

**Project status:** End-to-end research and modelling completed; production deployment not recommended without further validation.