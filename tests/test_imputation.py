
import pandas as pd
from src.data.imputation import EmploymentIncomeImputer


def test_group_median_imputation():
    """Verify missing incomes use employment-specific medians."""

    train = pd.DataFrame({
        "employment_type": [
            "Salaried", "Salaried", "Salaried",
            "Business Owner", "Business Owner"
        ],
        "monthly_income": [
            40000, 50000, 60000,
            80000, 100000
        ]
    })

    test = pd.DataFrame({
        "employment_type": [
            "Salaried", "Business Owner", "Unknown"
        ],
        "monthly_income": [None, None, None]
    })

    imputer = EmploymentIncomeImputer()

    # Learn medians using training data only
    imputer.fit(train)

    # Transform unseen records
    result = imputer.transform(test)

    # Salaried median = 50,000
    assert result.loc[0, "monthly_income"] == 50000

    # Business Owner median = 90,000
    assert result.loc[1, "monthly_income"] == 90000

    # Unknown category uses global median = 60,000
    assert result.loc[2, "monthly_income"] == 60000

    # No missing values should remain
    assert result["monthly_income"].isna().sum() == 0

    # Original test data must remain unchanged
    assert test["monthly_income"].isna().sum() == 3


def test_existing_income_preserved():
    """Verify valid income values are not overwritten."""

    train = pd.DataFrame({
        "employment_type": ["Salaried", "Salaried"],
        "monthly_income": [40000, 60000]
    })

    test = pd.DataFrame({
        "employment_type": ["Salaried", "Salaried"],
        "monthly_income": [75000, None]
    })

    imputer = EmploymentIncomeImputer()
    imputer.fit(train)

    result = imputer.transform(test)

    assert result.loc[0, "monthly_income"] == 75000
    assert result.loc[1, "monthly_income"] == 50000


def test_training_statistics_not_updated():
    """Verify unseen data cannot change learned medians."""

    train = pd.DataFrame({
        "employment_type": ["Salaried", "Salaried"],
        "monthly_income": [40000, 60000]
    })

    test = pd.DataFrame({
        "employment_type": ["Salaried", "Salaried"],
        "monthly_income": [1000000, None]
    })

    imputer = EmploymentIncomeImputer()
    imputer.fit(train)

    result = imputer.transform(test)

    assert imputer.group_medians_["Salaried"] == 50000
    assert result.loc[0, "monthly_income"] == 1000000
    assert result.loc[1, "monthly_income"] == 50000
