
import numpy as np
import pandas as pd
import pytest

from src.features.engineering import CreditFeatureEngineer


@pytest.fixture
def sample_applications():
    return pd.DataFrame({
        "credit_score": [750.0, np.nan, 680.0],
        "monthly_income": [50000.0, 100000.0, np.nan],
        "existing_emi": [10000.0, 0.0, 5000.0],
        "proposed_emi": [15000.0, 20000.0, 10000.0],
        "loan_amount": [300000.0, 600000.0, 200000.0]
    })


def test_foir_calculation(sample_applications):
    result = CreditFeatureEngineer().fit_transform(
        sample_applications
    )

    assert result.loc[0, "foir"] == pytest.approx(0.50)
    assert result.loc[1, "foir"] == pytest.approx(0.20)
    assert pd.isna(result.loc[2, "foir"])


def test_loan_to_income(sample_applications):
    result = CreditFeatureEngineer().fit_transform(
        sample_applications
    )

    assert result.loc[0, "loan_to_income"] == pytest.approx(0.50)
    assert result.loc[1, "loan_to_income"] == pytest.approx(0.50)
    assert pd.isna(result.loc[2, "loan_to_income"])


def test_ntc_flag(sample_applications):
    result = CreditFeatureEngineer().fit_transform(
        sample_applications
    )

    assert result["is_ntc"].tolist() == [0, 1, 0]


def test_original_data_unchanged(sample_applications):
    original = sample_applications.copy(deep=True)

    CreditFeatureEngineer().fit_transform(sample_applications)

    pd.testing.assert_frame_equal(sample_applications, original)


def test_zero_income_handling(sample_applications):
    sample_applications.loc[0, "monthly_income"] = 0

    result = CreditFeatureEngineer().fit_transform(
        sample_applications
    )

    assert pd.isna(result.loc[0, "foir"])
    assert pd.isna(result.loc[0, "loan_to_income"])
