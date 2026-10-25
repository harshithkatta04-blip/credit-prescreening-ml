
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create credit affordability and NTC features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Preserve whether the credit score was originally missing
        X["is_ntc"] = X["credit_score"].isna().astype(int)

        # Ensure the denominator is valid
        income = X["monthly_income"].where(
            X["monthly_income"] > 0, np.nan
        )

        # FOIR as a ratio (0.40 means 40%)
        X["foir"] = (
            X["existing_emi"] + X["proposed_emi"]
        ) / income

        # Requested loan / estimated annual income
        X["loan_to_income"] = (
            X["loan_amount"] / (income * 12)
        )

        return X
