
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class EmploymentIncomeImputer(BaseEstimator, TransformerMixin):
    """Fill missing income using employment-specific training medians."""

    def fit(self, X, y=None):
        X = X.copy()

        self.group_medians_ = (
            X.groupby("employment_type")["monthly_income"]
            .median()
        )

        self.global_median_ = X["monthly_income"].median()

        if pd.isna(self.global_median_):
            raise ValueError("Cannot fit: all training incomes are missing.")

        return self

    def transform(self, X):
        check_is_fitted(self, ["group_medians_", "global_median_"])

        X = X.copy()

        group_medians = X["employment_type"].map(
            self.group_medians_
        )

        X["monthly_income"] = (
            X["monthly_income"]
            .fillna(group_medians)
            .fillna(self.global_median_)
        )

        return X
