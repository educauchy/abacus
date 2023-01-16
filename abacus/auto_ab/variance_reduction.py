from typing import Optional, List
import pandas as pd
import statsmodels.api as sm
from category_encoders.target_encoder import TargetEncoder


class VarianceReduction:
    def __init__(self):
        pass

    def _target_encoding(self, X: pd.DataFrame, encoding_columns: List[str], target_column: str):
        """Encodes target column.
        """
        for col in X[encoding_columns].select_dtypes(include='O').columns:
            te = TargetEncoder()
            X[col] = te.fit_transform(X[col], X[target_column])
        return X

    def _predict_target(self, X: pd.DataFrame, target_prev: str,
                       factors_prev: List[str], factors_now: List[str]) -> pd.Series:
        """Covariate prediction with linear regression model.

        Args:
            X (pandas.DataFrame): Pandas DataFrame.
            target_prev (str): Target on previous period column name.
            factors_prev (List[str]): Factor columns for modelling.
            factors_now (List[str]): Factor columns for prediction on current period.

        Returns:
            pandas.Series: Pandas Series with predicted values
        """
        Y = X[target_prev]
        X_train = X[factors_prev]
        model = sm.OLS(Y, X_train)
        results = model.fit()
        print(results.summary())
        X_predict = X[factors_now]
        return results.predict(X_predict)

    @classmethod
    def cupac(cls, X: pd.DataFrame, target_prev: str, target_now: str,
              factors_prev: List[str], factors_now: List[str], groups: str) -> pd.DataFrame:
        """ Perform CUPED on target variable with covariate calculated
        as a prediction from a linear regression model.

        Original paper: https://doordash.engineering/2020/06/08/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/.

        Args:
            X (pandas.DataFrame): Pandas DataFrame for analysis.
            target_prev (str): Target on previous period column name.
            target_now (str): Target on current period column name.
            factors_prev (List[str]): Factor columns for modelling.
            factors_now (List[str]): Factor columns for prediction on current period.
            groups (str): Groups column name.

        Returns:
            pandas.DataFrame: Pandas DataFrame with additional columns: target_pred and target_now_cuped
        """
        X = cls._target_encoding(X, list(set(factors_prev + factors_now)), target_prev)
        X.loc[:, 'target_pred'] = cls._predict_target(X, target_prev, factors_prev, factors_now)
        X_new = cls.cuped(X, target_now, groups, 'target_pred')
        return X_new

    @classmethod
    def cuped(cls, df: pd.DataFrame, target: str, groups: str,
              covariate: str) -> pd.DataFrame:
        """ Perform CUPED on target variable with predefined covariate.

        Covariate has to be chosen with regard to the following restrictions:

        1. Covariate is independent of an experiment.
        2. Covariate is highly correlated with target variable.
        3. Covariate is continuous variable.

        Original paper: https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf.

        Args:
            df (pandas.DataFrame): Pandas DataFrame for analysis.
            target (str): Target column name.
            groups (str): Groups A and B column name.
            covariate (str): Covariate column name. If None, then most correlated column in considered as covariate.

        Returns:
            pandas.DataFrame: Pandas DataFrame with additional target CUPEDed column
        """
        X = df.copy()

        cov = X[[target, covariate]].cov().loc[target, covariate]
        var = X[covariate].var()
        theta = cov / var

        for group in X[groups].unique():
            X_subdf = X[X[groups] == group]
            group_y_cuped = X_subdf[target] - theta * (X_subdf[covariate] - X_subdf[covariate].mean())
            X.loc[X[groups] == group, target] = group_y_cuped

        return X
