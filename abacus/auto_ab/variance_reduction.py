from typing import List
import pandas as pd
import statsmodels.api as sm
from category_encoders.target_encoder import TargetEncoder


class VarianceReduction:
    def __init__(self):
        pass

    @staticmethod
    def _target_encoding(x: pd.DataFrame, encoding_columns: List[str], target_column: str):
        """Encodes target column.
        """
        for col in x[encoding_columns].select_dtypes(include='O').columns:
            te = TargetEncoder()
            x[col] = te.fit_transform(x[col], x[target_column])
        return x

    @staticmethod
    def _predict_target(x: pd.DataFrame, target_prev: str,
                        factors_prev: List[str], factors_now: List[str]) -> pd.Series:
        """Covariate prediction with linear regression model.

        Args:
            x (pandas.DataFrame): Pandas DataFrame.
            target_prev (str): Target on previous period column name.
            factors_prev (List[str]): Factor columns for modelling.
            factors_now (List[str]): Factor columns for prediction on current period.

        Returns:
            pandas.Series: Pandas Series with predicted values
        """
        y = x[target_prev]
        x_train = x[factors_prev]
        model = sm.OLS(y, x_train)
        results = model.fit()
        print(results.summary())
        x_predict = x[factors_now]
        return results.predict(x_predict)

    @classmethod
    def cupac(cls, x: pd.DataFrame, target_prev: str, target_now: str,
              factors_prev: List[str], factors_now: List[str], groups: str) -> pd.DataFrame:
        """ Perform CUPED on target variable with covariate calculated
        as a prediction from a linear regression model.

        Original paper: https://doordash.engineering/2020/06/08/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/.

        Args:
            x (pandas.DataFrame): Pandas DataFrame for analysis.
            target_prev (str): Target on previous period column name.
            target_now (str): Target on current period column name.
            factors_prev (List[str]): Factor columns for modelling.
            factors_now (List[str]): Factor columns for prediction on current period.
            groups (str): Groups column name.

        Returns:
            pandas.DataFrame: Pandas DataFrame with additional columns: target_pred and target_now_cuped
        """
        x = cls._target_encoding(x, list(set(factors_prev + factors_now)), target_prev)
        x.loc[:, 'target_pred'] = cls._predict_target(x, target_prev, factors_prev, factors_now)
        x_new = cls.cuped(x, target_now, groups, 'target_pred')
        return x_new

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
        x = df.copy()

        cov = x[[target, covariate]].cov().loc[target, covariate]
        var = x[covariate].var()
        theta = cov / var

        for group in x[groups].unique():
            x_subdf = x[x[groups] == group]
            group_y_cuped = x_subdf[target] - theta * (x_subdf[covariate] - x_subdf[covariate].mean())
            x.loc[x[groups] == group, target] = group_y_cuped

        return x
