from typing import Optional, List
import pandas as pd
import statsmodels.api as sm
from category_encoders.target_encoder import TargetEncoder


class VarianceReduction:
    def __init__(self):
        pass

    def _target_encoding(self, X: pd.DataFrame, encoding_columns:List[str], target_column:str):
        """Encodes target column

        """
        for col in X[encoding_columns].select_dtypes(include='O').columns:
            te=TargetEncoder()
            X[col]=te.fit_transform(X[col],X[target_column])
        return X

    def _predict_target(self, X: pd.DataFrame, target_prev: str = '',
                       factors_prev: List[str] = None, factors_now: List[str] = None) -> pd.Series:
        """ Simple linear regression for covariate prediction

        Args:
            X: Pandas DataFrame
            target_prev: Target on previous period column name
            factors_prev: Factor columns for modelling
            factors_now: Factor columns for prediction on current period

        Returns:
            Pandas Series with predicted values
        """
        Y = X[target_prev]
        X_train = X[factors_prev]
        model = sm.OLS(Y, X_train)
        results = model.fit()
        print(results.summary())
        X_predict = X[factors_now]
        return results.predict(X_predict)

    def cupac(self, X: pd.DataFrame, target_prev: str = '', target_now: str = '',
              factors_prev: List[str] = None, factors_now: List[str] = None, groups: str = '') -> pd.DataFrame:
        """ Perform CUPAC with prediction of target column on experiment period.
        Original paper: https://doordash.engineering/2020/06/08/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/.
        Previous period = before experiment, now_period = after experiment.

        Args:
            X: Pandas DataFrame for analysis
            target_prev: Target on previous period column name
            target_now: Target on current period column name
            factors_prev: Factor columns for modelling
            factors_now: Factor columns for prediction on current period
            groups: Groups A and B column name

        Returns:
            Pandas DataFrame with additional columns: target_pred and target_now_cuped
        """
        X = self._target_encoding(X, list(set(factors_prev+factors_now)), target_prev)
        X.loc[:, 'target_pred'] = self._predict_target(X, target_prev, factors_prev, factors_now)
        X_new = self.cuped(X, target_now, groups, 'target_pred')
        return X_new

    def cuped(self, df: pd.DataFrame, target: str = '', groups: str = '',
              covariate: Optional[str] = None) -> pd.DataFrame:
        """ Perform CUPED on target column with known/unknown covariate.
        Original paper: https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf.

        Args:
            df: Pandas DataFrame for analysis
            target: Target column name
            groups: Groups A and B column name
            covariate: Covariate column name. If None, then most correlated column in considered as covariate

        Returns:
            Pandas DataFrame with additional target CUPEDed column
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
