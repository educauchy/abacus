from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional, Iterable, Union
from pydantic.dataclasses import dataclass
from pydantic import validator, Field
import numpy as np

class ValidationConfig:
    validate_assignment = True
    arbitrary_types_allowed = True

@dataclass(config=ValidationConfig)
class DataParams:
    """Data description as column names of dataset generated during experiment.

    Parameters:
        id_col (str): ID of observations.
        group_col (str): Group of experiment.
        control_name (str): Name of control group in ``group_col``.
        treatment_name (str): Name of treatment group in ``group_col``.
        is_grouped (bool, Optional): Flag that shows whether observations are grouped.
        strata_col (str, Optional): Name of stratification column. Stratification column must be categorical.
        target (str, Optional): Target column name of continuous metric.
        target_flg (str, Optional): Target flag column name of binary metric.
        numerator (str, Optional): Numerator for ratio metric.
        denominator (str, Optional): Denominator for ratio metric.
        covariate (str, Optional): Covariate column for CUPED.
        target_prev (str, Optional): Target column name for previous period of continuous metric.
        predictors_now (List[str], Optional): List of columns to predict covariate.
        predictors_prev (List[str], Optional): List of columns to create linear model for covariate prediction.
        control (np.ndarray, Optional): Control group data used for quick access and excluding querying dataset.
        treatment (np.ndarray, Optional): Treatment group data used for quick access and excluding querying dataset.
        transforms (np.ndarray, Optional): List of transformations applied to experiment.
    """
    id_col: str = 'id'
    group_col: str = 'groups'
    control_name: str = 'A'
    treatment_name: str = 'B'
    is_grouped: Optional[bool] = True
    strata_col: Optional[str] = ''
    target: Optional[str] = ''
    target_flg: Optional[str] = ''
    numerator: Optional[str] = ''
    denominator: Optional[str] = ''
    covariate: Optional[str] = ''
    target_prev: Optional[str] = ''
    predictors_now: Optional[List[str]] = Field(default=['pred_now'])
    predictors_prev: Optional[List[str]] = Field(default=['pred_prev'])
    control: Optional[np.ndarray] = np.array([])
    treatment: Optional[np.ndarray] = np.array([])
    transforms: Optional[np.ndarray] = np.array([])

@dataclass(config=ValidationConfig)
class HypothesisParams:
    """Description of hypothesis parameters.

    Parameters:
        alpha (float): type I error.
        beta (float): type II error.
        alternative (str): directionality of hypothesis: less, greater, two-sided.
        metric_type (str): metric type: continuous, binary, ratio.
        metric_name (str): metric name: mean, median. If custom metric, then use here appropriate name.
        metric (Callable[[Iterable[float]], np.ndarray], Optional): if metric_name is custom, then you must define metric function.
        metric_transform (Callable[[np.ndarray], np.ndarray], Optional): applied transformations to experiment.
        metric_transform_info (Dict[str, Dict[str, Any]], Optional): information of applied transformations.
        filter_method (str, Optional): method for filtering outliers: top_5, isolation_forest.
        n_boot_samples (int, Optional): number of bootstrap iterations.
        n_buckets (int, Optional): number of buckets.
        strata (str, Optional): stratification column.
        strata_weights (Dict[str, float], Optional): historical strata weights.
    """
    alpha: float = 0.05
    beta: float = 0.2
    alternative: str = 'two-sided'  # less, greater, two-sided
    metric_type: str = 'continuous'  # continuous, binary, ratio
    metric_name: str = 'mean'  # mean, median
    metric: Optional[Callable[[Iterable[float]], float]] = np.mean
    metric_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    metric_transform_info: Optional[Dict[str, Dict[str, Any]]] = None
    filter_method: Optional[str] = 'top_5'  # top_5, isolation_forest
    n_boot_samples: Optional[int] = 200
    n_buckets: Optional[int] = 50
    strata: Optional[str] = ''
    strata_weights: Optional[Dict[str, float]] = Field(default={'1': 0.8, '2': 0.2})

    def __post_init__(self):
        if self.metric_name == 'mean':
            self.metric = np.mean
        if self.metric_name == 'median':
            self.metric = np.median

    @validator("alpha", always=True)
    @classmethod
    def alpha_validator(cls, alpha: float) -> float:
        assert 1 > alpha > 0
        return alpha

    @validator("beta", always=True)
    @classmethod
    def beta_validator(cls, beta: float) -> float:
        assert 1 > beta > 0
        return beta

    @validator("alternative", always=True)
    @classmethod
    def alternative_validator(cls, alternative: str) -> str:
        assert alternative in ['two-sided', 'less', 'greater']
        return alternative

    @validator("metric", always=True)
    @classmethod
    def metric_validator(cls,
                         metric: Union[Callable[[Any], Union[int,float]], str]) -> str:
        if type(metric) == str:
            assert metric in ['mean', 'median']
            return metric
        else:
            return metric

@dataclass
class ABTestParams:
    data_params: DataParams = DataParams()
    hypothesis_params: HypothesisParams = HypothesisParams()
