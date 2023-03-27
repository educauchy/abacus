from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional
from pydantic.dataclasses import dataclass
from pydantic import validator, Field
import numpy as np

class ValidationConfig:
    validate_assignment = True
    arbitrary_types_allowed = True

@dataclass(config=ValidationConfig)
class DataParams:
    id_col: str = 'id'
    group_col: str = 'groups'
    control_name: str = 'A'
    treatment_name: str = 'B'
    is_grouped: bool = True
    strata_col: Optional[str] = ''
    target: Optional[str] = ''
    target_flg: Optional[str] = ''
    numerator: Optional[str] = ''
    denominator: Optional[str] = ''
    covariate: Optional[str] = ''
    target_prev: Optional[str] = 'target_prev'
    predictors_now: Optional[List[str]] = Field(default=['pred_now'])
    predictors_prev: Optional[List[str]] = Field(default=['pred_prev'])
    control: Optional[np.ndarray] = np.array([])
    treatment: Optional[np.ndarray] = np.array([])
    transforms: Optional[np.ndarray] = np.array([])

@dataclass(config=ValidationConfig)
class HypothesisParams:
    alpha: float = 0.05
    beta: float = 0.2
    alternative: str = 'two-sided'  # less, greater, two-sided
    metric_type: str = 'continuous'  # continuous, binary, ratio
    metric_name: str = 'mean'  # mean, median
    filter_method: Optional[str] = 'top_5'  # top_5, isolation_forest
    metric: Optional[Callable[[Any], float]] = np.mean
    metric_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    metric_transform_info: Optional[Dict[str, Dict[str, Any]]] = None
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

    # @validator("metric", always=True)
    # @classmethod
    # def metric_validator(cls,
    #                      metric: Union[Callable[[Any], Union[int,float]], str]) -> str:
    #     if type(metric) == str:
    #         assert metric in ['mean', 'median']
    #         return metric
    #     else:
    #         return metric

@dataclass
class ABTestParams:
    data_params: DataParams = DataParams()
    hypothesis_params: HypothesisParams = HypothesisParams()
