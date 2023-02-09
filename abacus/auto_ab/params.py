from __future__ import annotations
from typing import List, Tuple, Any, Callable, Union, Optional
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
    strata_col: Optional[str] = 'country'
    target: str = 'height_now'
    target_flg: str = 'bought'
    predictors: List[str] = Field(default=['weight_now'])
    numerator: str = 'clicks'
    denominator: str = 'sessions'
    covariate: str = 'height_prev'
    target_prev: str = 'height_prev'
    predictors_prev: List[str] = Field(default=['weight_prev'])
    is_grouped: bool = True
    control: Optional[np.ndarray] = np.array([])
    treatment: Optional[np.ndarray] = np.array([])

@dataclass(config=ValidationConfig)
class HypothesisParams:
    alpha: float = 0.05
    beta: float = 0.2
    alternative: str = 'two-sided'  # less, greater, two-sided
    strata: Optional[str] = 'country'
    strata_weights: Optional[dict] = Field(default={1: 0.8, 2: 0.2})
    metric_type: str = 'solid'
    metric_name: str = 'mean'
    metric: Union[Callable[[Any], Union[int,float]], str] = np.mean
    n_boot_samples: Optional[int] = 200
    n_buckets: Optional[int] = 50

    def __post_init__(self):
        if type(self.metric) == str:
            if self.metric == 'mean':
                self.metric = np.mean
            if self.metric == 'median':
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
