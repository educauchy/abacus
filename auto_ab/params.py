from __future__ import annotations
from typing import List, Tuple, Any, Callable, Union
from pydantic.dataclasses import dataclass
from pydantic import validator, Field
import numpy as np

class ValidationConfig:
    validate_assignment = True
    arbitrary_types_allowed = True
    #error_msg_templates = {
    #    'value_error.any_str.max_length': 'max_length:{limit_value}',
    #}

@dataclass(config=ValidationConfig)
class DataParams:
    n_rows: int = 500
    path: str = '../notebooks/ab_data.csv'
    id_col: str = 'id'
    group_col: str = 'groups'
    target: str = 'height_now'
    target_flg: str = 'bought'
    predictors: List[str] = Field(default=['weight_now'])
    numerator: str = 'clicks'
    denominator: str = 'sessions'
    covariate: str = 'height_prev'
    target_prev: str = 'height_prev'
    predictors_prev: List[str] = Field(default=['weight_prev'])
    cluster_col: str = 'cluster_id'
    clustering_cols: List[str] = Field(default=['col1'])
    is_grouped: bool = True
    control: np.ndarray = np.array([])
    treatment: np.ndarray = np.array([])

@dataclass(config=ValidationConfig)
class ResultParams:
    to_csv: bool = True
    csv_path: str = '/app/data/internal/guide/solid_mde.csv'

@dataclass(config=ValidationConfig)
class SplitterParams:
    split_rate: float = 0.5
    name: str = 'default'

@dataclass(config=ValidationConfig)
class SimulationParams:
    n_iter: int = 100
    split_rates: List[float] = Field(default_factory=list)
    vars: List[float] = Field(default=[0, 1, 2, 3, 4, 5])
    extra_params: List = Field(default_factory=list)

@dataclass(config=ValidationConfig)
class HypothesisParams:
    alpha: float = 0.05
    beta: float = 0.2
    alternative: str = 'two-sided' # less, greater, two-sided
    split_ratios: Tuple[float, float]= Field(default=(0.5, 0.5))
    strategy: str = 'simple_test'
    strata: str = 'country'
    strata_weights: dict = Field(default={1: 0.8, 2: 0.2})
    metric_type: str = 'solid'
    metric_name: str = 'mean'
    metric: Union[Callable[[Any], Union[int,float]], str] = np.mean
    n_boot_samples: int = 200
    n_buckets: int = 50

    def __post_init__(self):
        if type(self.metric)==str:
            if self.metric=='mean':
                self.metric=np.mean
            if self.metric=='median':
                self.metric=np.median

    @validator("alpha", always=True)
    @classmethod
    def alpha_validator(cls, alpha: float):
        assert 1 > alpha > 0
        return alpha
    
    @validator("beta", always=True)
    @classmethod
    def beta_validator(cls, beta: float):
        assert 1 > beta > 0
        return beta

    @validator("alternative", always=True)
    @classmethod
    def alternative_validator(cls, alternative: float):
        assert alternative in ['two-sided', 'less', 'greater']
        return alternative
    
    @validator("split_ratios", always=True)
    @classmethod
    def split_validator(cls, split_ratios: float):
        assert len(split_ratios)==2
        assert sum(split_ratios)==1.0
        return split_ratios

    @validator("metric", always=True)
    @classmethod
    def metric_validator(cls, metric: float):
        if type(metric)==str:
            assert metric in ['mean', 'median']
            return metric
        else: 
            return metric

@dataclass
class ABTestParams:
    data_params: DataParams = DataParams()
    simulation_params: SimulationParams = SimulationParams()
    hypothesis_params: HypothesisParams = HypothesisParams()
    result_params: ResultParams = ResultParams()
    splitter_params: SplitterParams = SplitterParams()
