from __future__ import annotations
from typing import List, Any, Callable
from pydantic.dataclasses import dataclass
from pydantic import validator, Field
import numpy as np

@dataclass
class MetricParams:
    type: str = 'solid'
    name: str = 'mean'

    @validator("type", always=True)
    @classmethod
    def alternative_validator(cls, type: float):
        assert type in ['solid', 'ratio']
        return type

@dataclass
class DataParams:
    n_rows: int = 500
    path: str = '../notebooks/ab_data.csv'
    id_col: str = 'id'
    group_col: str = 'groups'
    target: str = 'height_now'
    predictors: List[str] = Field(default=['weight_now'])
    numerator: str = 'clicks'
    denominator: str = 'sessions'
    covariate: str = 'height_prev'
    target_prev: str = 'height_prev'
    predictors_prev: List[str] = Field(default=['weight_prev'])
    is_grouped: bool = True

@dataclass
class ResultParams:
    to_csv: bool = True
    csv_path: str = '/app/data/internal/guide/solid_mde.csv'

@dataclass
class SplitterParams:
    split_rate: float = 0.5

@dataclass
class SimulationParams:
    n_iter: int = 100
    split_rates: List[float] = Field(default_factory=list)
    vars: List[float] = Field(default=[0, 1, 2, 3, 4, 5])
    extra_params: List = Field(default_factory=list)

@dataclass
class BootstrapParams:
    metric: Callable[[Any], float] = np.mean
    n_boot_samples: int = 200

    @validator("metric", always=True)
    @classmethod
    def metric_validator(cls, metric: float):
        if type(metric)==str:
            assert metric in ['mean', 'median']
            return metric
        else: 
            return metric

    def __post_init__(self):
        if type(self.metric)==str:
            if self.metric=='mean':
                self.metric=np.mean
            if self.metric=='median':
                self.metric=np.median

@dataclass 
class HypothesisParams:
    alpha: float = 0.05
    beta: float = 0.2
    alternative: str = 'two-sided' # less, greater, two-sided
    split_ratios: List[float] = Field(default=[0.5, 0.5])
    strategy: str = 'simple_test'
    strata: str = 'country'
    strata_weights: dict = Field(default={'US': 0.8, 'UK': 0.2})
    n_boot_samples: int = 200
    n_buckets: int = 50

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
        assert sum(split_ratios)==1.0
        return split_ratios

@dataclass
class ABTestParams:
    metric_params: MetricParams = MetricParams()
    data_params: DataParams = DataParams()
    simulation_params: SimulationParams = SimulationParams()
    hypothesis_params: HypothesisParams = HypothesisParams()
    result_params: ResultParams = ResultParams()
    splitter_params: SplitterParams = SplitterParams()
    bootstrap_params: BootstrapParams = BootstrapParams()
