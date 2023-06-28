from __future__ import annotations
from typing import List, Dict, ClassVar, Optional
from pydantic.dataclasses import dataclass
from pydantic import Field
from pydantic import validator

@dataclass
class SplitBuilderParams:
    min_unique_values_in_col: ClassVar[int] = 3
    control_group_name: ClassVar[str] = "control"
    map_group_names_to_sizes: Dict[str, Optional[int]]
    main_strata_col: str
    split_metric_col: str
    metric_type: str = 'continuous'  # continuous, binary, ratio
    id_col: str = "customer_id"
    cols: List[str] = Field(default_factory=list)  # all cols for stratification
    cat_cols: List[str] = Field(default_factory=list) # this cols'll be encoded as category features
    n_bins: int = 3
    min_cluster_size: int = 100  # the number of top categories (by frequency) that will not be combined in 'other' categories
    strata_outliers_frac: float = 0.01
    alpha: float = 0.05

    def __post_init_post_parse__(self):
        self.cols.extend([self.split_metric_col])
        self.cols = list(set(self.cols))

    @validator("alpha")
    @classmethod
    def alpha_validator(cls, alpha: float):
        assert 0 < alpha < 1
        return alpha

    @validator("metric_type", always=True)
    @classmethod
    def alternative_validator(cls, metric_type: str) -> str:
        assert metric_type in ['continuous', 'binary', 'ratio'], "metric_type is not in ['continuous', 'binary', 'ratio']"
        return metric_type
