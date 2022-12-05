from __future__ import annotations
from scipy.stats import ttest_ind, mannwhitneyu
from typing import List, Dict, ClassVar, Callable, Tuple, Optional
from pydantic.dataclasses import dataclass
from pydantic import Field
from pydantic import validator

@dataclass
class SplitBuilderParams:
    _acceptable_stat_tests: ClassVar[Tuple[Callable]] = (ttest_ind, mannwhitneyu)  # for num col
        
    min_unique_values_in_col: ClassVar[int] = 3
    control_group_name: ClassVar[str] = "control"

    map_group_names_to_sizes: Dict[str, Optional[int]]
    main_strata_col: str
    split_metric_col: str
    id_col: str = "customer_id"
    cols: List[str] = Field(default_factory=list)  # all cols for stratification
    cat_cols: List[str] = Field(default_factory=list) # this cols'll be encoded as category features
    bin_min_size: int = 100
    n_bins_rto: int = 3
    pvalue: float = 0.05
    n_top_cat: int = 100  # the number of top categories (by frequency) that will not be combined in 'other' categories
    stat_test: str = "ttest"  # name of scipy function

    def __post_init_post_parse__(self):
        self.cols.extend([self.split_metric_col])
        self.cols = list(set(self.cols))

    @validator("stat_test")
    @classmethod
    def stat_test_validator(cls, stat_test: List[str]):
        _acceptable_stat_tests = list(map(lambda f: f.__name__, cls._acceptable_stat_tests))
        assert stat_test in _acceptable_stat_tests, \
        f"Not acceptable {stat_test}, available {_acceptable_stat_tests}."
        return stat_test

    @validator("map_group_names_to_sizes")
    @classmethod
    def map_group_names_to_sizes_validator(cls, map_group_names_to_sizes: List[str]):
        """It's necessary at least one group with naming cls.control_group_name.
        Target group names are arbitrary.

        Args:
            map_group_names_to_sizes: group name to size

        Returns: map_group_names_to_sizes

        """
        assert cls.control_group_name in map_group_names_to_sizes, \
        f"{cls.control_group_name} always required"
        return map_group_names_to_sizes

    @validator("pvalue")
    @classmethod
    def pvalue_validator(cls, pvalue: float):
        assert 1 > pvalue > 0
        return pvalue
