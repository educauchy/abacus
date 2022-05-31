from typing import List, Any, Dict, Optional, Callable, Union
from click import Abort
from pydantic import root_validator, validator
from pydantic.dataclasses import dataclass
import numpy as np
from auto_ab.abtest import ABTest
from fastcore.transform import Pipeline


class ValidationConfig:
    validate_assignment = True
    arbitrary_types_allowed = True
    #error_msg_templates = {
    #    'value_error.any_str.max_length': 'max_length:{limit_value}',
    #}

@dataclass(config=ValidationConfig)
class PrepilotParams:
    """Prepilot experiment parameters class.

    Args:
        metrics_names: metrics which will be compare in experiments.
        injects: injects represent MDE values.
        min_group_size: minimal value of groups sizes
        max_group_size: maximal value of groups sizes
        step: Spacing between min_group_size and max_group_size
        iterations_number: count of splits for each element in group_sizes
        max_beta_score: max level of II type error
        min_beta_score: min level of II type error, that will be calculated
        for greater group size, if it has been found on earlier
        and data haven't any changes turn to True otherwise False
    """
    metrics_names: List[str]
    injects: List[float]
    min_group_size: int
    max_group_size: int
    step: int
    variance_reduction: Optional[Callable[[ABTest], ABTest]] = None
    use_buckets: bool = False
    transformations: Any = None
    stat_test: Callable[[ABTest], Dict[str, Union[int, float] ]] = ABTest.test_hypothesis_boot_confint
    iterations_number: int = 10
    max_beta_score: float = 0.2
    min_beta_score: float = 0.05

    def __post_init__(self):
        if self.use_buckets:
            transformations = [self.variance_reduction, ABTest.bucketing]
        else:
            transformations = [self.variance_reduction]
        transformations = list(filter(None, transformations))
        self.transformations = Pipeline(transformations)

    @validator("stat_test", always=True)
    @classmethod
    def alternative_validator(cls, stat_test):
        assert stat_test in [ABTest.test_hypothesis_boot_confint, 
                             ABTest.test_hypothesis_boot_est,
                             ABTest.test_hypothesis_strat_confint,
                             ABTest.test_hypothesis_mannwhitney,
                             ABTest.test_hypothesis_ttest,
                             ABTest.delta_method,
                             ABTest.ratio_taylor,
                             ABTest.ratio_bootstrap,
                             ABTest.test_boot_hypothesis
                            ]
        return stat_test

    @root_validator
    @classmethod
    def groups_sizes_validator(cls, values):
        min_group_size = values.get("min_group_size")
        max_group_size = values.get("max_group_size")
        assert max_group_size > min_group_size, \
            "max_group_size should be more than min_group_size"
        return values
