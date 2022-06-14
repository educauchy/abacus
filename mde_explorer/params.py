from typing import Any, Dict, Optional, Callable, Union
from pydantic import validator, root_validator
from pydantic.dataclasses import dataclass
from auto_ab.abtest import ABTest
from fastcore.transform import Pipeline


class ValidationConfig:
    validate_assignment = True
    arbitrary_types_allowed = True


@dataclass(config=ValidationConfig)
class MdeExplorerParams:
    """Prepilot experiment parameters class.

    Args:
        inject: inject(minimum detectable effect) that will be applied for metric_name.
        metric_name: metric name which will be affected in experemint.
        eps: the minimum distance between calculated value and max_beta_score.
        min_group_fraction: minimum possible DataFrame fraction in group.
        variance_reduction: ABTest methods for variance reduction.
        use_buckets: use bucketize method.
        transformations: pipeline of experiment. Will be calulted in __post_init__.
        stat_test: statistical test type.
        iterations_number: number of group splits per group size.
        max_beta_score: desired II type error level.
    """
    inject: float
    metric_name: str
    eps: float
    min_group_fraction: float = 0.1
    variance_reduction: Optional[Callable[[ABTest], ABTest]] = None
    use_buckets: bool = False
    transformations: Any = None
    stat_test: Callable[[ABTest], Dict[str, Union[int, float] ]] = ABTest.test_hypothesis_boot_confint
    iterations_number: int = 10
    max_beta_score: float = 0.2

    def __post_init__(self):
        if self.use_buckets:
            transformations = [self.variance_reduction, ABTest.bucketing]
        else:
            transformations = [self.variance_reduction]

        transformations = list(filter(None, transformations))
        self.transformations = Pipeline(transformations)


    @validator("variance_reduction", always=True)
    @classmethod
    def alternative_validator(cls, variance_reduction):
        assert variance_reduction in [ABTest.cuped, ABTest.cupac, None]
        return variance_reduction

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
                             ABTest.ratio_bootstrap
                            ]
        return stat_test

    @root_validator
    @classmethod
    def eps_validator(cls, values):
        eps = values.get("eps")
        assert 1.0 >= eps > 0.0, \
            "eps should be 1.0 >= eps > 0.0"
        return values

    @root_validator
    @classmethod
    def min_fraction_validator(cls, values):
        min_group_fraction = values.get("min_group_fraction")
        assert 1.0 >= min_group_fraction > 0.0, \
            "min_group_fraction should be 1.0 >= eps > 0.0"
        return values
