from typing import List
from pydantic import root_validator
from pydantic.dataclasses import dataclass


@dataclass
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
    iterations_number: int = 10
    max_beta_score: float = 0.2
    min_beta_score: float = 0.05

    @root_validator
    @classmethod
    def groups_sizes_validator(cls, values):
        min_group_size = values.get("min_group_size")
        max_group_size = values.get("max_group_size")
        assert max_group_size > min_group_size, \
            "max_group_size should be more than min_group_size"
        return values
