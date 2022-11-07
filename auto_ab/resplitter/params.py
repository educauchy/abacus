from __future__ import annotations
from typing import Union
from pydantic.dataclasses import dataclass


class ValidationConfig:
    validate_assignment = True
    arbitrary_types_allowed = True
    #error_msg_templates = {
    #    'value_error.any_str.max_length': 'max_length:{limit_value}',
    #}

@dataclass
class GroupNames():
    test_group_name: Union[str, int] = 'test'
    control_group_name: Union[str, int] = 'control'


@dataclass(config=ValidationConfig)
class ResplitParams:
    """Resplit params class

    Args:
        group_names: group names
        strata_col: name of column with strata
        group_col: name of column with groups split
    """
    group_names: GroupNames
    strata_col: str = 'strata'
    group_col: str = 'group_col'
