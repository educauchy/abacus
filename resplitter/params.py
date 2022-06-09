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
class ResplitParams:
    path: str = '../notebooks/ab_data.csv'
    strata_col: str = 'strata'
    group_col: str = 'group_col'
    test_group_value: Union[str,int] = 'B'