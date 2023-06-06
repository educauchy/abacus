from typing import Any, Callable, Dict, Iterable, List, Tuple, Union, Optional
import numpy as np
import pandas as pd


# auto_ab types
MetricNameType = Union[str, Callable[[np.ndarray], Union[int, float]]]
StatTestType = Dict[str, Optional[Union[int, float]]]
