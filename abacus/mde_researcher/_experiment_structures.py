from pydantic.dataclasses import dataclass, Field


@dataclass
class BaseSplitElement():
    """General dataclass of experiment
    """
    group_sizes: tuple
    split_number: int
    control_group_size: int = Field(init=False)
    target_group_size: int = Field(init=False)
    def __post_init__(self):
        self.control_group_size = self.group_sizes[0]
        self.target_group_size = self.group_sizes[1]

@dataclass
class MdeAlphaExperiment(BaseSplitElement):
    """Dataclass for I type error calculactions
    """
    metric_name: str = Field(init=False)

@dataclass
class MdeBetaExperiment(MdeAlphaExperiment):
    """Dataclass for II type error calculactions
    """
    inject: float = Field(init=False)
