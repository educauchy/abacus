import logging
import itertools
import yaml
import numpy as np
import pandas as pd
from auto_ab.analysis.params import PeriodStatTestParams

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class AbstractExperimentBuilder:
    """Base class for Experiment Builders
    """
    _POST_ANALYSIS_CONFIG_PATH = "../analysis/configs/analysis_config.yaml"
    with open(_POST_ANALYSIS_CONFIG_PATH) as f:
        _POST_ANALYSIS_CONFIG = yaml.safe_load(f)

    def __init__(self,
                 spark,
                 guests: pd.DataFrame,
                 experiment_params):
        """
        Args:
            guests: pandas dataframe that collected by PrepilotGuestsCollector
            experiment_params: parameters for experiments

        """

        self.guests = guests
        self.spark = spark
        self.experiment_params = experiment_params
        self.stat_test_params = PeriodStatTestParams(**AbstractExperimentBuilder
                                                     ._POST_ANALYSIS_CONFIG["period_stat_test_params"])
        self._group_sizes = self._build_group_sizes()

    @property
    def experiment_params(self):
        return self._experiment_params

    @experiment_params.setter
    def experiment_params(self, new_experiment_params):
        self._experiment_params = new_experiment_params
        self._group_sizes = self._build_group_sizes()

    @property
    def group_sizes(self):
        return self._group_sizes

    def _build_group_sizes(self):
        """Build list of groups sizes tuples

        Returns: list of groups sizes pairs

        """
        control = np.sort(np.arange(self.experiment_params.min_group_size,
                                    self.experiment_params.max_group_size+1,
                                    self.experiment_params.step))
        groups_split = list()
        for el in control:
            groups_split.extend(list(itertools.product([el], [el])))
        return groups_split
