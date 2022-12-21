import pandas as pd
from auto_ab.splitter.params import SplitBuilderParams
from auto_ab.prepilot.prepilot_experiment_builder import PrepilotExperimentBuilder
from auto_ab.prepilot.params import PrepilotParams
from auto_ab.mde_explorer.params import MdeExplorerParams
from auto_ab.auto_ab.params import ABTestParams

_DEFAULT_SIZE = 1

class MdeExplorerBuilder():
    """Finds the nearest and the smallest group size.
    """

    def __init__(self,
                 guests: pd.DataFrame,
                 abtest_params: ABTestParams,
                 experiment_params: MdeExplorerParams,
                 stratification_params: SplitBuilderParams):
        """
        Args:
            guests: DataFrame that shold be split for control and test groups.
            abtest_params: params for A/B test that will be calculated in split process.
            experiment_params: params for MDE searching.
            stratification_params: params for stratification and splitting data.
        """
        self.guests = guests
        self.abtest_params = abtest_params
        self.experiment_params = experiment_params
        self.stratification_params = stratification_params
        self._prepilot_params = PrepilotParams(
            metrics_names=[self.experiment_params.metric_name],
            injects=[self.experiment_params.inject],
            min_group_size=_DEFAULT_SIZE, 
            max_group_size=_DEFAULT_SIZE, 
            step=_DEFAULT_SIZE,
            variance_reduction = self.experiment_params.variance_reduction,
            use_buckets = self.experiment_params.use_buckets,
            stat_test = self.experiment_params.stat_test,
            iterations_number = self.experiment_params.iterations_number,
            max_beta_score=self.experiment_params.max_beta_score,
        )

    def _calc_beta(self, group_size:int) -> float:
        """Calculations of II type error

        Args:
            group_size: size of group that will be pick out to test and control groups.

        Returns:
            II type error for group_size.
        """
        self._prepilot_params.min_group_size = group_size
        self._prepilot_params.max_group_size = group_size
        prepilot = PrepilotExperimentBuilder(self.guests, 
                                        self.abtest_params,
                                        self._prepilot_params,
                                        self.stratification_params)
        beta, _ = prepilot.collect(fill_with_default=False)

        return beta.values[0][0]
    
    def collect(self) -> pd.DataFrame:
        """Finds group_size that fit for MDE.

        Returns:
            DataFrame with 3 latest calculated group sizes.
        """
        max_diff_beta = 1.0
        group_size = len(self.guests)//2
        min_size = len(self.guests)*self.experiment_params.min_group_fraction//2
        result = []

        while max_diff_beta>=self.experiment_params.eps and group_size>=min_size:
            curr_beta = self._calc_beta(group_size)
            max_diff_beta = self.experiment_params.max_beta_score-curr_beta
            if len(result)==3:
                result.pop(0)
            result.append([group_size, curr_beta])

            if max_diff_beta<=self.experiment_params.eps:
                break
            if max_diff_beta>0.0:
                group_size = group_size//2
            elif max_diff_beta<0.0:
                group_size = (len(self.guests) + group_size)//2

        return pd.DataFrame(result, columns=["group_size", "beta_error"])
