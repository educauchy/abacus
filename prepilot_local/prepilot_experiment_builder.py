from typing import Union
import itertools
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from stratification.params import SplitBuilderParams
from prepilot_local.experiment_structures import PrepilotAlphaExperiment, PrepilotBetaExperiment
from prepilot_local.abstract_experiment_builder import AbstractExperimentBuilder
from prepilot_local.prepilot_split_builder import PrepilotSplitBuilder
from prepilot_local.params import PrepilotParams
from analysis.abtest import ABTest


class PrepilotExperimentBuilder(AbstractExperimentBuilder):
    """Calculates I and II type errors for different group sizes and injects
    """

    def __init__(self,
                 guests: DataFrame,
                 abtest_params,
                 experiment_params: PrepilotParams,
                 stratification_params: SplitBuilderParams):
        """
        Args:
            guests: dataframe that collected by PrepilotGuestsCollector
            experiment_params: prameters for prepilot experiments

        """
        super().__init__(guests, experiment_params)
        self.abtest_params = abtest_params
        self.stratification_params = stratification_params
        self._number_of_decimals = 10


    def _calc_experiment_grid_cell(self,
                                   guests_with_splits: pd.DataFrame,
                                   grid_element: Union[PrepilotBetaExperiment, PrepilotAlphaExperiment],
                                   use_buckets: bool = False):
        """Calculates stat test forr one experiment grid element

        Args:
            guests_with_splits: dataframe with calculated splits for experiment

        Returns: pandas DataFrame with calculated stat test and experiment parameters

        """
        #ab_test = ABTest(guests_with_splits, self.abtest_params, startup_config=True)

        row_dict = {
            "metric": [grid_element.metric_name],
            "split_rate": [(grid_element.control_group_size, grid_element.target_group_size)]
            }
        split_column = f"is_control_{grid_element.control_group_size}_{grid_element.target_group_size}_{grid_element.split_number}"
        metric_col = f"{grid_element.metric_name}"
        guests_with_splits[split_column] = (guests_with_splits[split_column]
                                            .map({1: 'A', 0: 'B'})
        )
        self.abtest_params['data']['group_col'] = split_column
        self.abtest_params['data']['target'] = metric_col

        ab_test = ABTest(guests_with_splits, self.abtest_params, startup_config=True)
        ab_test = self.experiment_params.transformations(ab_test)

        if isinstance(grid_element, PrepilotBetaExperiment):
            ab_test.config['treatment'] = ab_test.config['treatment']*grid_element.inject
            row_dict["MDE"] = [grid_element.inject]
        row_dict["effect_significance"] = self.experiment_params.stat_test(ab_test, self.experiment_params.bootstrap_metric)
        return pd.DataFrame(row_dict)

    def _fill_passed_experiments(self, aggregated_df):
        """Fill Nan for passed experiments

        Args:
            aggregated_df: dataframe with calculated experiments

        Returns: pandas DataFrame with filled values

        """
        for metric in self._experiment_params.metrics_names:
            for split in self.group_sizes:
                passed_mde = aggregated_df[(aggregated_df.metric == metric) &
                                           (aggregated_df.split_rate == split)]["MDE"].values
                last_experiment = min(passed_mde)
                passed_injects = np.setdiff1d(self._experiment_params.injects, passed_mde)
                if len(passed_injects) > 0:
                    failed = list(itertools.product([metric], [split], passed_injects[passed_injects < last_experiment],
                                                    [f">={self._experiment_params.max_beta_score}"])
                                 )  # error higher than max_beta_score
                    succes = list(itertools.product([metric], [split], passed_injects[passed_injects > last_experiment],
                                                    [f"<={self._experiment_params.min_beta_score}"])
                                 )
                    succes.extend(failed)
                    df_passed = pd.DataFrame.from_records(succes)
                    df_passed.columns = aggregated_df.columns
                    aggregated_df = pd.concat([aggregated_df, df_passed])
        return aggregated_df

    def _beta_score_calculation(self, df_with_calc: pd.DataFrame):
        """Calculates II type error for df with calculated experiments

        Args:
            beta_scores: dataframe with calculated experiments

        Returns: df with II type error

        """
        res_agg = (df_with_calc
                   .groupby(by=["metric", "split_rate", "MDE"])
                   .agg(sum=("effect_significance", sum),
                        count=("effect_significance", "count"))
                  )
        res_agg["beta"] = round((1.0 - res_agg["sum"]/res_agg["count"]),
                                self._number_of_decimals)
        return res_agg

    @staticmethod
    def _fill_res_with_default(df: pd.DataFrame,
                           column_name: str,
                           min_val: float,
                           max_val: float):
        """Fill column with defalt values

        Args:
            df: pandas Datafrmae for replace default values
            column_name: df's column name for replace values
            min_val: minimal value for replace
            max_val: maximal value for replace

        Returns: df with replaced values

        """
        df[column_name] = np.where(df[column_name] >= max_val,
                                   f">={max_val}",
                                   np.where(df[column_name] <= min_val,
                                             f"<={min_val}",
                                             df[column_name]))
        return df

    def _calc_beta(self, 
                   guests_with_splits, 
                   use_buckets: bool = False):
        """Calculates II type error

        Args:
            guests_with_splits: dataframe with precalculated splits

        Returns: pandas DataFrame with II type error

        """
        beta_scores = pd.DataFrame()
        res_agg = pd.DataFrame()
        for metric_name in self.experiment_params.metrics_names:
            # index of max inject in self.experiment_params.injects
            max_found_inject_ind = 0
            for group_size in self.group_sizes:
                found_min_inject_flg = False
                found_max_inject_flg = False
                for inject in sorted(self.experiment_params.injects[max_found_inject_ind:], reverse=True):
                    if(found_min_inject_flg and found_max_inject_flg):
                        continue
                    else:
                        for split_number in range(1,self.experiment_params.iterations_number + 1):
                            # experiment
                            experiment_params = PrepilotBetaExperiment(group_sizes=group_size,
                                                                       split_number=split_number,
                                                                       metric_name=metric_name,
                                                                       inject=inject)
                            split_column = f"is_control_{experiment_params.control_group_size}_{experiment_params.target_group_size}_{experiment_params.split_number}"
                            one_split_guests = (guests_with_splits.loc[guests_with_splits[split_column]
                                                                    .isin([0,1])]
                            )
                            experiment_res = self._calc_experiment_grid_cell(one_split_guests, 
                                                                             experiment_params,
                                                                             use_buckets)
                            beta_scores = beta_scores.append(experiment_res)

                        calculated_experiments = ((beta_scores["split_rate"] == group_size) &
                                                       (beta_scores["metric"] == metric_name) &
                                                       (beta_scores["MDE"] == inject))
                        res_inject_agg = self._beta_score_calculation(beta_scores[calculated_experiments])
                        res_agg = res_agg.append(res_inject_agg)
                        # check if beta score higher then min_beta
                        if(res_inject_agg["beta"].values >= round(self.experiment_params.min_beta_score,
                                                           self._number_of_decimals)) and not found_min_inject_flg:
                            max_found_inject_ind = self.experiment_params.injects.index(inject)
                            found_min_inject_flg = True

                        if(res_inject_agg["beta"].values >= round(self.experiment_params.max_beta_score,
                                                           self._number_of_decimals)) and not found_max_inject_flg:
                            found_max_inject_flg = True

        res_agg.drop(columns=["sum", "count"], inplace=True)
        res_agg = res_agg.reset_index()
        res_agg = self._fill_res_with_default(res_agg,
                                              "beta",
                                              self.experiment_params.min_beta_score,
                                              self.experiment_params.max_beta_score)
        # append passed experiments
        res_agg = self._fill_passed_experiments(res_agg)
        res_pivoted = pd.pivot_table(res_agg,
                                     values="beta",
                                     index=["metric", "MDE"],
                                     columns="split_rate",
                                     aggfunc=lambda x: x)
        res_pivoted.replace(0, f"<={self.experiment_params.min_beta_score}", inplace=True)
        return res_pivoted

    @staticmethod
    def _first_found_mde(df_column: pd.Series):
        """Calculate max possible MDE for group sizes.

        Args:
            df_column: df's column with II type error scores.

        Returns: max possible MDE for group sizes.

        """
        if not df_column[(df_column != 0) & (df_column != 1)]:
            return "Effect wasn't detected"
        else:
            return df_column[(df_column != 0) & (df_column != 1)].idxmax()[1]

    def calc_alpha(self, guests: pd.DataFrame,
                   use_buckets: bool = False,
                   is_splited: bool = False):
        """Calculates I type error

        Args:
            guests: dataframe with guests
            stratification_params: params for stratification
            is_splited: if False guests must contain splits for calculation.
            Otherwise splits will be compute for guests

        Returns: pandas DataFrame with I type error

        """
        if not is_splited:
            prepilot_guests_collector = PrepilotSplitBuilder(guests, self.experiment_params.metrics_names,
                                                             self.experiment_params.injects, self.group_sizes,
                                                             self.stratification_params,
                                                             self.experiment_params.iterations_number)
            guests = prepilot_guests_collector.multliple_split(guests)

        alpha_scores = pd.DataFrame()

        for metric_name in self.experiment_params.metrics_names:
            for group_size in self.group_sizes:
                for split_number in range(1,self.experiment_params.iterations_number + 1):
                    experiment_params = PrepilotAlphaExperiment(group_sizes=group_size,
                                                                split_number=split_number,
                                                                metric_name=metric_name)
                    split_column = f"is_control_{experiment_params.control_group_size}_{experiment_params.target_group_size}_{experiment_params.split_number}"
                    one_split_guests = (guests.loc[guests[split_column]
                                                            .isin([0,1])]
                    )
                    experiment = self._calc_experiment_grid_cell(one_split_guests, 
                                                                 experiment_params, 
                                                                 use_buckets)
                    alpha_scores = alpha_scores.append(experiment)

        res_agg = (alpha_scores.groupby(by=["metric", "split_rate"])
                   .agg(sum=("effect_significance", sum),
                        count=("effect_significance", "count")))
        res_agg["alpha"] = res_agg["sum"]/res_agg["count"]
        res_agg.drop(columns=["sum", "count"], inplace=True)
        res_agg = res_agg.reset_index()
        res_pivoted = (pd.pivot_table(res_agg,
                                      values="alpha",
                                      index=["metric"],
                                      columns="split_rate")
                       .style.background_gradient(cmap="YlGn",
                                                  vmin=0,
                                                  vmax=1.0))
        return res_pivoted

    def collect(self, use_buckets: bool = False) -> pd.DataFrame:
        """Calculates I and II types error using prepilot parameters.

        Args:
            stratification_params: params for stratification
            full: if True function will return full dataframe with results.
            Otherwise will be returned only max calculated MDE for each size.

        Returns: pandas DataFrames with aggregated results of experiment.

        """
        prepilot_split_builder = PrepilotSplitBuilder(self.guests,
                                                         self.experiment_params.metrics_names,
                                                         self.experiment_params.injects,
                                                         self.group_sizes,
                                                         self.stratification_params,
                                                         self.experiment_params.iterations_number)

        prepilot_guests = prepilot_split_builder.collect()

        beta = self._calc_beta(prepilot_guests, use_buckets)
        alpha = self.calc_alpha(prepilot_guests,
                                use_buckets,
                                True)
        #if not full:
        #    beta = (pd.DataFrame(beta.apply(self._first_found_mde, axis=0))
        #            .rename(columns={0: "MDE"})).T
        return beta, alpha
