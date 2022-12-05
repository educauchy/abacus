from typing import List
import copy
import itertools
import numpy as np
import pandas as pd
from auto_ab.stratification.params import SplitBuilderParams
from auto_ab.stratification.split_builder import StratificationSplitBuilder
from auto_ab.prepilot.experiment_structures import BaseSplitElement


class PrepilotSplitBuilder():
    """Columns with splits and injects will be added
    """
    def __init__(self,
                 guests: pd.DataFrame,
                 metrics_names: List[str],
                 injects: List[float],
                 group_sizes: List[int],
                 stratification_params: SplitBuilderParams,
                 iterations_number: int = 10):
        """Builds splits for experiments

        Args:
            guests: dataframe with data for calculations injects and splits
            metrics_names: list of metrics for which will be calculate injects columns
            injects: list of injects
            group_sizes: list of group sizes for split building
            stratification_params: stratification parameters
            iterations_number: number of columns that will be build for each group size
        """
        self.guests = guests
        self.metrics_names = metrics_names
        self.injects = injects
        self.iterations_number = iterations_number
        self.group_sizes = group_sizes
        self.stratification_params = copy.deepcopy(stratification_params)
        self.split_grid = self._build_splits_grid()
        self.split_builder = StratificationSplitBuilder(self.guests, self.stratification_params)

    def _build_splits_grid(self):
        return list(BaseSplitElement(el[0], el[1])
                    for el in itertools.product(self.group_sizes, np.arange(1, self.iterations_number+1)))

    def _update_strat_params(self):
        """Update stratification columns, because of columns names duplicated problem
        """
        self.stratification_params.cols = [el + "_strat"
                                           if el not in [self.stratification_params.main_strata_col, 
                                                         self.stratification_params.split_metric_col
                                                        ]
                                           else el
                                           for el in self.stratification_params.cols]
        self.stratification_params.cat_cols = [el + "_strat" for el in self.stratification_params.cat_cols]

    def calc_injected_merics(self, guests_for_injects: pd.DataFrame) -> pd.DataFrame:
        """Calculates injected metrics for guests df

        Args:
            guests_for_injects: dataframe with metrics columns

        Returns: dataframe with injected metrics columns
        """
        matched = list(itertools.product(self.metrics_names, self.injects))
        guests_for_injects_copy = guests_for_injects.copy()
        for pair in matched:
            guests_for_injects_copy[f"{pair[0]}_{pair[1]}"] = guests_for_injects_copy[f"{pair[0]}"] * pair[1]
        return guests_for_injects_copy

    def _build_split(self,
                     guests_with_strata: pd.DataFrame,
                     control_group_size: int,
                     target_group_size: int,
                     split_number: int = 1):
        """Calculate one split with stratification

        Args:
            guests_with_strata: Dataframe fwith stratas
            control_group_size: control group size
            target_group_size: target group size
            split_number: number of split. Uses as suffix for new column

        Returns: pandas DataFrame with split
        """
        map_group_names_to_sizes={
            "control": control_group_size,
            "target": target_group_size
        }

        self.stratification_params.map_group_names_to_sizes = map_group_names_to_sizes
        guests_groups = self.split_builder._build_split(guests_with_strata)
        guests_groups = guests_groups.join(
                        pd.get_dummies(guests_groups["group_name"])
                        .add_prefix("is_")
                        .add_suffix(f"_{control_group_size}_{target_group_size}_{split_number}")
        )
        return guests_groups[[self.stratification_params.id_col
                              ,f"is_control_{control_group_size}_{target_group_size}_{split_number}"]]

    def collect(self) -> pd.DataFrame:
        """Calculate multiple split with stratification

        Returns: pandas DataFrame with split columns
        """
        guests_data_with_strata = self.split_builder._assign_strata(self.split_builder.split_data)
        experiment_guests = self.guests.loc[:, [self.stratification_params.id_col]]
        for split in self.split_grid:
            experiment_column = f"is_control_{split.control_group_size}_{split.target_group_size}_{split.split_number}"
            guests_split = self._build_split(guests_data_with_strata,
                                             split.control_group_size,
                                             split.target_group_size,
                                             split.split_number)
            experiment_guests = (experiment_guests
                                 .merge(guests_split[[self.stratification_params.id_col
                                                      , experiment_column]],
                                        on=self.stratification_params.id_col,
                                        how="left"))
            del guests_split
        guests_data_with_strata = (guests_data_with_strata
                                   .merge(experiment_guests,
                                          on=self.stratification_params.id_col,
                                          how="inner"))
        del experiment_guests
        return guests_data_with_strata

    #def collect(self) -> pd.DataFrame:
    #    """Builds dataframe with data for prepilot experiments
#
#        Returns: pandas DataFrame with columns for splits and injected metrics
#        """
#        prepilot_df = self.multliple_split()
#        return prepilot_df
