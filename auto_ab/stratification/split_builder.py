import sys
import logging
import random
from typing import Dict, List
from math import floor
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from auto_ab.stratification.stat_test import StatTest
from auto_ab.stratification.binning import binnarize
from auto_ab.stratification.params import SplitBuilderParams
pd.options.mode.chained_assignment = None

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class StratificationSplitBuilder:
    def __init__(self, 
                 guests_data: pd.DataFrame, 
                 params: SplitBuilderParams):
        """Builds stratification split for DataFrame

        Args:
            guests_data: dataframe with data building split
            params: params for stratification and spilt
        """
        self.guests_data = guests_data.reset_index(drop=True)
        self.params = params


    def _prepare_cat_data(self) -> pd.DataFrame:
        """This function converts given categorical features into features suitable for clustering and
        stratification. This functionality is achieved by adding two new features for each categorical
        feature:

        first feature (_encoded): there are two cases:
            (1) If the number of unique values of feature more than the value "n_top_cat" from config,
                then values with low frequency will be combined into one ("other" with code DEFAULT_CAT_VALUE).
                After encoding feature will contain (n_top_cat + 1) unique values.
            (2) If the number of unique values of feature less than the value "n_top_cat" from config,
                the new column will be the same as the original.

        second feature (_freq): frequency with noise of the encoded feature

        Return:
            pd.DataFrame: DataFrame with extra columns
        """
        df = self.guests_data.copy()
    
        for col in self.params.cat_cols:
            counts = df[col].value_counts()
            counts.iloc[:self.params.n_top_cat] = (
                counts.iloc[:self.params.n_top_cat] 
                + 0.1 * (np.random.uniform(low=0., 
                                            high=1., 
                                            size=len(counts.iloc[:self.params.n_top_cat])) 
                        )
            )
            counts.iloc[self.params.n_top_cat:] = sys.maxsize
            counts = counts.to_dict()
            df[col] = (df[col]
                      .map(lambda x, counts=counts: counts[x] / self.guests_data.shape[0])
            )
        
        self.guests_data = df.reset_index(drop=True)


    def assign_strata(self) -> pd.DataFrame:
        """Assigns strata for rows

        Returns:
            DataFrame with strata columns
        """
        self._prepare_cat_data()
        log.info("Calculate stratas for guest table")
        strata = binnarize(self.guests_data, self.params)
        stratified_data = self.guests_data.loc[strata.index].assign(strata=strata)
        return stratified_data


    def _form_equal_group_splits(self, guests:pd.DataFrame) -> Dict[str, List[int]]:
        total_size = guests.shape[0]
        n_groups = len(self.params.map_group_names_to_sizes)
        size_one_group = int(floor(total_size / n_groups))

        if size_one_group == 0:
            log.error("Size one group equals 0. Check size of guests_data")
            raise ValueError("Impossible size of guests_data")

        # sample target groups
        group_guests_map = dict()
        s_kfolds = StratifiedKFold(n_splits=n_groups, shuffle=True)
        group_names = list(self.params.map_group_names_to_sizes.keys())
        for i, (_, test_index) in enumerate(s_kfolds.split(guests, guests["strata"])):
            group_guests_map[group_names[i]] = guests.loc[
                guests.index.intersection(test_index), self.params.customer_col
            ].tolist()

        return group_guests_map


    def _form_unequal_group_splits(self, guests:pd.DataFrame) -> Dict[str, List[int]]:
        already_taken_guests = []
        group_guests_map = dict()
        for group_name, group_size in self.params.map_group_names_to_sizes.items():
            new_df = guests.loc[~guests[self.params.customer_col].isin(already_taken_guests)].copy()
            group_frac_to_take = min(group_size / len(new_df), 1)

            group_guests = (
                new_df
                .groupby("strata", group_keys=False)
                .apply(lambda x, frac=group_frac_to_take: x.sample(frac=frac))
                [self.params.customer_col].tolist()
            )

            abs_percent_err = round(abs((len(group_guests) - group_size) / group_size) * 100, 2)
            log.info(f"{group_name}: Desired size = {group_frac_to_take}, \
            resulting size = {len(group_guests)}, diff = {abs_percent_err} %")

            group_guests_map[group_name] = group_guests
            already_taken_guests.extend(group_guests)

        return group_guests_map


    def _compare_groups(self, df: pd.DataFrame, split_dict: Dict) -> pd.DataFrame:
        cols_validate = self.params.cols
        cat_cols = self.params.cat_cols
        customer_col = self.params.customer_col
        for i, col in enumerate(cols_validate):
            unique_num = len(df[col].unique())
            if (unique_num < SplitBuilderParams.min_unique_values_in_col) and (col not in cat_cols):
                log.warning(f"Unable to include '{col}' for test, the number of unique values is {unique_num}."
                            "It can be encoded as category col.")
                cols_validate.pop(i)
        stats = StatTest(df, split_dict, customer_col, cols_validate, cat_cols, self.params.stat_test).compute()
        return stats


    def _check_splits(self, guests_data: pd.DataFrame, control_group: List[int],
                    target_groups: List[List[int]]) -> int:
        check_flag = 1
        for i, target_group in enumerate(target_groups):
            errors = self._compare_groups(
                guests_data,
                {SplitBuilderParams.control_group_name: control_group, "target": target_group},
            )

            if (errors < self.params.pvalue).any().any():
                check_flag = 0
                log.error(f"Could not split statistically {i} target and control")
                return check_flag

        return check_flag


    def _add_group_name(self, guests_data: pd.DataFrame, group_guests_map: Dict[str, List[int]]) -> pd.DataFrame:
        guests_data["group_name"] = None
        for group_name, group_guests in group_guests_map.items():
            guests_data.loc[
                guests_data[self.params.customer_col].isin(group_guests), "group_name"
            ] = group_name

        return guests_data.dropna(subset=["group_name"])


    def build_split(self, guests_data_with_strata: pd.DataFrame) -> pd.DataFrame:
        """Builds strarified split

        Args:
            guests_data_with_strata: DataFrame with strata column

        Returns:
            DataFrame with split
        """
        max_attempts = 50  # max times to find split
        for _ in range(max_attempts):
            is_all_sizes_none = all(x is None for x in self.params.map_group_names_to_sizes.values())
            is_any_size_none = any(x is None for x in self.params.map_group_names_to_sizes.values())
            if is_all_sizes_none:
                group_guests_map = self._form_equal_group_splits(guests_data_with_strata)
            elif is_any_size_none:
                raise ValueError("Sizes in map_group_names_to_sizes must be 'int' or 'None'.")
            else:
                group_guests_map = self._form_unequal_group_splits(guests_data_with_strata)

            control_group = group_guests_map[SplitBuilderParams.control_group_name]
            target_groups = [
                group_guests
                for group_name, group_guests in group_guests_map.items()
                if group_name != SplitBuilderParams.control_group_name
            ]
            check_flag = self._check_splits(guests_data_with_strata, control_group, target_groups)

            if check_flag:
                log.info("Success!")

                return self._add_group_name(guests_data_with_strata, group_guests_map)

        log.error("Split failed!")
        return guests_data_with_strata
    

    def build_target_control_groups(self) -> pd.DataFrame:
        if len(self.guests_data) == 0:
            log.error("Empty guests_data")
            return self.guests_data
        # calculate stratas
        guests_data_with_strata = self.assign_strata()
        return self.build_split(guests_data_with_strata)
