import sys
import logging
import random
from typing import Dict, List
from math import floor
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from auto_ab.stratification.stat_test import StatTest
from auto_ab.stratification.binning import binnarize
from auto_ab.stratification.params import SplitBuilderParams
pd.options.mode.chained_assignment = None

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def build_target_control_groups(guests_data: pd.DataFrame, params: SplitBuilderParams) -> pd.DataFrame:
    if len(guests_data) == 0:
        log.error("Empty guests_data")
        return guests_data

    guests_data, params = prepare_cat_data(guests_data, params)
    # calculate stratas
    guests_data_with_strata = assign_strata(guests_data.reset_index(drop=True), params)
    return build_split(guests_data_with_strata,params)


def build_split(guests_data_with_strata: pd.DataFrame, params: SplitBuilderParams) -> pd.DataFrame:
    max_attempts = 50  # max times to find split
    for _ in range(max_attempts):
        is_all_sizes_none = all(x is None for x in params.map_group_names_to_sizes.values())
        is_any_size_none = any(x is None for x in params.map_group_names_to_sizes.values())
        if is_all_sizes_none:
            group_guests_map = form_equal_group_splits(
               guests_data_with_strata, params
            )
        elif is_any_size_none:
            raise ValueError("Sizes in map_group_names_to_sizes must be 'int' or 'None'.")
        else:
            group_guests_map = form_unequal_group_splits(
               guests_data_with_strata, params
            )

        control_group = group_guests_map[SplitBuilderParams.control_group_name]
        target_groups = [
            group_guests
            for group_name, group_guests in group_guests_map.items()
            if group_name != SplitBuilderParams.control_group_name
        ]
        check_flag = check_splits(guests_data_with_strata, control_group, target_groups, params)

        if check_flag:
            log.info("Success!")

            return add_group_name(guests_data_with_strata, group_guests_map, params)

    log.error("Split failed!")
    return guests_data_with_strata


def assign_strata(guest_info: pd.DataFrame, params: SplitBuilderParams) -> pd.DataFrame:
    log.info("Calculate stratas for guest table")
    strata = binnarize(guest_info, params)
    stratified_data = guest_info.loc[strata.index].assign(strata=strata)
    return stratified_data


def form_equal_group_splits(guests_data: pd.DataFrame, params: SplitBuilderParams) -> Dict[str, List[int]]:
    total_size = guests_data.shape[0]
    n_groups = len(params.map_group_names_to_sizes)
    size_one_group = int(floor(total_size / n_groups))

    if size_one_group == 0:
        log.error("Size one group equals 0. Check size of guests_data")
        raise ValueError("Impossible size of guests_data")

    # sample target groups
    group_guests_map = dict()
    s_kfolds = StratifiedKFold(n_splits=n_groups, shuffle=True)
    group_names = list(params.map_group_names_to_sizes.keys())
    for i, (_, test_index) in enumerate(s_kfolds.split(guests_data, guests_data["strata"])):
        group_guests_map[group_names[i]] = guests_data.loc[
            guests_data.index.intersection(test_index), params.customer_col
        ].tolist()

    return group_guests_map


def form_unequal_group_splits(guests_data: pd.DataFrame, params: SplitBuilderParams) -> Dict[str, List[int]]:
    already_taken_guests = []
    group_guests_map = dict()
    for group_name, group_size in params.map_group_names_to_sizes.items():
        new_df = guests_data.loc[~guests_data[params.customer_col].isin(already_taken_guests)].copy()
        group_frac_to_take = min(group_size / len(new_df), 1)

        group_guests = (
            new_df
            .groupby("strata", group_keys=False)
            .apply(lambda x, frac=group_frac_to_take: x.sample(frac=frac))
            [params.customer_col].tolist()
        )

        abs_percent_err = round(abs((len(group_guests) - group_size) / group_size) * 100, 2)
        log.info(f"{group_name}: Desired size = {group_frac_to_take}, \
        resulting size = {len(group_guests)}, diff = {abs_percent_err} %")

        group_guests_map[group_name] = group_guests
        already_taken_guests.extend(group_guests)

    return group_guests_map


def check_splits(guests_data: pd.DataFrame, control_group: List[int],
                 target_groups: List[List[int]], params: SplitBuilderParams) -> int:
    check_flag = 1
    for i, target_group in enumerate(target_groups):
        errors = compare_groups(
            guests_data,
            {SplitBuilderParams.control_group_name: control_group, "target": target_group},
            params
        )

        if (errors < params.pvalue).any().any():
            check_flag = 0
            log.error(f"Could not split statistically {i} target and control")
            return check_flag

    return check_flag


def compare_groups(df: pd.DataFrame, split_dict: Dict, params: SplitBuilderParams) -> pd.DataFrame:
    cols_validate = params.cols
    cat_cols = params.cat_cols
    customer_col = params.customer_col
    for i, col in enumerate(cols_validate):
        unique_num = len(df[col].unique())
        if (unique_num < SplitBuilderParams.min_unique_values_in_col) and (col not in cat_cols):
            log.warning(f"Unable to include '{col}' for test, the number of unique values is {unique_num}."
                        "It can be encoded as category col.")
            cols_validate.pop(i)
    stats = StatTest(df, split_dict, customer_col, cols_validate, cat_cols, params.stat_test).compute()
    return stats


def add_group_name(guests_data: pd.DataFrame, group_guests_map: Dict[str, List[int]]
                   ,params: SplitBuilderParams) -> pd.DataFrame:
    guests_data["group_name"] = None
    for group_name, group_guests in group_guests_map.items():
        guests_data.loc[
            guests_data[params.customer_col].isin(group_guests), "group_name"
        ] = group_name

    return guests_data.dropna(subset=["group_name"])


def prepare_cat_data(guests: pd.DataFrame, params: SplitBuilderParams) -> pd.DataFrame:
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

    Args:
        guests (pd.DataFrame): DataFrame with stratification data
        params (SplitBuilderParams): stratification parameters

    Return:
        pd.DataFrame: DataFrame with extra columns
    """
    df = guests.copy()
    n_row = df.shape[0]
    n_top_cat = params.n_top_cat

    for col in params.cat_cols:
        top_cat = df.groupby(col).size().sort_values(ascending=False).index[:n_top_cat]
        encoded_col = df[col].copy()
        encoded_col[~encoded_col.isin(top_cat)] = sys.maxsize

        counts = encoded_col.value_counts().to_dict()
        # the addition of a random value is used to separate groups with the same frequency:
        # {A: 83, B: 83} -> {A: 83.015, B: 83.036}
        counts = {key: value + (random.random() / 10) for key, value in counts.items()}

        df[f"{col}_encoded"] = encoded_col
        df[f"{col}"] = df[f"{col}_encoded"].map(lambda x, counts=counts: counts[x] / n_row)
        del df[f"{col}_encoded"]

    return df
