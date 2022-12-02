import sys
import logging
from typing import Dict, List
from math import floor
import pandas as pd
import numpy as np
import hdbscan
from sklearn.preprocessing import robust_scale
from auto_ab.stratification.params import SplitBuilderParams
from fastcore.transform import Pipeline
from auto_ab.stratification.stat_test import StatTest
#from auto_ab.stratification.binning import binnarize
from auto_ab.stratification.params import SplitBuilderParams
from auto_ab.auto_ab.abtest import ABTest
from auto_ab.auto_ab.params import ABTestParams
from auto_ab.auto_ab.params import DataParams, SimulationParams, HypothesisParams, ResultParams, SplitterParams
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


    def _prepare_cat_data(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df_cat = df.copy()
        #log.info("Prepare categorical cols")
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
            df_cat[col] = (df_cat[col]
                          .map(lambda x, counts=counts: counts[x] / self.guests_data.shape[0])
            )
        
        #self.guests_data = df.reset_index(drop=True)
        return df_cat

    def binnarize(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Calculate stratas for guest table")
        lst = []
        for region in list(df[self.params.region_col].unique()):
            dfr = df[df[self.params.region_col] == region]

            # check size of selection by region to skip unreasonable split
            if len(dfr) >= self.params.bin_min_size * self.params.n_bins_rto:
                # construct rto bins
                if len(dfr[self.params.split_metric_col].unique()) > self.params.n_bins_rto:
                    labels = pd.qcut(
                        dfr[self.params.split_metric_col], self.params.n_bins_rto, labels=False, duplicates="drop"
                    ).astype(str)
                else:
                    labels = dfr[self.params.split_metric_col].astype(int).astype(str)

                # construct extra columns bins
                for label in list(labels.unique()):
                    res = self.bin_with_clustering(dfr[labels == label], region, label, self.params)
                    lst.append(res)
            else:
                res = (dfr[[self.params.region_col, self.params.split_metric_col]]
                    .rename(columns={self.params.split_metric_col: f"{self.params.split_metric_col}_bin"})
                )
                res["cls"] = -1
                res["label"] = "outlier"
                res = res.astype(str)
                lst.append(res)

        stratas = pd.concat(lst, axis=0)
        stratas_wo_outliers = stratas.query("cls != '-1'")
        n_outliers = stratas.shape[0] - stratas_wo_outliers.shape[0]
        if n_outliers > 0:
            log.info(f"{n_outliers} outliers found")
        
        return df.loc[stratas["label"].index].assign(strata=stratas["label"])

    @staticmethod
    def bin_with_clustering(
        df_region_labeled: pd.DataFrame, region: str, label: str, params: SplitBuilderParams
        ) -> pd.DataFrame:

        try:
            X = df_region_labeled[params.cols].values  # pylint: disable=invalid-name
            X_scaled = robust_scale(X)  # pylint: disable=invalid-name
            clusterer = hdbscan.HDBSCAN(min_cluster_size=params.bin_min_size)
            clusterer.fit(X_scaled)
            inlabels = clusterer.labels_.astype(str)
        except ValueError:
            inlabels = ["0"]

        res = pd.DataFrame(
            {
                params.region_col: region,
                f"{params.split_metric_col}_bin": label,
                "cls": inlabels
            }, index=df_region_labeled.index
        )
        res = res.assign(label=lambda x: x[params.region_col].astype(str) + x[f"{params.split_metric_col}_bin"] + x.cls)
        return res


    def assign_strata(self) -> pd.DataFrame:
        """Assigns strata for rows

        Returns:
            DataFrame with strata columns
        """
        #self._prepare_cat_data()
        transform = [self._prepare_cat_data,self.binnarize]
        pipeline = Pipeline(transform)
        stratified_data = pipeline(self.guests_data)
        #return pipeline(self.df)
        #cat_prepared_df = self._prepare_cat_data()
        #log.info("Calculate stratas for guest table")
        #strata = binnarize(cat_prepared_df, self.params)
        #stratified_data = self.guests_data.loc[strata.index].assign(strata=strata)
        return stratified_data


    def _map_stratified_samples(self, guests:pd.DataFrame) -> pd.DataFrame:
        if all(x is None for x in self.params.map_group_names_to_sizes.values()):
            (self.params.map_group_names_to_sizes
                .update((key,len(guests)//len(self.params.map_group_names_to_sizes)) 
                        for key in self.params.map_group_names_to_sizes
                        )
            )
    
        group_guests_map = pd.DataFrame(columns=[self.params.customer_col, "group_name"])#dict()
        for group_name, group_size in self.params.map_group_names_to_sizes.items():
            available_guests = (guests.loc[~guests[self.params.customer_col]
                                .isin(group_guests_map[self.params.customer_col].values)]
                                .copy()
            )
            group_frac_to_take = min(group_size / len(available_guests), 1)

            group_guests = (
                available_guests
                .groupby("strata", group_keys=False)
                .apply(lambda x, frac=group_frac_to_take: x.sample(frac=frac))[self.params.customer_col]
                .to_frame().reset_index(drop=True)
            )
            group_guests["group_name"] = group_name
            group_guests_map = pd.concat([group_guests_map, group_guests])

        guests = guests.merge(group_guests_map, 
                              on = self.params.customer_col,
                              how = "left"
        )
        return guests
    
    def _check_groups(self, 
                    df_with_groups:pd.DataFrame,
                    control_name:str, 
                    target_groups_names: List[str]):
        tests_results = {}
        check_flag = 1

        for group in target_groups_names:
            #удалить после внесение изменений в парметры аб теста
            map_dict = {control_name:"A",group:"B"}
            guests_for_test = df_with_groups.copy()
            guests_for_test["group_name"] = guests_for_test["group_name"].map(map_dict)
            result_params = ResultParams()
            splitter_params = SplitterParams()
            simulation_params = SimulationParams()
            ####

            hypothesis_params = HypothesisParams(alpha=self.params.pvalue)
            for column in self.params.cols + self.params.cat_cols :
                data_params = DataParams(group_col="group_name",
                                    id_col = self.params.customer_col,
                                    target = column
                )
                ab_params = ABTestParams(data_params, simulation_params, hypothesis_params, result_params, splitter_params)
                ab_test = ABTest(guests_for_test, ab_params)

                if column in self.params.cols: 
                    test_result = ab_test.test_hypothesis_ttest()
                else:
                    test_result = ab_test.test_hypothesis_ztest_prop()

                tests_results[column] = test_result['p-value'].round(4)

            result = pd.DataFrame(tests_results, index=["1"])
            if(result < hypothesis_params.alpha).any().any():
                check_flag = 0
                log.error(f"Could not split statistically {group} and control")
                return check_flag
        return check_flag


    def build_split(self, guests_data_with_strata: pd.DataFrame) -> pd.DataFrame:
        """Builds strarified split

        Args:
            guests_data_with_strata: DataFrame with strata column

        Returns:
            DataFrame with split
        """
        max_attempts = 50  # max times to find split
        for _ in range(max_attempts):
            group_guests_map = self._map_stratified_samples(guests_data_with_strata)
            target_groups = group_guests_map["group_name"].unique().tolist()
            target_groups.remove(SplitBuilderParams.control_group_name)
            check_flag = self._check_groups(group_guests_map, SplitBuilderParams.control_group_name, target_groups)

            if check_flag:
                log.info("Success!")

                return group_guests_map

        log.error("Split failed!")
        return guests_data_with_strata
    

    def build_target_control_groups(self) -> pd.DataFrame:
        if len(self.guests_data) == 0:
            log.error("Empty guests_data")
            return self.guests_data
        # calculate stratas
        guests_data_with_strata = self.assign_strata()
        return self.build_split(guests_data_with_strata)
