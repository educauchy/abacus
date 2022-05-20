from typing import Dict, List
import logging
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class StatTest:
    def __init__(
        self,
        df: pd.DataFrame,
        split_dict: Dict[str, List[int]],
        customer_col: str,
        cols_validate: List[str],
        cat_cols: List[str],
        stat_test: str
    ):
        self.df = df
        self.split_dict = split_dict
        self.customer_col = customer_col
        self.numeric_cols = [x for x in cols_validate if x not in cat_cols]
        self.categorical_cols = cat_cols
        self.stat_test = stat_test

    def compute(self) -> pd.DataFrame:
        """This function compares the distributions of values in groups.

        Return:
            pd.DataFrame: DataFrame with test results. Columns - tested features. Rows - test.

        """
        num_result = self.compare_continuous_variables()
        cat_result = self.compare_conversion()
        all_result = num_result.append(cat_result)
        return all_result

    def compare_continuous_variables(self) -> pd.DataFrame:
        pvalue = {}
        for col in self.numeric_cols:
            dct = {}
            for group_name in self.split_dict.keys():
                dct[group_name] = self._sampled_group_col(
                    self.df, self.split_dict[group_name], col
                )

            if self.stat_test == "ttest_ind":
                test_res = ttest_ind(dct["target"], dct["control"], equal_var=False)
            elif self.stat_test == "mannwhitneyu":
                test_res = mannwhitneyu(
                    dct["target"], dct["control"], alternative="two-sided"
                )
            else:
                log.error("Unknown test!")

            pvalue[col] = test_res[1].round(4)
        return pd.DataFrame(pvalue, index=[self.stat_test])

    def compare_conversion(self) -> pd.DataFrame:
        chi, dct = {}, {}
        for col in self.categorical_cols:
            for group_name in self.split_dict.keys():
                sampled_guests_df = self._sampled_group_col(
                    self.df, self.split_dict[group_name], col
                ).to_frame()
                sampled_guests_df["group_name"] = group_name
                dct[group_name] = sampled_guests_df

            data_for_test = dct["target"].append(dct["control"], ignore_index=True)
            crosstab_df = pd.crosstab(data_for_test["group_name"], data_for_test[col])
            chi[col] = chi2_contingency(crosstab_df, correction=False)[1]  # p-value
        return pd.DataFrame(chi, index=["chisq"])

    def _sampled_group_col(self, df, guests_list: List[int], col: str):
        sampled_guests_df = pd.DataFrame({self.customer_col : guests_list})
        sampled_guests_df = sampled_guests_df.merge(df, on=self.customer_col , how="inner")
        return sampled_guests_df[col]
