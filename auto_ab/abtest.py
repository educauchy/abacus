import copy
import warnings

import numpy as np
import pandas as pd
import os
import sys
import yaml
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, mode, t
from typing import Dict, Union, Optional, Callable, Tuple, List, Any
from tqdm.auto import tqdm
from splitter import Splitter
from graphics import Graphics
from variance_reduction import VarianceReduction

sys.path.append('..')
from analysis.ab_params import *


metric_name_typing = Union[str, Callable[[np.array], Union[int, float]]]
stat_test_typing = Dict[ str, Union[int, float] ]

class ABTest:
    """Perform AB-test"""
    def __init__(self,
                 dataset: pd.DataFrame,
                 params: ABTestParams = ABTestParams(),
                 splitter: Optional[Callable[[np.array], Any]] = None,
                 metric: Optional[Callable[[np.array], Union[int, float]]] = None
                 ) -> None:
        self.dataset = dataset
        self.params = params

        if len(self.params.data_params.control) == 0:
            self.params.data_params.control   = self.__get_group('A')
            self.params.data_params.treatment = self.__get_group('B')

        if self.params.splitter_params.name == 'custom':
            self.splitter = splitter

        if self.params.metric_params.name == 'custom':
            self.metric = metric
        else:
            self.metric = self.__metric_calc(self.params.metric_params.name)
        # TODO: add validation for column presence in dataset

    def __str__(self):
        return f"ABTest(alpha={self.params.hypothesis_params.alpha}, " \
               f"beta={self.params.hypothesis_params.beta}, " \
               f"alternative='{self.params.hypothesis_params.alternative}')"

    def __get_group(self, group_label: str = 'A', df: Optional[pd.DataFrame] = None):
        X = df if df is not None else self.dataset

        group = X.loc[X[self.params.data_params.group_col] == group_label, \
                        self.params.data_params.target].tolist()
        return group

    def __metric_calc(self, name: str):
        if name == 'mean':
            return np.mean
        elif name == 'median':
            return np.median
        else: # default
            return np.mean

    def _manual_ttest(self, A_mean: float, A_var: float, A_size: int, B_mean: float, B_var: float, B_size: int) -> int:
        t_stat_empirical = (A_mean - B_mean) / (A_var / A_size + B_var / B_size) ** (1/2)
        df = A_size + B_size - 2

        test_result: int = 0
        if self.params.hypothesis_params.alternative == 'two-sided':
            lcv, rcv = t.ppf(self.params.hypothesis_params.alpha / 2, df), t.ppf(1.0 - self.params.hypothesis_params.alpha / 2, df)
            if not (lcv < t_stat_empirical < rcv):
                test_result = 1
        elif self.params.hypothesis_params.alternative == 'left':
            lcv = t.ppf(self.params.hypothesis_params.alpha, df)
            if t_stat_empirical < lcv:
                test_result = 1
        elif self.params.hypothesis_params.alternative == 'right':
            rcv = t.ppf(1 - self.params.hypothesis_params.alpha, df)
            if t_stat_empirical > rcv:
                test_result = 1

        return test_result

    def _linearize(self):
            X = self.dataset.loc[self.dataset[self.params.data_params.group_col] == 'A']
            K = round(sum(X[self.params.data_params.numerator]) / sum(X[self.params.data_params.denominator]), 4)

            self.dataset.loc[:, f"{self.params.data_params.numerator}_{self.params.data_params.denominator}"] = \
                        self.dataset[self.params.data_params.numerator] - K * self.dataset[self.params.data_params.denominator]
            self.target = f"{self.params.data_params.numerator}_{self.params.data_params.denominator}"

    def _delta_params(self, X: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculated expectation and variance for ratio metric using delta approximation
        :param X: Pandas DataFrame of particular group (A, B, etc)
        :return: Tuple with mean and variance of ratio
        """
        num = X[self.params.data_params.numerator]
        den = X[self.params.data_params.denominator]
        num_mean, den_mean = num.mean(), den.mean()
        num_var, den_var = num.var(), den.var()
        cov = X[[self.params.data_params.numerator, self.params.data_params.denominator]].cov().iloc[0, 1]
        n = len(num)

        bias_correction = (den_mean / num_mean ** 3) * (num_var / n) - cov / (n * num_mean ** 2)
        mean = den_mean / num_mean - 1 + bias_correction
        var = den_var / num_mean ** 2 - 2 * (den_mean / num_mean ** 3) * cov + (den_mean ** 2 / num_mean ** 4) * num_var

        return (mean, var)

    def _taylor_params(self, X: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculated expectation and variance for ratio metric using Taylor expansion approximation
        :param X: Pandas DataFrame of particular group (A, B, etc)
        :return: Tuple with mean and variance of ratio
        """
        num = X[self.params.data_params.numerator]
        den = X[self.params.data_params.denominator]
        mean = num.mean() / den.mean() - X[[self.params.data_params.numerator, self.params.data_params.denominator]].cov()[0, 1] \
               / (den.mean() ** 2) + den.var() * num.mean() / (den.mean() ** 3)
        var = (num.mean() ** 2) / (den.mean() ** 2) * (num.var() / (num.mean() ** 2) - \
                2 * X[[self.params.data_params.numerator, self.params.data_params.denominator]].cov()[0, 1]) \
                / (num.mean() * den.mean() + den.var() / (den.mean() ** 2))

        return (mean, var)

    def ratio_bootstrap(self, X: pd.DataFrame = None, Y: pd.DataFrame = None) -> stat_test_typing:
        if X is None and Y is None:
            X = self.dataset[self.dataset[self.params.data_params.group_col] == 'A']
            Y = self.dataset[self.dataset[self.params.data_params.group_col] == 'B']

        a_metric_total = sum(X[self.params.data_params.numerator]) / sum(X[self.params.data_params.denominator])
        b_metric_total = sum(Y[self.params.data_params.numerator]) / sum(Y[self.params.data_params.denominator])
        origin_mean = b_metric_total - a_metric_total
        boot_diffs = []
        boot_a_metric = []
        boot_b_metric = []

        for _ in tqdm(range(self.params.hypothesis_params.n_boot_samples)):
            a_boot = X[X[self.params.data_params.id_col].isin(X[self.params.data_params.id_col].sample(X[self.params.data_params.id_col].nunique(), replace=True))]
            b_boot = Y[Y[self.params.data_params.id_col].isin(Y[self.params.data_params.id_col].sample(Y[self.params.data_params.id_col].nunique(), replace=True))]
            a_boot_metric = sum(a_boot[self.params.data_params.numerator]) / sum(a_boot[self.params.data_params.denominator])
            b_boot_metric = sum(b_boot[self.params.data_params.numerator]) / sum(b_boot[self.params.data_params.denominator])
            boot_a_metric.append(a_boot_metric)
            boot_b_metric.append(b_boot_metric)
            boot_diffs.append(b_boot_metric - a_boot_metric)

        # correction
        boot_mean = np.mean(boot_diffs)
        delta = abs(origin_mean - boot_mean)
        boot_diffs = [boot_diff + delta for boot_diff in boot_diffs]
        delta_a = abs(a_metric_total - np.mean(boot_a_metric))
        delta_b = abs(b_metric_total - np.mean(boot_b_metric))
        boot_a_metric = [boot_a_diff + delta_a for boot_a_diff in boot_a_metric]
        boot_b_metric = [boot_b_diff + delta_b for boot_b_diff in boot_b_metric]

        pd_metric_diffs = pd.DataFrame(boot_diffs)

        left_quant  = self.params.hypothesis_params.alpha / 2
        right_quant = 1 - self.params.hypothesis_params.alpha / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        test_result: int = 0 # 0 - cannot reject H0, 1 - reject H0
        if ci_left > 0 or ci_right < 0: # left border of ci > 0 or right border of ci < 0
            test_result = 1

        result = {
            'stat': None,
            'p-value': None,
            'result': test_result
        }
        return result

    def ratio_taylor(self) -> stat_test_typing:
        """
        Calculate expectation and variance of ratio for each group
        and then use t-test for hypothesis testing
        Source: http://www.stat.cmu.edu/~hseltman/files/ratio.pdf
        :return: Hypothesis test result: 0 - cannot reject H0, 1 - reject H0
        """
        X = self.dataset[self.dataset[self.params.data_params.group_col] == 'A']
        Y = self.dataset[self.dataset[self.params.data_params.group_col] == 'B']

        A_mean, A_var = self._taylor_params(X)
        B_mean, B_var = self._taylor_params(Y)
        test_result: int = self._manual_ttest(A_mean, A_var, X.shape[0], B_mean, B_var, Y.shape[0])

        result = {
            'stat': None,
            'p-value': None,
            'result': test_result
        }
        return result

    def delta_method(self) -> stat_test_typing:
        """
        Delta method with bias correction for ratios
        Source: https://arxiv.org/pdf/1803.06336.pdf
        :return: Hypothesis test result: 0 - cannot reject H0, 1 - reject H0
        """
        X = self.dataset[self.dataset[self.params.data_params.group_col] == 'A']
        Y = self.dataset[self.dataset[self.params.data_params.group_col] == 'B']

        A_mean, A_var = self._delta_params(X)
        B_mean, B_var = self._delta_params(Y)
        test_result: int = self._manual_ttest(A_mean, A_var, X.shape[0], B_mean, B_var, Y.shape[0])

        result = {
            'stat': None,
            'p-value': None,
            'result': test_result
        }
        return result

    def linearization(self) -> None:
        """
        Important: there is an assumption that all data is already grouped by user
        s.t. numerator for user = sum of numerators for user for different time periods
        and denominator for user = sum of denominators for user for different time periods
        Source: https://research.yandex.com/publications/148
        :return: None
        """
        if not self.params.data_params.is_grouped:
            not_ratio_columns = self.dataset.columns[~self.dataset.columns.isin([self.params.data_params.numerator,
                                                                                 self.params.data_params.denominator])].tolist()
            df_grouped = self.dataset.groupby(by=not_ratio_columns, as_index=False).agg({
                self.params.data_params.numerator: 'sum',
                self.params.data_params.denominator: 'sum'
            })
            self.dataset = df_grouped
        self._linearize()

    def test_hypothesis(self) -> stat_test_typing:
        """
        Perform Welch's t-test / Mann-Whitney test for means/medians
        :return: Tuple: (test result: 0 - cannot reject H0, 1 - reject H0,
                        statistics,
                        p-value)
        """
        X = self.params.data_params.control
        Y = self.params.data_params.treatment

        test_result: int = 0
        pvalue: float = 1.0
        stat: float = 0.0
        if self.params.metric_params.name == 'mean':
            normality_passed = shapiro(X)[1] >= self.params.hypothesis_params.alpha \
                               and shapiro(Y)[1] >= self.params.hypothesis_params.alpha
            if not normality_passed:
                warnings.warn('One or both distributions are not normally distributed')
            stat, pvalue = ttest_ind(X, Y, equal_var=False, alternative=self.params.hypothesis_params.alternative)
        elif self.params.metric_params.name == 'median':
            stat, pvalue = mannwhitneyu(X, Y, alternative=self.params.hypothesis_params.alternative)

        if pvalue <= self.params.hypothesis_params.alpha:
            test_result = 1

        result = {
            'stat': stat,
            'p-value': pvalue,
            'result': test_result
        }
        return result

    def test_hypothesis_buckets(self) -> stat_test_typing:
        """
        Perform buckets hypothesis testing
        :return: Test result: 1 - significant different, 0 - insignificant difference
        """
        X = self.params.data_params.control
        Y = self.params.data_params.treatment

        np.random.shuffle(X)
        np.random.shuffle(Y)
        X_new = np.array([ self.metric(x) for x in np.array_split(X, self.params.hypothesis_params.n_buckets) ])
        Y_new = np.array([ self.metric(y) for y in np.array_split(Y, self.params.hypothesis_params.n_buckets) ])

        test_result: int = 0
        if shapiro(X_new)[1] >= self.params.hypothesis_params.alpha and shapiro(Y_new)[1] >= self.params.hypothesis_params.alpha:
            _, pvalue = ttest_ind(X_new, Y_new, equal_var=False, alternative=self.params.hypothesis_params.alternative)
            if pvalue <= self.params.hypothesis_params.alpha:
                test_result = 1
        else:
            def metric(X: np.array):
                modes, _ = mode(X)
                return sum(modes) / len(modes)
            stat, pvalue, test_result = self.test_hypothesis_boot_confint()

        result = {
            'stat': None,
            'p-value': None,
            'result': test_result
        }
        return result

    def test_hypothesis_strat_confint(self, strata_col: str = '',
                                    weights: Dict[str, float] = None) -> stat_test_typing:
        """
        Perform stratification with confidence interval
        :return: Test result: 1 - significant different, 0 - insignificant difference
        """
        metric_diffs: List[float] = []
        X = self.dataset.loc[self.dataset[self.params.data_params.group_col] == 'A']
        Y = self.dataset.loc[self.dataset[self.params.data_params.group_col] == 'B']
        for _ in tqdm(range(self.params.hypothesis_params.n_boot_samples)):
            x_strata_metric = 0
            y_strata_metric = 0
            for strat in weights.keys():
                X_strata = X.loc[X[strata_col] == strat, self.params.data_params.target]
                Y_strata = Y.loc[Y[strata_col] == strat, self.params.data_params.target]
                x_strata_metric += (self.metric(np.random.choice(X_strata, size=X_strata.shape[0] // 2, replace=False)) * weights[strat])
                y_strata_metric += (self.metric(np.random.choice(Y_strata, size=Y_strata.shape[0] // 2, replace=False)) * weights[strat])
            metric_diffs.append(self.metric(x_strata_metric) - self.metric(y_strata_metric))
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        left_quant = self.params.hypothesis_params.alpha / 2
        right_quant = 1 - self.params.hypothesis_params.alpha / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        test_result: int = 0 # 0 - cannot reject H0, 1 - reject H0
        if ci_left > 0 or ci_right < 0: # left border of ci > 0 or right border of ci < 0
            test_result = 1

        result = {
            'stat': None,
            'p-value': None,
            'result': test_result
        }
        return result

    def test_hypothesis_boot_est(self) -> stat_test_typing:
        """
        Perform bootstrap confidence interval with
        :returns: Type I error rate
        """
        X = self.params.data_params.control
        Y = self.params.data_params.treatment

        metric_diffs: List[float] = []
        for _ in tqdm(range(self.params.hypothesis_params.n_boot_samples)):
            x_boot = np.random.choice(X, size=X.shape[0], replace=True)
            y_boot = np.random.choice(Y, size=Y.shape[0], replace=True)
            metric_diffs.append( self.metric(x_boot) - self.metric(y_boot) )
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        left_quant = self.params.hypothesis_params.alpha / 2
        right_quant = 1 - self.params.hypothesis_params.alpha / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        criticals = [0, 0]
        for boot in pd_metric_diffs:
            if boot < 0 and boot < ci_left:
                criticals[0] += 1
            elif boot > 0 and boot > ci_right:
                criticals[1] += 1
        false_positive = min(criticals) / pd_metric_diffs.shape[0]

        test_result: int = 0 # 0 - cannot reject H0, 1 - reject H0
        if false_positive <= self.params.hypothesis_params.alpha:
            test_result = 1

        result = {
            'stat': None,
            'p-value': false_positive,
            'result': test_result
        }
        return result

    def test_hypothesis_boot_confint(self) -> stat_test_typing:
        """
        Perform bootstrap confidence interval
        :returns: Ratio of rejected H0 hypotheses to number of all tests
        """
        X = self.params.data_params.control
        Y = self.params.data_params.treatment

        metric_diffs: List[float] = []
        for _ in tqdm(range(self.params.hypothesis_params.n_boot_samples)):
            x_boot = np.random.choice(X, size=X.shape[0], replace=True)
            y_boot = np.random.choice(Y, size=Y.shape[0], replace=True)
            metric_diffs.append( self.metric(x_boot) - self.metric(y_boot) )
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        left_quant = self.params.hypothesis_params.alpha / 2
        right_quant = 1 - self.params.hypothesis_params.alpha / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        test_result: int = 0 # 0 - cannot reject H0, 1 - reject H0
        if ci_left > 0 or ci_right < 0: # left border of ci > 0 or right border of ci < 0
            test_result = 1

        result = {
            'stat': None,
            'p-value': None,
            'result': test_result
        }
        return result

    def test_boot_hypothesis(self) -> float:
        """
        Perform T-test for independent samples with unequal number of observations and variance
        :returns: Ratio of rejected H0 hypotheses to number of all tests
        """
        X = self.params.data_params.control
        Y = self.params.data_params.treatment

        T: int = 0
        for _ in range(self.params.hypothesis_params.n_boot_samples):
            x_boot = np.random.choice(X, size=X.shape[0], replace=True)
            y_boot = np.random.choice(Y, size=Y.shape[0], replace=True)

            T_boot = (np.mean(x_boot) - np.mean(y_boot)) / (np.var(x_boot) / x_boot.shape[0] + np.var(y_boot) / y_boot.shape[0])
            test_res = ttest_ind(x_boot, y_boot, equal_var=False, alternative=self.params.hypothesis_params.alternative)

            if T_boot >= test_res[1]:
                T += 1

        pvalue = T / self.params.hypothesis_params.n_boot_samples

        return pvalue

    def cuped(self):
        vr = VarianceReduction()
        result_df = vr.cuped(self.dataset,
                            target=self.params.data_params.target,
                            groups=self.params.data_params.group_col,
                            covariate=self.params.data_params.covariate)

        self.params_new = copy.deepcopy(self.params)
        self.params_new.data_params.control = self.__get_group('A', result_df)
        self.params_new.data_params.treatment = self.__get_group('B', result_df)

        return ABTest(self.dataset, self.params_new)

    def cupac(self):
        vr = VarianceReduction()
        result_df = vr.cupac(self.dataset,
                               target_prev=self.params.data_params.target_prev,
                               target_now=self.params.data_params.target,
                               factors_prev=self.params.data_params.predictors_prev,
                               factors_now=self.params.data_params.predictors,
                               groups=self.params.data_params.group_col)

        self.params_new = copy.deepcopy(self.params)
        self.params_new.data_params.control = self.__get_group('A', result_df)
        self.params_new.data_params.treatment = self.__get_group('B', result_df)

        return ABTest(self.dataset, self.params_new)

    def __bucketize(self, X: List[float]):
        np.random.shuffle(X)
        X_new = np.array([ self.metric(x) for x in np.array_split(X, self.params.hypothesis_params.n_buckets) ])
        return X_new.tolist()

    def bucketing(self):
        self.params_new = copy.deepcopy(self.params)
        self.params_new.data_params.control   = self.__bucketize(self.params.data_params.control)
        self.params_new.data_params.treatment = self.__bucketize(self.params.data_params.treatment)

        return ABTest(self.dataset, self.params_new)

    def plot(self) -> None:
        a = self.__get_group('A')
        b = self.__get_group('B')

        if self.params.metric_params.name == 'mean':
            Graphics().plot_mean_experiment(a, b,
                                            self.params.hypothesis_params.alternative,
                                            self.params.metric_params.name,
                                            self.params.hypothesis_params.alpha,
                                            self.params.hypothesis_params.beta)


if __name__ == '__main__':
    with open("../analysis/configs/auto_ab.config.yaml", "r") as stream:
        try:
            ab_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    metric_params = MetricParams(**ab_config['metric_params'])
    data_params = DataParams(**ab_config['data_params'])
    simulation_params = SimulationParams(**ab_config['simulation_params'])
    hypothesis_params = HypothesisParams(**ab_config['hypothesis_params'])
    result_params = ResultParams(**ab_config['result_params'])
    splitter_params = SplitterParams(**ab_config['splitter_params'])
    # bootstrap_params = BootstrapParams(**ab_config['bootstrap_params'])

    ab_params = ABTestParams(metric_params,
                             data_params,
                             simulation_params,
                             hypothesis_params,
                             result_params,
                             splitter_params)

    df = pd.read_csv('../notebooks/ab_data.csv')

    ab_obj = ABTest(df, ab_params)
    ab_obj = ab_obj.cuped().bucketing()
    print(ab_obj.params.data_params)
