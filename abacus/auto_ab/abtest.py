from typing import Dict, Union, Optional, Callable, Tuple, List
import copy
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, mode, t, chisquare, norm
from statsmodels.stats.proportion import proportions_ztest
from abacus.auto_ab.graphics import Graphics
from abacus.auto_ab.variance_reduction import VarianceReduction
from abacus.auto_ab.params import ABTestParams
from abacus.resplitter.resplit_builder import ResplitBuilder
from abacus.resplitter.params import ResplitParams


metric_name_typing = Union[str, Callable[[np.ndarray], Union[int, float]]]
stat_test_typing = Dict[str, Union[int, float]]


class ABTest:
    """Performs different calculations of A/B-test.

    - Results evaluation for different metric types (continuous, binary, ratio).
    - Bucketing (decrease number of points, normal distribution of metric of interest)

    Example:
        >>> x = pd.read_csv('data.csv')
        >>> ab_params = ABTestParams(...)
        >>> ab_test = ABTest(x, ab_params)
        >>> ab_test.test_welch()
        {'stat': 5.172, 'p-value': 0.312, 'result': 1}
    """
    def __init__(self,
                 dataset: pd.DataFrame,
                 params: ABTestParams
                 ) -> None:
        self.params = params
        self.__check_required_columns(dataset, 'init')
        self.__dataset = dataset
        self.params.data_params.control = self.__get_group(self.params.data_params.control_name, self.dataset)
        self.params.data_params.treatment = self.__get_group(self.params.data_params.treatment_name, self.dataset)

    @property
    def dataset(self):
        return self.__dataset

    def __str__(self):
        return f"ABTest(alpha={self.params.hypothesis_params.alpha}, " \
               f"beta={self.params.hypothesis_params.beta}, " \
               f"alternative='{self.params.hypothesis_params.alternative}')"

    def __check_required_columns(self, df: pd.DataFrame, method: str) -> None:
        """Check presence of columns in dataframe.

        Args:
            df (pandas.DataFrame): DataFrame to check.
            method (str): Stage of A/B process which you'd like to test.

        Raises:
            ValueError: If `is_valid_col` is False. Experiment cannot be provided
            if required columns are absent.
        """
        cols: List[str] = []
        if method == 'init':
            cols = ['id_col', 'group_col']
            if self.params.hypothesis_params.metric_type == 'solid':
                cols.append('target')
            elif self.params.hypothesis_params.metric_type == 'binary':
                cols.append('target_flg')
            elif self.params.hypothesis_params.metric_type == 'ratio':
                cols.extend(['numerator', 'denominator'])
        elif method == 'cuped':
            cols = ['covariate']
        elif method == 'cupac':
            cols = ['predictors_prev', 'target_prev', 'predictors']
        elif method == 'resplit_df':
            cols = ['strata_col']

        is_valid_col: bool = True
        invalid_cols = []
        for col in cols:
            curr_col = getattr(self.params.data_params, col)
            if isinstance(curr_col, str) and curr_col is not None:
                if curr_col not in df:
                    is_valid_col = False
                    invalid_cols.append(curr_col)

            elif isinstance(curr_col, list) and curr_col is not None:
                for curr_c in curr_col:
                    if curr_c not in df:
                        is_valid_col = False
                        invalid_cols.append(curr_c)

        if not is_valid_col:
            raise ValueError(f'The following columns are not in dataframe: {*invalid_cols, }')

    def __get_group(self, group_label: str, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Gets target metric column based on desired group label.

        Args:
            group_label (str): Group label, e.g. 'A', 'B'.
            df (pd.DataFrame, optional): DataFrame to query from.

        Returns:
            numpy.ndarray: Target column for a desired group.
        """
        x = df if df is not None else self.__dataset
        group = np.array([])
        if self.params.hypothesis_params.metric_type == 'solid':
            group = x.loc[x[self.params.data_params.group_col] == group_label,
                          self.params.data_params.target].to_numpy()
        elif self.params.hypothesis_params.metric_type == 'binary':
            group = x.loc[x[self.params.data_params.group_col] == group_label,
                          self.params.data_params.target_flg].to_numpy()
        return group

    def __bucketize(self, x: np.ndarray) -> np.ndarray:
        """Split array ``x`` into N non-overlapping buckets.

        There are two purposes for this actions:

        1. Decrease number of data points of experiment.
        2. Get normal distribution of a metric of interest.

        Procedure:

        1. Shuffle elements of an array.
        2. Split points into N non-overlapping buckets.
        3. On every bucket calculate metric of interest.

        Args:
            x (np.ndarray): Array to split.

        Returns:
            np.ndarray: Splitted into buckets array.
        """
        np.random.shuffle(x)
        x_new = np.array([self.params.hypothesis_params.metric(x_)
                    for x_ in np.array_split(x, self.params.hypothesis_params.n_buckets)])
        return x_new

    def _manual_ttest(self, ctrl_mean: float, ctrl_var: float, ctrl_size: int,
                      treat_mean: float, treat_var: float, treat_size: int) -> stat_test_typing:
        """Performs Welch's t-test based on aggregation of metrics instead of datasets.

        For empirical calculation of T-statistic we need: expectation, variance, array size for each group.

        Args:
            ctrl_mean (float): Mean of control group.
            ctrl_var (float): Variance of control group.
            ctrl_size (int): Size of control group.
            treat_mean (float): Mean of treatment group.
            treat_var (float): Variance of treatment group.
            treat_size (int): Size of treatment group.

        Returns:
            stat_test_typing: Dictionary with following properties: test statistic, p-value, test result. Test result: 1 - significant different, 0 - insignificant difference.
        """
        t_stat_empirical = (ctrl_mean - treat_mean) / (ctrl_var / ctrl_size + treat_var / treat_size) ** (1/2)
        df = ctrl_size + treat_size - 2

        test_result: int = 0
        if self.params.hypothesis_params.alternative == 'two-sided':
            lcv, rcv = t.ppf(self.params.hypothesis_params.alpha / 2, df), \
                       t.ppf(1.0 - self.params.hypothesis_params.alpha / 2, df)
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

        result = {
            'stat': None,
            'p-value': None,
            'result': test_result
        }
        return result

    def _delta_params(self, x: pd.DataFrame) -> Tuple[float, float]:
        """Calculated expectation and variance for ratio metric using delta approximation.

        Source: https://arxiv.org/pdf/1803.06336.pdf.

        Args:
            x (pandas.DataFrame): Pandas DataFrame of particular group (A, B, etc).

        Returns:
            Tuple[float, float]: Mean and variance of ratio metric.
        """
        num = x[self.params.data_params.numerator]
        den = x[self.params.data_params.denominator]
        num_mean, den_mean = num.mean(), den.mean()
        num_var, den_var = num.var(), den.var()
        cov = x[[self.params.data_params.numerator, self.params.data_params.denominator]].cov().iloc[0, 1]
        n = len(num)

        bias_correction = (den_mean / num_mean ** 3) * (num_var / n) - cov / (n * num_mean ** 2)
        mean = den_mean / num_mean - 1 + bias_correction
        var = den_var / num_mean ** 2 - 2 * (den_mean / num_mean ** 3) * cov + (den_mean ** 2 / num_mean ** 4) * num_var

        return mean, var

    def _taylor_params(self, x: pd.DataFrame) -> Tuple[float, float]:
        """ Calculated expectation and variance for ratio metric using Taylor expansion approximation.

        Source: https://www.stat.cmu.edu/~hseltman/files/ratio.pdf.

        Args:
            x (pandas.DataFrame): Pandas DataFrame of particular group (A, B, etc).

        Returns:
            Tuple[float, float]: Mean and variance of ratio metric.
        """
        num = x[self.params.data_params.numerator]
        den = x[self.params.data_params.denominator]
        mean = num.mean() / den.mean() - x[[self.params.data_params.numerator, self.params.data_params.denominator]].cov().iloc[0, 1] \
            / (den.mean() ** 2) + den.var() * num.mean() / (den.mean() ** 3)
        var = (num.mean() ** 2) / (den.mean() ** 2) * (num.var() / (num.mean() ** 2) - 2 *
            x[[self.params.data_params.numerator, self.params.data_params.denominator]].cov().iloc[0, 1]) / \
            (num.mean() * den.mean() + den.var() / (den.mean() ** 2))

        return mean, var

    def bucketing(self):
        """Performs bucketing in order to accelerate results computation.

        Returns:
            ABTest: New instance of ``ABTest`` class with modified control and treatment.
        """
        params_new = copy.deepcopy(self.params)
        params_new.data_params.control = self.__bucketize(self.params.data_params.control)
        params_new.data_params.treatment = self.__bucketize(self.params.data_params.treatment)

        return ABTest(self.__dataset, params_new)

    def cuped(self):
        """Performs CUPED for variance reduction.

        Returns:
            ABTest: New instance of ``ABTest`` class with modified control and treatment.
        """
        self.__check_required_columns(self.__dataset, 'cuped')
        result_df = VarianceReduction.cuped(self.__dataset,
                                            target=self.params.data_params.target,
                                            groups=self.params.data_params.group_col,
                                            covariate=self.params.data_params.covariate)

        params_new = copy.deepcopy(self.params)
        params_new.data_params.control = self.__get_group(self.params.data_params.control_name, result_df)
        params_new.data_params.treatment = self.__get_group(self.params.data_params.treatment_name, result_df)

        return ABTest(result_df, params_new)

    def cupac(self):
        """Performs CUPAC for variance reduction.

        Returns:
            ABTest: New instance of ``ABTest`` class with modified control and treatment.
        """
        self.__check_required_columns(self.__dataset, 'cupac')
        result_df = VarianceReduction.cupac(self.__dataset,
                                            target_prev=self.params.data_params.target_prev,
                                            target_now=self.params.data_params.target,
                                            factors_prev=self.params.data_params.predictors_prev,
                                            factors_now=self.params.data_params.predictors,
                                            groups=self.params.data_params.group_col)

        params_new = copy.deepcopy(self.params)
        params_new.data_params.control = self.__get_group(self.params.data_params.control_name, result_df)
        params_new.data_params.treatment = self.__get_group(self.params.data_params.treatment_name, result_df)

        return ABTest(result_df, params_new)

    def linearization(self) -> None:
        """Creates linearized continuous metric based on ratio-metric.
        Important: there is an assumption that all data is already grouped by user
        s.t. numerator for user = sum of numerators for user for different time periods
        and denominator for user = sum of denominators for user for different time periods

        Source: https://research.yandex.com/publications/148.
        """
        if not self.params.data_params.is_grouped:
            not_ratio_columns = self.__dataset.columns[~self.__dataset.columns.isin([self.params.data_params.numerator,
                                                                                     self.params.data_params.denominator])].tolist()
            df_grouped = self.__dataset.groupby(by=not_ratio_columns, as_index=False).agg({
                self.params.data_params.numerator: 'sum',
                self.params.data_params.denominator: 'sum'
            })
            self.__dataset = df_grouped

        x = self.__dataset.loc[self.__dataset[self.params.data_params.group_col] == self.params.data_params.control_name]
        k = round(sum(x[self.params.data_params.numerator]) / sum(x[self.params.data_params.denominator]), 4)

        self.__dataset.loc[:, f"{self.params.data_params.numerator}_{self.params.data_params.denominator}"] = \
            self.__dataset[self.params.data_params.numerator] - k * self.__dataset[self.params.data_params.denominator]
        self.target = f"{self.params.data_params.numerator}_{self.params.data_params.denominator}"

    def plot(self) -> None:
        """Plot experiment.

        Plot figure type depends on the following parameters:

        - hypothesis_params.metric_name
        - hypothesis_params.strategy
        """
        if self.params.hypothesis_params.metric_name == 'mean':
            Graphics.plot_mean_experiment(self.params)
        elif self.params.hypothesis_params.metric_name == 'median':
            Graphics.plot_median_experiment(self.params)

    def resplit_df(self):
        """Resplit dataframe.

        Returns:
            ABTest: New instance of ``ABTest`` class with modified control and treatment.
        """
        resplit_params = ResplitParams(
            group_col=self.params.data_params.group_col,
            strata_col=self.params.data_params.strata_col
        )
        resplitter = ResplitBuilder(self.__dataset, resplit_params)
        new_dataset = resplitter.collect()

        return ABTest(new_dataset, self.params)

    def test_boot_fp(self) -> stat_test_typing:
        """ Performs bootstrap hypothesis testing by calculation by calculation of false positives.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        x = self.params.data_params.control
        y = self.params.data_params.treatment

        metric_diffs: List[float] = []
        for _ in range(self.params.hypothesis_params.n_boot_samples):
            x_boot = np.random.choice(x, size=x.shape[0], replace=True)
            y_boot = np.random.choice(y, size=y.shape[0], replace=True)
            metric_diffs.append(self.params.hypothesis_params.metric(x_boot) - self.params.hypothesis_params.metric(y_boot))
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

        test_result: int = 0  # 0 - cannot reject H0, 1 - reject H0
        if false_positive <= self.params.hypothesis_params.alpha:
            test_result = 1

        result = {
            'stat': None,
            'p-value': false_positive,
            'result': test_result
        }
        return result

    def test_boot_confint(self) -> stat_test_typing:
        """ Performs bootstrap confidence interval and zero
        statistical significance.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        x = self.params.data_params.control
        y = self.params.data_params.treatment

        metric_diffs: List[float] = []
        for _ in range(self.params.hypothesis_params.n_boot_samples):
            x_boot = np.random.choice(x, size=x.shape[0], replace=True)
            y_boot = np.random.choice(y, size=y.shape[0], replace=True)
            metric_diffs.append(self.params.hypothesis_params.metric(x_boot) -
                                self.params.hypothesis_params.metric(y_boot))
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        boot_mean = pd_metric_diffs.mean()
        boot_std = pd_metric_diffs.std()
        zero_pvalue = norm.sf(0, loc=boot_mean, scale=boot_std)

        test_result: int = 0  # 0 - cannot reject H0, 1 - reject H0
        if self.params.hypothesis_params.alternative == 'two-sided':
            left_quant = self.params.hypothesis_params.alpha / 2
            right_quant = 1 - self.params.hypothesis_params.alpha / 2
            ci = pd_metric_diffs.quantile([left_quant, right_quant])
            ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

            if ci_left > 0 or ci_right < 0:  # 0 is not in critical area
                test_result = 1
        elif self.params.hypothesis_params.alternative == 'left':
            left_quant = self.params.hypothesis_params.alpha
            ci = pd_metric_diffs.quantile([left_quant])
            ci_left = float(ci.iloc[0])
            if ci_left < 0:  # 0 is not is critical area
                test_result = 1
        elif self.params.hypothesis_params.alternative == 'right':
            right_quant = self.params.hypothesis_params.alpha
            ci = pd_metric_diffs.quantile([right_quant])
            ci_right = float(ci.iloc[0])
            if 0 < ci_right:  # 0 is not in critical area
                test_result = 1

        result = {
            'stat': None,
            'p-value': zero_pvalue,
            'result': test_result
        }
        return result

    def test_boot_ratio(self, x: pd.DataFrame, y: pd.DataFrame) -> stat_test_typing:
        """Performs bootstrap for ratio-metric.

        Args:
            x (pandas.DataFrame): Control group dataframe.
            y (pandas.DataFrame): Treatment group dataframe.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        if x is None and y is None:
            x = self.__dataset[self.__dataset[self.params.data_params.group_col] == self.params.data_params.control_name]
            y = self.__dataset[self.__dataset[self.params.data_params.group_col] == self.params.data_params.treatment_name]

        a_metric_total = sum(x[self.params.data_params.numerator]) / sum(x[self.params.data_params.denominator])
        b_metric_total = sum(y[self.params.data_params.numerator]) / sum(y[self.params.data_params.denominator])
        origin_mean = b_metric_total - a_metric_total
        boot_diffs = []
        boot_a_metric = []
        boot_b_metric = []

        for _ in range(self.params.hypothesis_params.n_boot_samples):
            a_ids = x[self.params.data_params.id_col].sample(x[self.params.data_params.id_col].nunique(), replace=True)
            b_ids = y[self.params.data_params.id_col].sample(y[self.params.data_params.id_col].nunique(), replace=True)

            a_boot = x[x[self.params.data_params.id_col].isin(a_ids)]
            b_boot = y[y[self.params.data_params.id_col].isin(b_ids)]
            a_boot_metric = sum(a_boot[self.params.data_params.numerator]) / sum(a_boot[self.params.data_params.denominator])
            b_boot_metric = sum(b_boot[self.params.data_params.numerator]) / sum(b_boot[self.params.data_params.denominator])
            boot_a_metric.append(a_boot_metric)
            boot_b_metric.append(b_boot_metric)
            boot_diffs.append(b_boot_metric - a_boot_metric)

        # correction
        boot_mean = np.mean(boot_diffs)
        delta = abs(origin_mean - boot_mean)
        boot_diffs = [boot_diff + delta for boot_diff in boot_diffs]
        pd_metric_diffs = pd.DataFrame(boot_diffs)

        left_quant = self.params.hypothesis_params.alpha / 2
        right_quant = 1 - self.params.hypothesis_params.alpha / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        test_result: int = 0  # 0 - cannot reject H0, 1 - reject H0
        if ci_left > 0 or ci_right < 0:  # left border of ci > 0 or right border of ci < 0
            test_result = 1

        result = {
            'stat': None,
            'p-value': None,
            'result': test_result
        }
        return result

    def test_boot_welch(self) -> stat_test_typing:
        r""" Performs Welch's t-test for independent samples with unequal number of observations and variance.

        Welch's t-test is used as a wider approaches with less restrictions on samples size as in Student's t-test.

        Statistic of the test:

        .. math::
            t = \frac{\hat{X}_1 - \hat{X}_2}{\sqrt{\frac{s_1}{\sqrt{N_1}} + \frac{s_2}{\sqrt{N_2}}}}.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        x = self.params.data_params.control
        y = self.params.data_params.treatment

        t_calc: int = 0
        for _ in range(self.params.hypothesis_params.n_boot_samples):
            x_boot = np.random.choice(x, size=x.shape[0], replace=True)
            y_boot = np.random.choice(y, size=y.shape[0], replace=True)

            t_boot = (np.mean(x_boot) - np.mean(y_boot)) / (np.var(x_boot) / x_boot.shape[0] + np.var(y_boot) / y_boot.shape[0])
            test_res = ttest_ind(x_boot, y_boot, equal_var=False, alternative=self.params.hypothesis_params.alternative)

            if t_boot >= test_res[1]:
                t_calc += 1

        pvalue = t_calc / self.params.hypothesis_params.n_boot_samples

        test_result: int = 0  # 0 - cannot reject H0, 1 - reject H0
        if pvalue <= self.params.hypothesis_params.alpha:
            test_result = 1

        result = {
            'stat': None,
            'p-value': pvalue,
            'result': test_result
        }
        return result

    def test_buckets(self) -> stat_test_typing:
        """ Performs buckets hypothesis testing.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        x = self.params.data_params.control
        y = self.params.data_params.treatment

        np.random.shuffle(x)
        np.random.shuffle(y)
        x_new = np.array([self.params.hypothesis_params.metric(x)
                          for x in np.array_split(x, self.params.hypothesis_params.n_buckets)])
        y_new = np.array([self.params.hypothesis_params.metric(y)
                          for y in np.array_split(y, self.params.hypothesis_params.n_buckets)])

        test_result: int = 0
        if (shapiro(x_new).pvalue >= self.params.hypothesis_params.alpha) \
                and (shapiro(y_new).pvalue >= self.params.hypothesis_params.alpha):
            stat, pvalue = ttest_ind(x_new, y_new, equal_var=False, alternative=self.params.hypothesis_params.alternative)
            if pvalue <= self.params.hypothesis_params.alpha:
                test_result = 1
        else:
            def metric(arr: np.array):
                modes, _ = mode(arr)
                return sum(modes) / len(modes)
            self.params.hypothesis_params.metric = metric
            stat, pvalue, test_result = self.test_boot_confint()

        result = {
            'stat': stat,
            'p-value': pvalue,
            'result': test_result
        }
        return result

    def test_chisquare(self) -> stat_test_typing:
        """Performs Chi-Square test.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        x = self.__get_group(self.params.data_params.control_name, self.dataset)
        y = self.__get_group(self.params.data_params.treatment_name, self.dataset)

        observed = np.array([sum(y), len(y) - sum(y)])
        expected = np.array([sum(x), len(x) - sum(x)])
        stat, pvalue = chisquare(observed, expected)

        test_result: int = 0
        if pvalue <= self.params.hypothesis_params.alpha:
            test_result = 1

        result = {
            'stat': stat,
            'p-value': pvalue,
            'result': test_result
        }
        return result

    def test_delta_ratio(self) -> stat_test_typing:
        """ Delta method with bias correction for ratios.

        Source: https://arxiv.org/pdf/1803.06336.pdf.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        x = self.__dataset[self.__dataset[self.params.data_params.group_col] == self.params.data_params.control_name]
        y = self.__dataset[self.__dataset[self.params.data_params.group_col] == self.params.data_params.treatment_name]

        ctrl_mean, ctrl_var = self._delta_params(x)
        treat_mean, treat_var = self._delta_params(y)

        return self._manual_ttest(ctrl_mean, ctrl_var, x.shape[0], treat_mean, treat_var, y.shape[0])

    def test_mannwhitney(self) -> stat_test_typing:
        r"""Performs Mann-Whitney U test.

        Test works on continues metrics and their ranks.

        Assumptions of Mann-Whitney test:

        1. Independence of observations.
        2. Same shape of metric distributions.

        Statistic of the test:

        .. math::
            U = \sum_{i=1}^{n} \sum_{j=1}^{m} S(X_i, Y_j).

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        x = self.params.data_params.control
        y = self.params.data_params.treatment

        if self.params.hypothesis_params.metric_name != 'median':
            warnings.warn('Metric of the test is {}, \
                        but you use mann-whitney test with it'.format(self.params.hypothesis_params.metric_name))

        test_result: int = 0
        stat, pvalue = mannwhitneyu(x, y, alternative=self.params.hypothesis_params.alternative)

        if pvalue <= self.params.hypothesis_params.alpha:
            test_result = 1

        result = {
            'stat': stat,
            'p-value': pvalue,
            'result': test_result
        }
        return result

    def test_strat_confint(self) -> stat_test_typing:
        """ Performs stratification with confidence interval.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        metric_diffs: List[float] = []
        x = self.__dataset.loc[self.__dataset[self.params.data_params.group_col] == self.params.data_params.control_name]
        y = self.__dataset.loc[self.__dataset[self.params.data_params.group_col] == self.params.data_params.treatment_name]

        for _ in range(self.params.hypothesis_params.n_boot_samples):
            x_strata_metric = 0
            y_strata_metric = 0
            for strat in self.params.hypothesis_params.strata_weights.keys():
                x_strata = x.loc[x[self.params.hypothesis_params.strata] == strat, self.params.data_params.target]
                y_strata = y.loc[y[self.params.hypothesis_params.strata] == strat, self.params.data_params.target]
                x_strata_metric += (self.params.hypothesis_params
                                    .metric(np.random.choice(x_strata, size=x_strata.shape[0] // 2, replace=False)) *
                                    self.params.hypothesis_params.strata_weights[strat])
                y_strata_metric += (self.params.hypothesis_params
                                    .metric(np.random.choice(y_strata, size=y_strata.shape[0] // 2, replace=False)) *
                                    self.params.hypothesis_params.strata_weights[strat])
            metric_diffs.append(self.params.hypothesis_params.metric(x_strata_metric) - self.params.hypothesis_params.metric(y_strata_metric))
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        left_quant = self.params.hypothesis_params.alpha / 2
        right_quant = 1 - self.params.hypothesis_params.alpha / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        test_result: int = 0  # 0 - cannot reject H0, 1 - reject H0
        if ci_left > 0 or ci_right < 0:  # left border of ci > 0 or right border of ci < 0
            test_result = 1

        result = {
            'stat': None,
            'p-value': None,
            'result': test_result
        }
        return result

    def test_taylor_ratio(self) -> stat_test_typing:
        """ Calculate expectation and variance of ratio for each group and then use t-test for hypothesis testing.

        Source: http://www.stat.cmu.edu/~hseltman/files/ratio.pdf.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        x = self.__dataset[self.__dataset[self.params.data_params.group_col] == self.params.data_params.control_name]
        y = self.__dataset[self.__dataset[self.params.data_params.group_col] == self.params.data_params.treatment_name]

        ctrl_mean, ctrl_var = self._taylor_params(x)
        treat_mean, treat_var = self._taylor_params(y)

        return self._manual_ttest(ctrl_mean, ctrl_var, x.shape[0], treat_mean, treat_var, y.shape[0])

    def test_welch(self) -> stat_test_typing:
        """Performs Welch's t-test.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        x = self.params.data_params.control
        y = self.params.data_params.treatment

        normality_passed = (shapiro(x).pvalue >= self.params.hypothesis_params.alpha) \
            and (shapiro(y).pvalue >= self.params.hypothesis_params.alpha)

        if not normality_passed:
            warnings.warn('One or both distributions are not normally distributed')
        if self.params.hypothesis_params.metric_name != 'mean':
            warnings.warn('Metric of the test is {}, \
                        but you use t-test with it'.format(self.params.hypothesis_params.metric_name))

        test_result: int = 0
        stat, pvalue = ttest_ind(x, y, equal_var=False, alternative=self.params.hypothesis_params.alternative)

        if pvalue <= self.params.hypothesis_params.alpha:
            test_result = 1

        result = {
            'stat': stat,
            'p-value': pvalue,
            'result': test_result
        }
        return result

    def test_z_proportions(self) -> stat_test_typing:
        r"""Performs z-test for proportions.

        The two-proportions z-test is used to compare two observed proportions.

        Statistic of the test:

        .. math::
            Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{n_1} + \frac{1}{n_2})}}.

        Returns:
            stat_test_typing: Dictionary with following properties: ``test statistic``, ``p-value``, ``test result``. Test result: 1 - significant different, 0 - insignificant difference.
        """
        x = self.__get_group(self.params.data_params.control_name, self.dataset)
        y = self.__get_group(self.params.data_params.treatment_name, self.dataset)

        count = np.array([sum(x), sum(y)])
        nobs = np.array([len(x), len(y)])
        stat, pvalue = proportions_ztest(count, nobs)

        test_result: int = 0
        if pvalue <= self.params.hypothesis_params.alpha:
            test_result = 1

        result = {
            'stat': stat,
            'p-value': pvalue,
            'result': test_result
        }
        return result
