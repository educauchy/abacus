from typing import Dict, List, Any, Union, Optional, Callable
from abc import ABC
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, mode, chisquare
from scipy.stats.stats import Power_divergenceResult
from analysis.params import StatTestParams, PeriodStatTestParams, PreperiodStatTestParams


class StatTest(ABC):
    SEED = 10

    SIGNIFICANCE_SUFFIX = "_significance"

    def __init__(self,
                 control_guests_metric: List[float], target_guests_metric: List[float],
                 metric_name: str, stat_test_params: StatTestParams):
        """Calculate effect bootstrap or ttest|mannwhitneyu|chisquare tests

        Args:
            control_guests_metric: metric distribution for control group
            target_guests_metric: metric distribution for target group
            stat_test_params: parameters for statistical tests (pvalue, bootstrap iterations, etc.)

        """
        np.random.seed(self.SEED)
        self.control_guests_metric = np.array(control_guests_metric)
        self.target_guests_metric = np.array(target_guests_metric)
        self.metric_name = metric_name
        self.stat_test_params = stat_test_params
        self._sort_and_shuffle_metrics()

    def _sort_and_shuffle_metrics(self):
        """For more stable effects, because spark always shuffles data"""
        self.control_guests_metric.sort()
        self.target_guests_metric.sort()
        np.random.shuffle(self.control_guests_metric)
        np.random.shuffle(self.target_guests_metric)


class PeriodStatTest(StatTest):
    BOOTSTRAP_ITERATIONS_PER_BUCKET = 5  # influence on speed and memory consumption

    def __init__(self,
                 control_guests_metric: List[float], target_guests_metric: List[float],
                 metric_name: str, stat_test_params: PeriodStatTestParams):
        super().__init__(control_guests_metric, target_guests_metric, metric_name, stat_test_params)
        self._effect_bootstrap_distribution = np.array([], dtype="float64")
        self._control_mean_bootstrap_distribution = np.array([], dtype="float64")
        self._target_mean_bootstrap_distribution = np.array([], dtype="float64")

    def calculate_period_effect(self) -> Dict[str, Union[float, int]]:
        """
        Returns: map for metric: amount of group effect, significance and size of target group
        """
        self._calculate_bootstrap_distribution()
        effect_stats = self._calculate_effect_stats()
        return effect_stats

    @staticmethod
    def bootstrap_calculate_effect(metric: Optional[Callable[[Any], float]],
                                   target_guests_metric: pd.DataFrame, 
                                   control_guests_metric: pd.DataFrame,
                                   stat_test_params: StatTestParams):
        """
        Perform bootstrap confidence interval
        :param X: Null hypothesis distribution
        :param Y: Alternative hypothesis distribution
        :param metric: Custom metric (mean, median, percentile (1, 2, ...), etc
        :returns: Ratio of rejected H0 hypotheses to number of all tests
        """
        len_target = len(target_guests_metric)
        len_control = len(control_guests_metric)
        metric_diffs: List[float] = []
        for _ in range(stat_test_params.bootstrap_iterations_number):
            target_boot = np.random.choice(target_guests_metric,
                                      size=len_target, replace=True)
            control_boot = np.random.choice(control_guests_metric,
                                      size=len_control, replace=True)
            metric_diffs.append(metric(target_boot) - metric(control_boot))
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        left_quant = (1 - stat_test_params.confidence_level) / 2
        right_quant = stat_test_params.confidence_level + left_quant
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        test_result: int = 0 # 0 - cannot reject H0, 1 - reject H0
        if ci_left > 0 or ci_right < 0: # left border of ci > 0 or right border of ci < 0
                test_result = 1

        return {
            "effect_control_group_size": int(len_control),
            "effect_target_group_size": int(len_target),
            f"effect_significance": int(test_result)
        }

    @staticmethod
    def buckets_calculate_effect(metric: Optional[Callable[[Any], float]],
                                target_guests_metric: pd.DataFrame,
                                control_guests_metric: pd.DataFrame,
                                stat_test_params: StatTestParams,
                                n_buckets: int = 1000) -> int:
        """
        Perform buckets hypothesis testing
        :param X: Null hypothesis distribution
        :param Y: Alternative hypothesis distribution
        :param metric: Custom metric (mean, median, percentile (1, 2, ...), etc
        :param n_buckets: Number of buckets
        :return: Test result: 1 - significant different, 0 - insignificant difference
        """
        np.random.shuffle(target_guests_metric)
        np.random.shuffle(control_guests_metric)

        Y_new = np.array([ metric(x) for x in np.array_split(control_guests_metric, n_buckets) ])
        X_new = np.array([ metric(y) for y in np.array_split(target_guests_metric, n_buckets) ])

        alpha = 1 - stat_test_params.confidence_level
        test_result: int = 0
        if shapiro(X_new)[1] >= alpha and shapiro(Y_new)[1] >= alpha:
            _, pvalue = ttest_ind(X_new, Y_new, equal_var=False, alternative='two-sided')
            if pvalue <= ():
                test_result = 1
        else:
            def metric_bucket(X: np.array):
                modes, _ = mode(X)
                return sum(modes) / len(modes)
            test_result = PeriodStatTest.bootstrap_calculate_effect(metric_bucket, 
                                                          X_new, Y_new, stat_test_params)

        return test_result


    def _calculate_bootstrap_distribution(self):
        len_control = len(self.control_guests_metric)
        len_target = len(self.target_guests_metric)

        bootstrap_buckets = int(np.ceil(
            self.stat_test_params.bootstrap_iterations_number /
            self.BOOTSTRAP_ITERATIONS_PER_BUCKET
        ))

        for _ in range(bootstrap_buckets):
            target_sample_mean = np.random.choice(self.target_guests_metric,
                                                  size=(self.BOOTSTRAP_ITERATIONS_PER_BUCKET, len_target),
                                                  replace=True).mean(axis=1)

            control_sample_mean = np.random.choice(self.control_guests_metric,
                                                   size=(self.BOOTSTRAP_ITERATIONS_PER_BUCKET, len_control),
                                                   replace=True).mean(axis=1)
            effect = target_sample_mean - control_sample_mean

            self._effect_bootstrap_distribution = np.append(self._effect_bootstrap_distribution, effect)
            self._control_mean_bootstrap_distribution = np.append(self._control_mean_bootstrap_distribution, control_sample_mean)
            self._target_mean_bootstrap_distribution = np.append(self._target_mean_bootstrap_distribution, target_sample_mean)

    def _calculate_effect_stats(self) -> Dict[str, Union[float, int]]:
        left_bound = (1 - self.stat_test_params.confidence_level) / 2
        effect_left_bound = np.percentile(self._effect_bootstrap_distribution, left_bound * 100)
        right_bound = self.stat_test_params.confidence_level + left_bound
        effect_right_bound = np.percentile(self._effect_bootstrap_distribution, right_bound * 100)
        effect_mean = np.mean(self._effect_bootstrap_distribution)

        is_significant_positive_effect = (effect_left_bound > 0) & (effect_mean > 0)
        is_significant_negative_effect = (effect_right_bound < 0) & (effect_mean < 0)
        is_significant_effect = is_significant_positive_effect | is_significant_negative_effect

        guest_means_fraction_delta = (
            np.mean(self._target_mean_bootstrap_distribution) /
            np.mean(self._control_mean_bootstrap_distribution)
            - 1
        )
        control_group_size = len(self.control_guests_metric)
        target_group_size = len(self.target_guests_metric)
        group_effect = effect_mean * target_group_size
        return {
            "effect_control_group_size": int(control_group_size),
            "effect_target_group_size": int(target_group_size),
            f"effect_{self.metric_name}_guest_means_fraction_delta": float(guest_means_fraction_delta),
            f"effect_{self.metric_name}_group": float(group_effect),
            f"effect_{self.metric_name}{self.SIGNIFICANCE_SUFFIX}": int(is_significant_effect)
        }


class PreperiodStatTest(StatTest):
    def calculate_preperiod_check(self):
        """
        Returns: map for metric: test pvalue, significance of this pvalue and size of target group
        """
        map_test_to_metrics = self.stat_test_params.map_test_to_metrics

        if self.metric_name in map_test_to_metrics.get(PreperiodStatTestParams.ttest_ind, []):
            test_result = ttest_ind(
                self.target_guests_metric, self.control_guests_metric, equal_var=False
            )
        elif self.metric_name in map_test_to_metrics.get(PreperiodStatTestParams.mannwhitneyu, []):
            test_result = mannwhitneyu(
                self.target_guests_metric, self.control_guests_metric, alternative="two-sided"
            )
        elif self.metric_name in map_test_to_metrics.get(PreperiodStatTestParams.chisquare, []):
            test_result = self._compute_chisquare(
                self.target_guests_metric, self.control_guests_metric
            )
        else:
            raise ValueError(f"Metric {self.metric_name} doesn't mention in map_test_to_metrics")

        target_group_size = len(self.target_guests_metric)
        pvalue = test_result.pvalue
        is_significant_pvalue = (self.stat_test_params.pvalue > pvalue)
        return {
            "test_target_group_size": int(target_group_size),
            f"{self.metric_name}_pvalue": float(pvalue),
            f"{self.metric_name}{self.SIGNIFICANCE_SUFFIX}": int(is_significant_pvalue),
        }

    def _compute_chisquare(self, observed: List[float], expected: List[float]) -> Power_divergenceResult:
        all_categories = set(observed) | set(expected)
        default_counter = dict(zip(all_categories, (0 for _ in range(len(all_categories)))))
        observed_cat_counts = list(map(
            lambda x: x[1], sorted({**default_counter, **Counter(observed)}.items(), key=lambda x: x[0])
        ))  # last passed dict value will be actual while merging dicts
        expected_cat_counts = list(map(
            lambda x: x[1], sorted({**default_counter, **Counter(expected)}.items(), key=lambda x: x[0])
        ))
        test_result = chisquare(observed_cat_counts, expected_cat_counts)
        return test_result
