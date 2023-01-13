from typing import Optional, Dict, Tuple, List, Any, Callable, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, t
from collections import defaultdict
from hyperopt import hp, fmin, tpe, Trials, space_eval


class Simulation:
    def __init__(self, alpha: float = 0.05, beta: float = 0.2,
                 n_iter: int = 5000, stds: List[float] = None,
                 effect_sizes: List[float] = None,
                 sample_sizes: List[int] = None, mdes: List[float] = None):
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.stds = stds
        self.effect_sizes = effect_sizes
        self.sample_sizes = sample_sizes
        self.mdes = mdes

        self.split_rates: List[float] = None
        self.increment_list: List[float] = None
        self.increment_extra: Dict[str, float] = None

    def _add_increment(self, X: Union[pd.DataFrame, np.ndarray],
                       inc_value: Union[float, int]) -> np.ndarray:
        """ Add constant increment to a list

        Args:
            X (np.ndarray): Numpy array to modify
            inc_value (Union[float, int]): Constant addendum to each value

        Returns:
            numpy.ndarray: Modified X array
        """
        if self.config['metric_type'] == 'solid':
            return X + inc_value
        elif self.config['metric_type'] == 'ratio':
            X.loc[:, 'inced'] = X[self.config['numerator']] + inc_value
            X.loc[:, 'diff'] = X[self.config['denominator']] - X[self.config['numerator']]
            X.loc[:, 'rand_inc'] = np.random.randint(0, X['diff'] + 1, X.shape[0])
            X.loc[:, 'numerator_new'] = X[self.config['numerator']] + X['rand_inc']

            X[self.config['numerator']] = np.where(X['inced'] < X[self.config['denominator']], X['inced'], X['numerator_new'])
            return X[[self.config['numerator'], self.config['denominator']]]

    def set_increment(self, inc_var: List[float] = None, extra_params: Dict[str, float] = None) -> None:
        self.config['increment_list']  = inc_var
        self.config['increment_extra'] = extra_params

    def sample_size_simulation(self):
        ss = []
        ess = []
        ns = []
        for std in self.stds:
            for effect_size in self.effect_sizes:
                for sample_size in self.sample_sizes:
                    rejected = 0
                    for n in range(self.n_iter):
                        a = np.random.normal(0, std, sample_size)
                        b = np.random.normal(effect_size, std, sample_size)

                        if ttest_ind(a, b, equal_var=False)[0] < self.alpha:
                            rejected += 1

                    proportion = rejected / self.n_iter
                    if proportion > (1 - self.beta):
                        ss.append(std)
                        ess.append(effect_size)
                        ns.append(sample_size)
                        break

        data = pd.DataFrame({
            'sigma': ss,
            'effect_size': ess,
            'sample_size': ns
        })

        data_pivot = data.pivot(index='sigma', columns='effect_size', values='sample_size')

        plt.figure(figsize=(20, 12))
        sns.heatmap(data_pivot, cmap='Oranges', annot=True, fmt='g')
        sns.set(font_scale=2)
        plt.title('Расчет требуемого размера группы имитационно')
        plt.xlabel(r'$\mu_T - \mu_C$')
        plt.ylabel(r'$\sigma$')
        plt.show()


    def mde_simulation(self, strategy: str = 'simple_test', strata: Optional[str] = '',
                       metric_type: str = 'solid', n_boot_samples: Optional[int] = 10000, n_buckets: Optional[int] = None,
                       metric: Optional[Callable[[Any], float]] = None, strata_weights: Optional[Dict[str, float]] = None,
                       use_correction: Optional[bool] = True, to_csv: bool = False,
                       csv_path: str = None, verbose: bool = False) -> Dict[float, Dict[float, float]]:
        """ Simulation process of determining appropriate split rate and increment rate for experiment

        Args:
            n_iter: Number of iterations of simulation
            strategy: Name of strategy to use in experiment assessment
            strata: Strata column
            n_boot_samples: Number of bootstrap samples
            n_buckets: Number of buckets
            metric: Custom metric (mean, median, percentile (1, 2, ...), etc
            strata_weights: Pre-experiment weights for strata column
            use_correction: Whether or not to use correction for multiple tests
            to_csv: Whether or not to save result to a .csv file
            csv_path: CSV file path

        Returns:
            Dict with ratio of hypotheses rejected under certain split rate and increment
        """
        if n_boot_samples < 1:
            raise Exception('Number of bootstrap samples must be 1 or more. Your input: {}.'.format(n_boot_samples))
        self.n_boot_samples = n_boot_samples
        imitation_log: Dict[float, Dict[float, int]] = defaultdict(float)
        csv_pd = pd.DataFrame()
        for split_rate in self.split_rates:
            imitation_log[split_rate] = {}
            for inc in self.increment_list:
                imitation_log[split_rate][inc] = 0
                curr_iter = 0
                for it in range(self.n_iter):
                    curr_iter += 1
                    if verbose:
                        print(f'Split_rate: {split_rate}, inc_rate: {inc}, iter: {it}')
                    self._split_data(split_rate)
                    if self.__metric_type == 'solid':
                        control, treatment = self.dataset.loc[
                                                 self.dataset[self.__group_col] == 'A', self.target].to_numpy(), \
                                             self.dataset.loc[self.dataset[self.__group_col] == 'B', self.target].to_numpy()
                        treatment = self._add_increment('solid', treatment, inc)
                    elif self.__metric_type == 'ratio':
                        control, treatment = self.dataset.loc[
                                                 self.dataset[self.__group_col] == 'A', [self.numerator, self.denominator]], \
                                             self.dataset.loc[
                                                 self.dataset[self.__group_col] == 'B', [self.numerator, self.denominator]]
                        treatment = self._add_increment('ratio', treatment, inc)

                    if strategy == 'simple_test':
                        test_result: int = self.test_hypothesis(control, treatment)
                        imitation_log[split_rate][inc] += test_result
                    elif strategy == 'delta_method':
                        test_result: int = self.delta_method(control, treatment)
                        imitation_log[split_rate][inc] += test_result
                    elif strategy == 'taylor_method':
                        test_result: int = self.taylor_method(control, treatment)
                        imitation_log[split_rate][inc] += test_result
                    elif strategy == 'boot_hypothesis':
                        pvalue: float = self.test_boot_hypothesis(control, treatment, use_correction=use_correction)
                        if pvalue <= self.__alpha:
                            imitation_log[split_rate][inc] += 1
                    elif strategy == 'boot_est':
                        pvalue: float = self.test_hypothesis_boot_est(control, treatment, metric=metric)
                        if pvalue <= self.__alpha:
                            imitation_log[split_rate][inc] += 1
                    elif strategy == 'boot_confint':
                        test_result: int = self.test_hypothesis_boot_confint(control, treatment, metric=metric)
                        imitation_log[split_rate][inc] += test_result
                    elif strategy == 'strata_confint':
                        test_result: int = self.test_hypothesis_strat_confint(metric=metric,
                                                                              strata_col=strata,
                                                                              weights=strata_weights)
                        imitation_log[split_rate][inc] += test_result
                    elif strategy == 'buckets':
                        test_result: int = self.test_hypothesis_buckets(control, treatment,
                                                                        metric=metric,
                                                                        n_buckets=n_buckets)
                        imitation_log[split_rate][inc] += test_result
                    elif strategy == 'formula':
                        pass

                    # do not need to proceed if already achieved desired level
                    if imitation_log[split_rate][inc] / curr_iter >= self.__beta:
                        break

                imitation_log[split_rate][inc] /= curr_iter

                row = pd.DataFrame({
                    'split_rate': [split_rate],
                    'increment': [inc],
                    'pval_sign_share': [imitation_log[split_rate][inc]]})
                csv_pd = csv_pd.append(row)

        if to_csv:
            csv_pd.to_csv(csv_path, index=False)
        return dict(imitation_log)

    def mde_hyperopt(self, n_iter: int = 20000, strategy: str = 'simple_test', params: Dict[str, List[float]] = None,
                     to_csv: bool = False, csv_path: str = None) -> None:
        def objective(params) -> float:
            split_rate, inc = params['split_rate'], params['inc']
            self._split_data(split_rate)

            control = self.config['control']
            treatment = self.config['treatment']

            treatment = self._add_increment('solid', treatment, inc)
            pvalue_mean = 0
            for it in range(n_iter):
                pvalue_mean += self.test_hypothesis()
            pvalue_mean /= n_iter
            return -pvalue_mean

        space = {}
        for param, values in params.items():
            space[param] = hp.uniform(param, values[0], values[1])
            # space[param] = hp.choice(param, )

        trials = Trials()
        print('\nSpace')
        print(space)
        best = fmin(objective,
                    space,
                    algo=tpe.suggest,
                    max_evals=10,
                    trials=trials
                    )
        print('\nBest')
        print(best)

        # Get the values of the optimal parameters
        best_params = space_eval(space, best)
        print('\nBest params')
        print(best_params)

    def sample_size(self, std: float = None, effect_size: float = None,
                    split_ratios: Tuple[float, float] = None) -> Tuple[int, int]:
        """ Calculation of sample size for each test group

        Args:
            :param std: Standard deviation of a test metric
            :param effect_size: Lift in metric
            :param group_shares: Shares of A and B groups

        Returns:
            Number of observations needed in each group
        """
        control_share, treatment_share = split_ratios if split_ratios is not None else self.config['split_ratios']
        if treatment_share == 0.5:
            alpha: float = (1 - self.config['alpha'] / 2) if self.config['alternative'] == 'two-sided' else (1 - self.config['alpha'])
            n_samples: int = round(2 * (t.ppf(alpha) + t.ppf(1 - self.config['beta'])) * std ** 2 / (effect_size ** 2), 0) + 1
            return (n_samples, n_samples)
        else:
            alpha: float = (1 - self.config['alpha'] / 2) if self.config['alternative'] == 'two-sided' else (1 - self.config['alpha'])
            n: int = round((((t.ppf(alpha) + t.ppf(1 - self.config['beta'])) * std ** 2 / (effect_size ** 2))) \
                           / (treatment_share * control_share), 0) + 1
            a_samples, b_samples = int(round(n * control_share, 0) + 1), int(round(n * treatment_share, 0) + 1)
            return (a_samples, b_samples)

    def mde(self, std: float, n_samples: int) -> float:
        """ Calculate Minimum Detectable Effect using Margin of Error formula

        Args:
            std: Pooled standard deviatioin
            n_samples: Number of samples for each group

        Returns:
            MDE, in absolute lift
        """
        alpha: float = (1 - self.config['alpha'] / 2) if self.config['alternative'] == 'two-sided' else (1 - self.config['alpha'])
        mde: float = np.sqrt( 2 * (t.ppf(alpha) + t.ppf(1 - self.config['beta'])) * std / n_samples )
        return mde
