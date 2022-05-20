import warnings

import numpy as np
import pandas as pd
import os
import sys
import yaml
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, mode, t
from typing import Dict, List, Any, Union, Optional, Callable, Tuple
from tqdm.auto import tqdm
#from splitter import Splitter
#from graphics import Graphics
from analysis.variance_reduction import VarianceReduction
#from hyperopt import hp, fmin, tpe, Trials, space_eval

metric_name_typing = Union[str, Callable[[np.array], Union[int, float]]]

class ABTest:
    """Perform AB-test"""
    def __init__(self, dataset: pd.DataFrame, config: Dict[Any, Any] = None,
                 startup_config: bool = False) -> None:
        if config is not None:
            self.dataset = dataset 
            self.startup_config = startup_config
            self.config: Dict[Any, Any] = {}
            self.config_load(config)
        else:
            raise Exception('You must pass config file')

    @property
    def beta(self) -> float:
        return self.config.beta

    @beta.setter
    def beta(self, value: float) -> None:
        if 0 <= value <= 1:
            self.config.beta = value
        else:
            raise Exception('Beta must be inside interval [0, 1]. Your input: {}.'.format(value))

    @property
    def split_ratios(self) -> Tuple[float, float]:
        return self.config.split_ratios

    @split_ratios.setter
    def split_ratios(self, value: Tuple[float, float]) -> None:
        if isinstance(value, tuple) and len(value) == 2 and sum(value) == 1:
            self.config.split_ratios = value
        else:
            raise Exception('Split ratios must be a tuple with two shares which has a sum of 1. Your input: {}.'.format(value))

    @property
    def alternative(self) -> str:
        return self.config.alternative

    @alternative.setter
    def alternative(self, value: str) -> None:
        if value in ['less', 'greater', 'two-sided']:
            self.config.alternative = value
        else:
            raise Exception("Alternative must be either 'less', 'greater', or 'two-sided'. Your input: '{}'.".format(value))

    @property
    def metric_type(self) -> str:
        return self.config.metric_type

    @metric_type.setter
    def metric_type(self, value: str) -> None:
        if value in ['solid', 'ratio']:
            self.config.metric_type = value
        else:
            raise Exception("Metric type must be either 'solid' or 'ratio'. Your input: '{}'.".format(value))

    @property
    def metric_name(self) -> metric_name_typing:
        return self.metric_name

    @metric_name.setter
    def metric_name(self, value: metric_name_typing) -> None:
        if value in ['mean', 'median'] or callable(value):
            self.config.metric_name = value
        else:
            raise Exception("Metric name must be either 'mean' or 'median'. Your input: '{}'.".format(value))

    @property
    def target(self) -> str:
        return self.config.target

    @target.setter
    def target(self, value: str) -> None:
        if value in self.config.dataset.columns:
            self.config.target = value
        else:
            raise Exception('Target column name must be presented in dataset. Your input: {}.'.format(value))

    @property
    def denominator(self) -> str:
        return self.config.denominator

    @denominator.setter
    def denominator(self, value: str) -> None:
        if value in self.config.dataset.columns:
            self.config.denominator = value
        else:
            raise Exception('Denominator column name must be presented in dataset. Your input: {}.'.format(value))

    @property
    def numerator(self) -> str:
        return self.config.numerator

    @numerator.setter
    def numerator(self, value: str) -> None:
        if value in self.dataset.columns:
            self.config.numerator = value
        else:
            raise Exception('Numerator column name must be presented in dataset. Your input: {}.'.format(value))

    @property
    def group_col(self) -> str:
        return self.config.group_col

    @group_col.setter
    def group_col(self, value: str) -> None:
        if value in self.dataset.columns:
            self.config.group_col = value
        else:
            raise Exception('Group column name must be presented in dataset. Your input: {}.'.format(value))

    def config_load(self, config: Dict[Any, Any]) -> None:
        if self.startup_config:
            self.config['alpha']          = config['hypothesis']['alpha']
            self.config['beta']           = config['hypothesis']['beta']
            self.config['alternative']    = config['hypothesis']['alternative']
            self.config['n_buckets']      = config['hypothesis']['n_buckets']
            self.config['n_boot_samples'] = config['hypothesis']['n_boot_samples']
            self.config['split_ratios']   = config['hypothesis']['split_ratios']
            self.config['metric_type']    = config['metric']['type']
            self.config['metric_name']    = config['metric']['name']

            #if config['data']['path'] != '':
            #    df: pd.DataFrame = self.load_dataset(config['data']['path'])
            #    n_rows = df.shape[0] + 1 if config['data']['n_rows'] == -1 else config['data']['n_rows']
            #    df = df.iloc[:n_rows]
            #    self.config['dataset'] = df.to_dict()
            #    self.dataset = df

            self.config['target']         = config['data']['target']
            self.config['predictors']     = config['data']['predictors']
            self.config['numerator']      = config['data']['numerator']
            self.config['denominator']    = config['data']['denominator']
            self.config['covariate']      = config['data']['covariate']
            self.config['group_col']      = config['data']['group_col']
            self.config['id_col']         = config['data']['id_col']
            self.config['is_grouped']     = config['data']['is_grouped']
            self.config['target_prev']    = config['data']['target_prev']
            self.config['predictors_prev']     = config['data']['predictors_prev']

            self.config['control']        = self.dataset.loc[self.dataset[self.config['group_col']] == 'A', \
                                                                    self.config['target']].to_numpy()
            self.config['treatment']      = self.dataset.loc[self.dataset[self.config['group_col']] == 'B', \
                                                                    self.config['target']].to_numpy()

        else:
            self.config['alpha']          = config['alpha']
            self.config['beta']           = config['beta']
            self.config['alternative']    = config['alternative']
            self.config['n_buckets']      = config['n_buckets']
            self.config['n_boot_samples'] = config['n_boot_samples']
            self.config['split_ratios']   = config['split_ratios']
            self.config['metric_type']    = config['metric_type']
            self.config['metric_name']    = config['metric_name']
            self.config['split_ratios']   = config['split_ratios']

            #if config['dataset'] != '':
            #    self.dataset: pd.DataFrame = pd.DataFrame.from_dict(config['dataset'])
            #    self.config['dataset'] = config['dataset']

            self.config['target']         = config['target']
            self.config['predictors']     = config['predictors']
            self.config['numerator']      = config['numerator']
            self.config['denominator']    = config['denominator']
            self.config['covariate']      = config['covariate']
            self.config['group_col']      = config['group_col']
            self.config['id_col']         = config['id_col']
            self.config['is_grouped']     = config['is_grouped']
            self.config['target_prev']    = config['target_prev']
            self.config['predictors_prev']     = config['predictors_prev']

            self.config['control']        = config['control']
            self.config['treatment']      = config['treatment']

        # self.splitter: Splitter = None
        # self.split_rates: List[float] = None
        # self.increment_list: List[float] = None
        # self.increment_extra: Dict[str, float] = None

    def load_dataset(self, path: str = '') -> pd.DataFrame:
        """
        Load dataset for analysis
        :param path: Path to the dataset for analysis
        :param id_col: Id column name
        :param target: Target column name
        :param numerator: Ratio numerator column name
        :param denominator: Ratio denominator column name
        """
        return self._read_file(path)

    def _read_file(self, path: str) -> pd.DataFrame:
        """
        Read file and return pandas dataframe
        :param path: Path to file
        :returns: Pandas DataFrame
        """
        _, file_ext = os.path.splitext(path)
        if file_ext == '.csv':
            return pd.read_csv(path, encoding='utf8')
        elif file_ext == '.xls' or file_ext == '.xlsx':
            return pd.read_excel(path, encoding='utf8')
    
    def __get_group(self, group_label: str = 'A'):
        group = self.dataset.loc[self.dataset[self.config['group_col']] == group_label, \
                                            self.config['target']].to_numpy()
        return group

    def test_hypothesis_boot_confint(self, metric: Optional[Callable[[Any], float]] = None) -> int:
        """
        Perform bootstrap confidence interval
        :param X: Null hypothesis distribution
        :param Y: Alternative hypothesis distribution
        :param metric: Custom metric (mean, median, percentile (1, 2, ...), etc
        :returns: Ratio of rejected H0 hypotheses to number of all tests
        """
        X = self.config['control']
        Y = self.config['treatment']

        metric_diffs: List[float] = []
        for _ in tqdm(range(self.config['n_boot_samples'])):
            x_boot = np.random.choice(X, size=X.shape[0], replace=True)
            y_boot = np.random.choice(Y, size=Y.shape[0], replace=True)
            metric_diffs.append( metric(x_boot) - metric(y_boot) )
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        left_quant = self.config['alpha'] / 2
        right_quant = 1 - self.config['alpha'] / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        test_result: int = 0 # 0 - cannot reject H0, 1 - reject H0
        if ci_left > 0 or ci_right < 0: # left border of ci > 0 or right border of ci < 0
            test_result = 1

        return test_result

    def cuped(self):
        vr = VarianceReduction()
        result_df = vr.cuped(self.dataset,
                            target=self.config['target'],
                            groups=self.config['group_col'],
                            covariate=self.config['covariate'])

        #self.config['dataset'] = result_df.to_dict()
        self.dataset = result_df

        self.config['control'] = self.__get_group('A')
        self.config['treatment'] = self.__get_group('B')

        return ABTest(result_df, self.config)

    def cupac(self):
        vr = VarianceReduction()
        result_df = vr.cupac(self.dataset,
                               target_prev=self.config['target_prev'],
                               target_now=self.config['target'],
                               factors_prev=self.config['predictors_prev'],
                               factors_now=self.config['predictors'],
                               groups=self.config['group_col'])
        self.dataset = result_df

        self.config['control'] = self.__get_group('A')
        self.config['treatment'] = self.__get_group('B')

        #self.config['dataset'] = result_df.to_dict()
        return ABTest(result_df, self.config)

    def __metric_calc(self, X: Union[List[Any], np.array]):
        if self.config['metric_name'] == 'mean':
            return np.mean(X)
        elif self.config['metric_name'] == 'median':
            return np.median(X)
        elif self.config['metric_name'] == 'custom':
            return self.config['metric'](X)

    def __bucketize(self, X: pd.DataFrame):
        np.random.shuffle(X)
        X_new = np.array([ self.__metric_calc(x) for x in np.array_split(X, self.config['n_buckets']) ])
        return X_new

    def bucketing(self):
        self.config['control']   = self.__bucketize(self.config['control'])
        self.config['treatment'] = self.__bucketize(self.config['treatment'])

        return ABTest(self.dataset, self.config)
