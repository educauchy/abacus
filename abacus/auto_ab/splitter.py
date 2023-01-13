from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


class Splitter:
    def __init__(self, config: Dict[Any, Any] = None) -> None:
        if config is not None:
            self.config: Dict[Any, Any] = {}
            self.config_load(config)
        else:
            raise Exception('You must pass config file')

    def config_load(self, config: Dict[Any, Any]) -> None:
        self.config['metric_type']      = config['metric']['type']
        self.config['metric_name']      = config['metric']['name']
        self.config['split_rate']       = config['splitter']['split_rate']
        self.config['id_col']           = config['data']['id_col']
        self.config['cluster_col']      = config['data']['cluster_col']
        self.config['target']           = config['data']['target']
        self.config['numerator']        = config['data']['numerator']
        self.config['denominator']      = config['data']['denominator']

        self.config['alpha']            = config['aa_test']['alpha']
        self.config['aa_method']        = config['aa_test']['method']
        self.config['clustering_cols']  = config['data']['clustering_cols']
        self.config['to_cluster']       = config['aa_test']['to_cluster']
        self.config['n_clusters']       = config['aa_test']['n_clusters']

        if config['splitter']['type'] == 'custom':
            self.splitter = lambda x: np.mean(x)
        elif config['splitter']['type'] == 'builtin':
            self.splitter = self.__default_splitter

        if config['data']['path'] != '':
            df: pd.DataFrame = pd.read_csv(config['splitter']['data'], encoding='utf8')
            n_rows = df.shape[0] + 1 if config['data']['n_rows'] == -1 else config['data']['n_rows']
            df = df.iloc[:n_rows]
            self.config['dataset'] = df.to_dict()
            self.dataset = df

    def _split_data(self, split_rate: float) -> None:
        """ Add 'group' column

        Args:
            split_rate (float): Split rate of control/treatment
        """
        split_rate: float = self.params.splitter_params.split_rate if split_rate is None else split_rate
        self.dataset = self.config['splitter'].fit(self.dataset,
                                                   self.params.data_params.target,
                                                   self.params.data_params.numerator,
                                                   self.params.data_params.denominator,
                                                   split_rate)

    def __default_splitter(self, split_rate: float=0.5) -> pd.DataFrame:
        """Performs split into A and B using default splitter

        Args:
            split_rate (float): Share of control group

        Returns:
            pandas.DataFrame: Initial dataframe with additional 'group_col'
        """
        A_data, B_data = train_test_split(self.dataset, train_size=split_rate, random_state=0)
        A_data.loc[:, self.config['group_col']] = 'A'
        B_data.loc[:, self.config['group_col']] = 'B'
        Z = pd.concat([A_data, B_data]).reset_index(drop=True)
        return Z

    def __clustering(self) -> None:
        """ Clustering for dataset
        """
        X = self.dataset.copy()
        kmeans = KMeans(n_clusters=self.config['n_clusters'])
        kmeans.fit(X)
        self.dataset.loc[self.config['cluster_col']] = kmeans.predict(X)

    def __kl_divergence(self, n_bins: int=50) -> Tuple[float, float]:
        """ Kullback-Leibler divergence for two arrays of cluster ids for A/A test

        Args:
            n_bins (int): Number of clusters

        Returns:
            Tuple[float, float]: Kullback-Leibler divergence a to b and b to a
        """
        a = self.dataset.loc[self.dataset[self.config['group_col'] == 'A'], self.config['cluster_col']].tolist()
        b = self.dataset.loc[self.dataset[self.config['group_col'] == 'B'], self.config['cluster_col']].tolist()

        bins_arr = list(range(1, n_bins+1))
        a = np.histogram(a, bins=bins_arr, density=True)[0]
        b = np.histogram(b, bins=bins_arr, density=True)[0]

        ent_ab = entropy(a, b)
        ent_ba = entropy(b, a)

        return (ent_ab, ent_ba)

    def __model_classify(self) -> float:
        """ Classification model for A/A test
        Returns:
            float: ROC-AUC score for model
        """
        X = self.dataset[self.config['clustering_cols']]
        target = self.config['target']
        clf = RandomForestClassifier()
        clf.fit(X, target)
        pred = clf.predict_proba(X)
        roc_auc = roc_auc_score(target, pred)

        return roc_auc

    def __alpha_simulation(self, n_iter: int = 10000) -> float:
        """ Perform A/A test

        Args:
            n_iter (int): Number of iterations.

        Returns:
            float: Actual alpha (= share of iterations when control and treatment groups are equal).
        """
        result: int = 0
        for it in range(n_iter):
            if self.config['metric_type'] == 'solid':
                _, pvalue = ks_2samp(self.config['control'], self.config['treatment'])
                if pvalue >= self.config['alpha']:
                    result += 1
            elif self.config['metric_type'] == 'ratio':
                num_control, num_treatment = self.dataset.loc[self.dataset[self.config['group_col']] == 'A',
                                                              self.config['numerator']].to_numpy(), \
                                            self.dataset.loc[self.dataset[self.config['group_col']] == 'B',
                                                             self.config['numerator']].to_numpy()
                _, num_pvalue = ks_2samp(num_control, num_treatment)

                den_control, den_treatment = self.dataset.loc[self.dataset[self.config['group_col']] == 'A', self.config['denominator']].to_numpy(), \
                                             self.dataset.loc[self.dataset[self.config['group_col']] == 'B', self.config['denominator']].to_numpy()
                _, den_pvalue = ks_2samp(den_control, den_treatment)

                if num_pvalue >= self.config['alpha'] and den_pvalue >= self.config['alpha']:
                    result += 1

        result /= n_iter

        return result

    def aa_test(self):
        """Performs A/A test
        """
        if self.config['to_cluster']:
            self.__clustering()

        if self.config['aa_method'] == 'alpha-simulation':
            actual_alpha = self.__alpha_simulation()
            return actual_alpha
        elif self.config['aa_method'] == 'kl-divergence':
            kl_divs = self.__kl_divergence(50)
            return kl_divs
        elif self.config['aa_method'] == 'model-classify':
            roc_auc = self.__model_classify()
            return roc_auc

    def fit(self) -> None:
        """ Split DataFrame and add group column based on splitting

        Returns:
            DataFrame with additional 'group' column
        """
        return self.splitter(self.config['split_rate'])
