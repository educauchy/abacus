import numpy as np
import pandas as pd
import hashlib, pprint
import yaml
from typing import List, Tuple, Dict, Union, Callable, Optional, Any
from collections import Counter, defaultdict
from scipy.stats import ks_2samp, entropy
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
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
        """
        Add 'group' column
        :param split_rate: Split rate of control/treatment
        :return: None
        """
        split_rate: float = self.params.splitter_params.split_rate if split_rate is None else split_rate
        self.dataset = self.config['splitter'].fit(self.dataset,
                                                   self.params.data_params.target,
                                                   self.params.data_params.numerator,
                                                   self.params.data_params.denominator,
                                                   split_rate)

    def __default_splitter(self, split_rate: float = 0.5):
        A_data, B_data = train_test_split(self.dataset, train_size=split_rate, random_state=0)
        A_data.loc[:, self.config['group_col']] = 'A'
        B_data.loc[:, self.config['group_col']] = 'B'
        Z = pd.concat([A_data, B_data]).reset_index(drop=True)
        return Z

    def __clustering(self) -> None:
        """
        Clustering for dataset
        """
        X = self.dataset.copy()
        kmeans = KMeans(n_clusters=self.config['n_clusters'])
        kmeans.fit(X)
        self.dataset.loc[self.config['cluster_col']] = kmeans.predict(X)

    def __kl_divergence(self, n_bins: int = 50) -> Tuple[float, float]:
        """
        Kullback-Leibler divergence for two arrays of cluster ids for A/A test
        :param n_bins: Number of clusters
        :return: Kullback-Leibler divergence a to b and b to a
        """
        a = self.dataset.loc[self.dataset[self.config['group_col'] == 'A'], self.config['cluster_col']].tolist()
        b = self.dataset.loc[self.dataset[self.config['group_col'] == 'B'], self.config['cluster_col']].tolist()

        bins_arr = list(range(1, n_bins+1))
        a = np.histogram(a, bins=bins_arr, density=True)[0]
        b = np.histogram(b, bins=bins_arr, density=True)[0]

        ent_ab = entropy(a, b)
        ent_ba = entropy(b, a)

        return (ent_ab, ent_ba)

    def __model_classify(self, X: pd.DataFrame = None, target: np.array = None) -> float:
        """
        Classification model for A/A test
        :param X: Dataset to classify
        :param target: Target for model
        :return: ROC-AUC score for model
        """
        X = self.dataset[self.config['clustering_cols']]
        target = self.config['target']
        clf = RandomForestClassifier()
        clf.fit(X, target)
        pred = clf.predict_proba(X)
        roc_auc = roc_auc_score(target, pred)

        return roc_auc

    def __alpha_simulation(self, n_iter: int = 10000) -> float:
        """
        Perform A/A test
        :param n_iter: Number of iterations
        :return: Actual alpha; Share of iterations when control and treatment groups are equal
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
        """
        Split DataFrame and add group column based on splitting
        :param X: Pandas DataFrame to split
        :param split_rate: Split rate of control to treatment
        :return: DataFrame with additional 'group' column
        """
        return self.splitter(self.config['split_rate'])

    def create_level(self, X: pd.DataFrame, id_column: str = '', salt: Union[str, int] = '',
                     n_buckets: int = 100) -> pd.DataFrame:
        """
        Create new levels in split all users into buckets
        :param X: Pandas DataFrame
        :param id_column: User id column name
        :param salt: Salt string for the experiment
        :param n_buckets: Number of buckets for level
        :return: Pandas DataFrame extended by column 'bucket'
        """
        ids: np.array = X[self.config['id_col']].to_numpy()
        salt: str = salt if type(salt) is str else str(int)
        salt: bytes = bytes(salt, 'utf-8')
        hasher = hashlib.blake2b(salt=salt)

        bucket_ids: np.array = np.array([])
        for id in ids:
            hasher.update( bytes(str(id), 'utf-8') )
            bucket_id = int(hasher.hexdigest(), 16) % n_buckets
            bucket_ids = np.append(bucket_ids, bucket_id)

        X.loc[:, 'bucket_id'] = bucket_ids
        X = X.astype({'bucket_id': 'int32'})
        return X


if __name__ == '__main__':
    # Test hash function
    X = pd.DataFrame({
        'id': range(0, 20000)
    })
    sp = Splitter()
    X_new = sp.create_level(X, 'id', 5, 200)
    level = X_new['bucket_id']
    pprint.pprint(Counter(level))

    # Test splitter
    X = pd.DataFrame({
        'sex': ['f' for _ in range(14)] + ['m' for _ in range(6)],
        'married': ['yes' for _ in range(5)] + ['no' for _ in range(9)] + ['yes' for _ in range(4)] + ['no', 'no'],
        'country': [np.random.choice(['UK', 'US'], 1)[0] for _ in range(20)],
    })
    conf = ['sex', 'married']
    stratify_by = ['country']
    X_out = Splitter().fit()


    df = pd.read_csv('../../examples/storage/data/ab_data.csv')

    with open("../../config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
