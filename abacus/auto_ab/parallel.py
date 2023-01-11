import numpy as np
import pandas as pd
from typing import List, Any, Union, Optional
import hashlib
import secrets


class ParallelExperiments:
    def __init__(self, dataset: pd.DataFrame, id_col: str = 'id', method: str = 'hashing', n_buckets: int = 200):
        self.dataset = dataset
        self.id_col = id_col
        self.method = method
        self.n_buckets = n_buckets

    def _modulo(self):
        """ Creates buckets using module approach

        Returns:
            None
        """
        self.dataset['bucket_id'] = np.remainder( self.dataset[self.id_col].to_numpy(),
                                                  self.n_buckets)

    def _hashing(self, hash_func: Optional[str] = None, salt: Optional[str] = None):
        """ Creates buckets using hash function approach

        Args:
            hash_func: Hash function
            salt: Salt string for the experiment

        Returns:
            None
        """
        if salt is None:
            salt = secrets.token_hex(8)
        salt: bytes = bytes(salt, 'utf-8')

        if hash_func is not None:
            pass
        else:
            hasher = hashlib.blake2b(salt=salt)

        bucket_ids: np.array = np.array([])
        ids: np.array = self.dataset[self.id_col].to_numpy()
        for id in ids:
            hasher.update( bytes(str(id), 'utf-8') )
            bucket_id = int(hasher.hexdigest(), 16) % self.dataset.n_buckets
            bucket_ids = np.append(bucket_ids, bucket_id)

        self.dataset.loc[:, 'bucket_id'] = bucket_ids
        self.dataset = self.dataset.astype({'bucket_id': 'int32'})
