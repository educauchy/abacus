from typing import List, Optional, Union
import numpy as np
import pandas as pd
import hashlib
import secrets
from abacus.types import DataFrameType


class ParallelExperiments:
    def __init__(self,
                 dataset: DataFrameType,
                 id_col: str = 'id',
                 method: str = 'hashing',
                 n_buckets: int = 200
                 ) -> None:
        self.dataset = dataset
        self.id_col = id_col
        self.method = method
        self.n_buckets = n_buckets

    def _modulo(self) -> None:
        """ Creates buckets using modulo approach.
        """
        self.dataset['bucket_id'] = np.remainder(self.dataset[self.id_col].to_numpy(),
                                                 self.n_buckets)

    def _hashing(self,
                 salt: str,
                 hash_func: Optional[str] = 'blake2b'
                 ) -> None:
        """ Creates buckets using hash function approach.

        Algorithm:

        1. Create hash function with predefined salt.
        2. Hash every value of initial array.
        3. Consider number as hex16 format.
        4. Calculate bucket id as modulo of hex16 number divided by number of buckets.

        Args:
            hash_func (str, optional): Hash function.
            salt (str, optional): Salt string for the experiment.
        """
        if salt is None:
            salt = secrets.token_hex(8)
        salt: bytes = bytes(salt, 'utf-8')

        if hash_func == 'blake2b':
            hasher = hashlib.blake2b(salt=salt)
        elif hash_func == 'blake2s':
            hasher = hashlib.blake2s(salt=salt)
        else:
            hasher = hashlib.blake2b(salt=salt)

        bucket_ids: List[int] = []
        ids: List[Union[float, int]] = self.dataset[self.id_col].tolist()
        for id_ in ids:
            hasher.update(bytes(str(id_), 'utf-8'))
            bucket_id: int = int(int(hasher.hexdigest(), 16) % self.dataset.n_buckets)
            bucket_ids.append(bucket_id)

        self.dataset.loc[:, 'bucket_id'] = bucket_ids
        self.dataset = self.dataset.astype({'bucket_id': 'int32'})
