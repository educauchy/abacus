import os
import sys
import logging
import pandas as pd
import numpy as np
import yaml

from auto_ab.stratification.params import SplitBuilderParams
from auto_ab.prepilot.params import PrepilotParams
from auto_ab.prepilot.prepilot_experiment_builder import PrepilotExperimentBuilder
from auto_ab.prepilot.prepilot_split_builder import PrepilotSplitBuilder
from auto_ab.auto_ab.abtest import ABTest
from auto_ab.auto_ab.params import ABTestParams
from auto_ab.auto_ab.params import *

POSSIBLE_TESTS = [ABTest.test_hypothesis_boot_confint, 
                ABTest.test_hypothesis_boot_est,
                ABTest.test_hypothesis_strat_confint,
                ABTest.test_hypothesis_mannwhitney,
                ABTest.test_hypothesis_ttest,
                ABTest.delta_method,
                ABTest.taylor_method,
                ABTest.ratio_bootstrap
            ]


if __name__=="__main__":

    df = pd.read_csv('./notebooks/ab_data.csv')

    with open("./auto_ab/configs/auto_ab.config.yaml", "r") as stream:
        try:
            ab_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    df["moda_city"] = np.random.randint(1, 5, df.shape[0])
    df["moda_city"] = df["moda_city"].astype(str)
    df["country"] = np.random.randint(1, 3, df.shape[0])
    df["id"] = df.index

    df["numerator"] = np.random.randint(1, 5, df.shape[0])
    df["denominator"] = np.random.randint(1, 5, df.shape[0])
    df["country"] = np.random.randint(1, 3, df.shape[0])

    data_params = DataParams(**ab_config['data_params'])
    hypothesis_params = HypothesisParams(**ab_config['hypothesis_params'])

    ab_params = ABTestParams(data_params,hypothesis_params)

    ab_params = ABTestParams()
    ab_params.data_params.numerator = 'numerator'
    ab_params.data_params.denominator = 'denominator'

    split_builder_params = SplitBuilderParams(
        map_group_names_to_sizes={
            'control': None,
            'target': None
        },
        region_col = "moda_city",
        split_metric_col = "height_now",
        customer_col = "id",
        cols = [],
        cat_cols=[
        ],
        pvalue=0.05,
        n_top_cat=100,
        stat_test="ttest_ind"
    )

    ab_params.hypothesis_params.n_boot_samples = 2

    for test in POSSIBLE_TESTS:
        print(test)
        prepilot_params = PrepilotParams(
            metrics_names=['height_now'],
            injects=[1.0006,1.0005,1.0004,1.0003],
            min_group_size=50000, 
            max_group_size=52000, 
            step=10000,
            variance_reduction = None,
            use_buckets = False,
            stat_test = test,
            iterations_number = 3,
            max_beta_score=2.0,
            min_beta_score=0.0,
        )

        prepilot = PrepilotExperimentBuilder(df, ab_params,
                                        prepilot_params,
                                        split_builder_params
                                        )
        beta,alpha = prepilot.collect()
        
                            
