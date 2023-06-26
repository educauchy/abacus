import yaml
import pandas as pd
import pytest

from abacus.auto_ab.abtest import ABTest
from abacus.auto_ab.params import ABTestParams, DataParams, HypothesisParams


@pytest.mark.smoke
def test_simple_continuous_experiment_creation_default_params():
    pass

@pytest.mark.smoke
def test_simple_continuous_experiment_creation_default_params():
    """
    Test workability of simple experiment creation with default params
    """
    with open('config/simple_continuous_experiment_creation_default_params.yaml', 'r') as file:
        config = yaml.safe_load(file)

    df = pd.read_csv('./data/ab_data_new.csv')
    data_params = DataParams(**config['data_params'])
    hypothesis_params = HypothesisParams(**config['hypothesis_params'])
    ab_params = ABTestParams(data_params, hypothesis_params)

    ab_test = ABTest(df, ab_params)

@pytest.mark.smoke
def test_simple_continuous_experiment_creation():
    """
    Test workability of simple experiment creation
    """
    with open('config/simple_continuous_experiment_creation.yaml', 'r') as file:
        config = yaml.safe_load(file)

    df = pd.read_csv('./data/ab_data_new.csv')
    data_params = DataParams(**config['data_params'])
    hypothesis_params = HypothesisParams(**config['hypothesis_params'])
    ab_params = ABTestParams(data_params, hypothesis_params)

    ab_test = ABTest(df, ab_params)

@pytest.mark.smoke
def test_simple_binary_experiment_creation():
    """
    Test workability of simple experiment creation
    """
    with open('config/simple_binary_experiment_creation.yaml', 'r') as file:
        config = yaml.safe_load(file)

    df = pd.read_csv('./data/ab_data_new.csv')
    data_params = DataParams(**config['data_params'])
    hypothesis_params = HypothesisParams(**config['hypothesis_params'])
    ab_params = ABTestParams(data_params, hypothesis_params)

    ab_test = ABTest(df, ab_params)

@pytest.mark.smoke
def test_simple_ratio_experiment_creation():
    """
    Test workability of simple experiment creation
    """
    with open('config/simple_ratio_experiment_creation.yaml', 'r') as file:
        config = yaml.safe_load(file)

    df = pd.read_csv('./data/ab_data_new.csv')
    data_params = DataParams(**config['data_params'])
    hypothesis_params = HypothesisParams(**config['hypothesis_params'])
    ab_params = ABTestParams(data_params, hypothesis_params)

    ab_test = ABTest(df, ab_params)
