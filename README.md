# A/B framework for fast hypothesis testing

## Config 

Example of config is presented below
```yaml
data_params:
  n_rows: 500
  path: '../notebooks/ab_data_new.csv'
  id_col: 'id'
  group_col: 'groups'
  target: 'height_now'
  target_flg: 'bought'
  predictors: ['weight_now']
  numerator: 'clicks'
  denominator: 'sessions'
  covariate: 'height_prev'
  target_prev: 'height_prev'
  predictors_prev: ['weight_prev']
  cluster_col: 'kl-divergence'
  clustering_cols: ['col1', 'col2', 'col3']
  is_grouped: True
simulation_params:
  n_iter: 100
  split_rates: [0.1, 0.2, 0.3, 0.4, 0.5]
  vars: [0, 1, 2, 3, 4, 5]
  extra_params: []
hypothesis_params:
  alpha: 0.05
  beta: 0.2
  alternative: 'two-sided' # less, greater, two-sided
  split_ratios: [0.5, 0.5]
  strategy: 'simple_test'
  strata: 'country'
  strata_weights:
    'US': 0.8
    'UK': 0.2
  n_boot_samples: 200
  n_buckets: 50
  metric_type: 'solid'
  metric_name: 'mean'
result_params:
  to_csv: True
  csv_path: '/app/data/internal/guide/solid_mde.csv'
splitter_params:
  split_rate: 0.5
  name: 'default'
bootstrap_params:
  metric: 'mean'
  num_iterations: 200
aatest_params:
  alpha: 0.05
  method: 'kl-divergence'
  to_cluster: True
  n_clusters: 50
```

### Description of config

| Section | Property | Type | Description | Default | Possible values | Comment |
|-------|----------|--------|-------------|---------|-----------------|------------|
|data_params|n_rows|integer|Number of rows to read from dataframe|-1|-1 for all rows or any positive integer value|-|
|data_params|path|string|Path to the dataframe|-|-|Currently not used|
|data_params|id_col|string|Name of id column|id|-|-|-|
|data_params|group_col|string|Name of group label column|-|-|-|
|data_params|target|string|Name of continuous target column|-|-|-|
|data_params|target_flg|string|Name of binary target column|-|-|-|
|data_params|predictors|list of strings|Name of columns used for covariate prediction in CUPAC|-|-|-|
|data_params|numerator|string|Name of numerator column for ratio metric|-|-|-|
|data_params|denominator|string|Name of denominator column for ratio metric|-|-|-|
|data_params|covariate|string|Name of covariate column|-|-|-|
|data_params|target_prev|string|Name of continuous target column for pre-experiment period for covariate prediction in CUPAC|-|-|-|
|data_params|predictors_prev|list of strings|Name of columns for pre-experiment period for covariate prediction in CUPAC|-|-|-|
|data_params|cluster_col|string|Name of column with cluster id|-|-|-|
|data_params|clustering_cols|list of strings|Name of columns which will be used for clustering|-|-|-|
|data_params|is_grouped|boolean|Whether of not data is grouped by user|True|-|-|
|simulation_params|n_iter|integer|Number of iterations in simulation|1000|-|Currently not used|
|simulation_params|split_rates|list of floats|Range of split rates to test|[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]|-|Currently not used|
|simulation_params|vars|list of integers/floats|Range of standard deviations to test|[1, 2, 3, 4, 5]|-|Currently not used|
|simulation_params|extra_params|list|Extra parameters|-|-|Currently not used|
|hypothesis_params|alpha|boolean|Type I error|0.05|-|-|
|hypothesis_params|beta|string|Type II error|0.2|-|-|
|hypothesis_params|alternative|string|Direction of hypothesis|two-sided|less, greater, two-sided|-|-|
|hypothesis_params|split_ratios|list of ratios|Ratios for each variation|[0.5, 0.5]|-|Currently not used|
|hypothesis_params|strategy|string|Strategy of hypothesis testing|simple_test|-|Currently not used|
|hypothesis_params|strata|string|Name of strata column|-|-|Currently not used|
|hypothesis_params|strata_weights|dict: string-float|Weights for each strata|-|-|Currently not used|
|hypothesis_params|n_boot_samples|integer|Number of bootstrap samples|1000|-|Currently not used|
|hypothesis_params|n_buckets|integer|Number of buckets|100|-|-|
|hypothesis_params|metric_type|string|Type of metric for experiment|solid|solid, binary, ratio|-|
|hypothesis_params|metric_name|string|Name of metric for experiment|mean|mean, median, custom|-|
|result_params|to_csv|boolean|Whether or not to save results to csv|False|-|Current not used|
|result_params|csv_path|string|Path to csv file|-|-|Currently not used|
|splitter_params|split_rate|float|Share of control group|0.5|-|Currently not used|
|splitter_params|name|string|Name of splitter|default|-|Currently not used|
|bootstrap_params|metric|string|Name of bootstrap metric|-|-|-|
|bootstrap_params|num_iterations|integer|Number of bootstrap samples|-|-|-|
|aatest_params|alpha|float|Type I error|0.05|-|Currently not used|
|aatest_params|method|string|Name of A/A test to use|kl-divergence|-|Currently not used|
|aatest_params|to_cluster|boolean|Whether of not to use clustering before A/A test|True|-|Currently not used|
|aatest_params|n_clusters|integer|Number of clusters|50|-|Currently not used|
