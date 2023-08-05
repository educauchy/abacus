# ABacus: fast hypothesis testing and experiment design solution

**ABacus** is a Python library developed for A/B experimentation and testing.
It includes versatile instruments for different experimentation tasks like
prepilot, sample size determination, results calculation, visualisations and reporting.

## Important features

* Experiment design: type I and II errors, effect size, sample size simulations.
* Groups splitting with flexible configuration and stratification.
* A/A test and evaluation of splitter accuracy.
* Evaluation of experiment results with various statistical tests and approaches.
* Sensitivity increasing techniques like stratification, CUPED and CUPAC.
* Visualisation of experiment.
* Reporting in a human-readable format.

## Installation

You can use **pip** to install **ABacus** and use it for your projects:
```shell
pip install kolmogorov-abacus
```

Note the requirement of Python 3.11+.

## Quick example

To define an experiment and analyse it is as easy as to describe your experiment and data:
```shell
from abacus.auto_ab.abtest import ABTest
from abacus.auto_ab.params import ABTestParams, DataParams, HypothesisParams

data_params = DataParams(...)
hypothesis_params = HypothesisParams(...)
ab_params = ABTestParams(data_params, hypothesis_params)

data = pd.read_csv('abtest_data.csv')

ab_test = ABTest(data, ab_params)

ab_test.report()
```

The result of code execution is the following:

[//]: # (![Experiment report!]&#40;docs/source/_static/report_example.png&#41;)
<img src="docs/source/_static/report_example.png" width="400">

## Documentation and Examples

Detailed [documentation](https://github.com/educauchy/abacus/tree/main/examples>) and [examples](https://github.com/educauchy/abacus/tree/main/examples>) are available for your usage.

## Communication

Authors and developers:
* [Vadim Glukhov](https://github.com/educauchy)
* [Egor Shishkovets](https://github.com/egorshishkovets)
* [Dmitry Zabavin](https://github.com/dmitryzabavin)
