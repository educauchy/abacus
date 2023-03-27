Experiment Evaluation
=====================

After the initialization of experiment, we are ready to dive into the analysis.

You have the following options for analysis:

- Statistical Inference
- Metric Transformations
- Increasing Sensitivity (Variance Reduction)
- Visualizations
- Reporting

"""""""""""""""""""""
Statistical Inference
"""""""""""""""""""""

**ABacus** supports three types of metrics: continuous, binary, and ratio.
Each of these types requires its own particular methods to conduct statistical analysis of experiment.

**ABacus** has the following statistical tests for each type of metric:

1. For continuous metrics: ``Welch t-test``, ``Mann-Whitney U-test``, ``bootstrap``.
2. For binary metrics: ``chi-squared test``, ``Z-test``.
3. For ratio metrics: ``delta method``, ``Taylor method``.

To get the result of a test, just call the appropriate statistical method on your ABTest instance:

.. code:: python

    ab_test = ABTest(...)

    ab_test.test_welch()
    {'stat': 5.172, 'p-value': 0.312, 'result': 0}

    # or

    ab_test.test_z_proportions()
    {'stat': 0.12, 'p-value': 0.67, 'result': 0}

As a result, you'll get dictionary with

- statistic of the test,
- p-value of this empirical statistic,
- result in binary form: 0 - H0 is not rejected, 1 - H0 is not accepted.

""""""""""""""""""""""
Metric Transformations
""""""""""""""""""""""

Sometimes experiment data cannot be analyzed directly due to different limitations such as presense of outliers or form of distribution.
Metric transformation techniques available in ABacus are:

- **Outliers removal**: direct exclusion of outliers according to some algorithm. There are two implemented in **ABacus**: remove ``top 5%`` and ``isolation forest``.
- **Functional transformation**: application of any function to your target metric in order to make it more normal or remove outliers.
- **Bucketing**: aggregation of target metric into buckets in order to obtain smaller points for analysis and from initial distribution to distributions of means.
- **Linearization**: remove dependence of observations (and move from ratio target) using linearization approach.

Outliers removal method must be defined in hypothesis params:

.. code:: python

    ab_test = ABTest(...)
    ab_test_2 = ab_test.filter_outliers()

    print(ab_test.params.data_params.control)
    # 200 000

    print(ab_test_2.params.data_params.control)
    # 198 201


The following example includes functional transformation with ``sqrt`` function:

.. code:: python

    ab_test = ABTest(...)
    ab_test_2 = ab_test.metric_transform()

"""""""""""""""""""""""""""""""""""""""""""
Increasing Sensitivity (Variance Reduction)
"""""""""""""""""""""""""""""""""""""""""""

As you want to make your metrics more sensitive, you will mostly likely want to use some sensitivity increasing techniques.
And **ABacus** help you use them easily as another method calling. It supports the following options for increasing sensitivity of your experiments:

* CUPED
* CUPAC
* Stratification

All of them are presented in framework, so you can easily use them in your post-experiment analysis.

""""""""""""""
Visualizations
""""""""""""""

A picture is worth a thousand words. No doubt that you want to visually explore your experiment.

You can plot experiments with continuous and binary variables.
Continuous plots illustrates not only distributions of desired targe variable, but also a desired metric of a distribution.
You can also plot a bootstrap distribution of differences if you want to estimate your experiment with bootstrap approach.

Here is the output of ``ab_test.plot()`` method:

.. image:: ../../../docs/source/_static/experiment_plot_example.png
  :target: docs/build/html/usage.html
  :width: 700
  :alt: Experiment plot example

"""""""""
Reporting
"""""""""

As you may wish to get some sort of report with information of your experiment, you can definitely do it with ABacus.

You just need to call ``ab_test.report()`` and get information about preprocessing steps and results of statistical tests:

.. image:: ../../../docs/source/_static/report_example.png
  :width: 500
  :alt: Report example

Report is available for any metric type. On each metric type, you will get a bit different results.

""""""""""""""""""
Everything at once
""""""""""""""""""

You can freely mix everything you saw above using chaining:




