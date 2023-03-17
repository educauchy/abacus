Usage
=====

------------
Installation
------------

Stable version is released on every tag to ``main`` branch.

.. code:: bash

    pip install abacus

**Note**: ABacus requires Python 3.7+.


--------
Overview
--------

There are three core elements in ABacus:

* Experiment evaluation
* Splitter
* Experiment design


^^^^^^^^^^^^^^^^^^^^^
Experiment evaluation
^^^^^^^^^^^^^^^^^^^^^

Experiment evaluation depends on parameters of experiment and data generated during this experiment.

Parameters of experiment include but not limited to:

- metric (e.g. mean, median, percentile);
- type I and II errors;
- direction of hypothesis.

Data can have the following characteristics:

- target column (or two columns in case of a ratio metric);
- stratification columns;
- id column;
- covariate columns;
- group column.

""""""""""""""""""""""
Statistical approaches
""""""""""""""""""""""

**ABacus** supports three types of metrics: continuous, binary, and ratio.
Each of these types requires its own particular methods to conduct statistical analysis of experiment.

**ABacus** has the following statistical tests for each type of metric:

1. For continuous metrics: Welch t-test, Mann-Whitney U-test, bootstrap (3 implementations).
2. For binary metrics: chi-squared test, Z-test, bootstrap.
3. For ratio metrics: delta method and Taylor method.

There are other options that can help you during the experiment evaluation:

- Bucketing: aggregate your observations in order to get fewer observations and distribution of means.
- Linearization: replace the initial ratio metric with its linearized version that is on per-user level, co-directed and i.i.d.

""""""""""""""""""""""
Increasing Sensitivity
""""""""""""""""""""""

As you want to make your metrics more sensitive, you will mostly likely want to use some sensitivity increasing techniques.
And **ABacus** help you use them easily as another method calling. It supports the following options for increasing sensitivity of your experiments:

* Outliers removal
* Functional transformations
* Stratification
* CUPED
* CUPAC

All of them are presented in framework, so you can easily use them in your post-experiment analysis.

""""""""""""""
Visualisations
""""""""""""""

A picture is worth a thousand words. No doubt that you want to visually explore your experiment.
And **ABacus** can help you with that.

You can plot experiments with continuous and binary variables.
Continuous plots illustrates not only distributions of desired targe variable, but also a desired metric of a distribution.
You can also plot a bootstrap distribution of differences if you want to estimate your experiment with bootstrap approach.

Here is the output of `plot()` method on some experiment:

.. image:: docs/source/_static/experiment_plot_example.png
  :target: docs/build/html/usage.html
  :width: 400
  :alt: Experiment plot example

"""""""""
Reporting
"""""""""

As you may wish to get some sort of report with information of your experiment, you can definitely do it with ABacus.

You just need to call method `report()` and get something similar:

.. image:: docs/source/_static/report_example.png
  :width: 400
  :alt: Report example


^^^^^^^^
Splitter
^^^^^^^^

Splitter is a core instrument that allows you to get 'equal' groups for your experiment. Groups of an experiment are
equal in the sense of users' desired characteristic of experiment are equal.

It is a crucial part of any experiment design - to get approximately equal groups.
Splitter in **ABacus** not only allows you to split your observations into groups, but also assesses the quality of this split.

^^^^^^^^^^^^
MDE Explorer
^^^^^^^^^^^^

MDE Explorer makes experimental design in order to get all the information about experiment.
The main purpose of its usage is calculation of samples size needed to detect particular effect size based on type
I and II errors, directionality of hypothesis and other parameters.


