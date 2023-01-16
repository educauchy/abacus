Auto A/B
========

.. currentmodule:: abacus.auto_ab

ABTest
------

.. autoclass:: abacus.auto_ab.ABTest
..
    .. automethod:: bucketing, cupac, cuped, delta_method, linearization, plot, ratio_bootstrap,
        resplit_df, test_boot_hypothesis, test_chisquare, test_hypothesis_ttest, test_hypothesis_mannwhitney,
        test_hypothesis_boot

.. autosummary::
    :nosignatures:

    ABTest
    ABTest.bucketing
    ABTest.cupac
    ABTest.cuped
    ABTest.delta_method
    ABTest.linearization
    ABTest.plot
    ABTest.ratio_bootstrap
    ABTest.resplit_df
    ABTest.test_boot_confint
    ABTest.test_boot_fp
    ABTest.test_boot_welch
    ABTest.test_buckets
    ABTest.test_chisquare
    ABTest.test_mannwhitney
    ABTest.test_strat_confint
    ABTest.test_welch
    ABTest.test_z_proportions

.. autofunction:: abacus.auto_ab.ABTest.__bucketize
.. autofunction:: abacus.auto_ab.ABTest.__check_required_columns
.. autofunction:: abacus.auto_ab.ABTest.__get_group
.. autofunction:: abacus.auto_ab.ABTest._delta_params
.. autofunction:: abacus.auto_ab.ABTest._linearize
.. autofunction:: abacus.auto_ab.ABTest._manual_ttest
.. autofunction:: abacus.auto_ab.ABTest._taylor_params

.. autofunction:: abacus.auto_ab.ABTest.bucketing
.. autofunction:: abacus.auto_ab.ABTest.cupac
.. autofunction:: abacus.auto_ab.ABTest.cuped
.. autofunction:: abacus.auto_ab.ABTest.delta_method
.. autofunction:: abacus.auto_ab.ABTest.linearization
.. autofunction:: abacus.auto_ab.ABTest.plot
.. autofunction:: abacus.auto_ab.ABTest.ratio_bootstrap
.. autofunction:: abacus.auto_ab.ABTest.resplit_df
.. autofunction:: abacus.auto_ab.ABTest.test_boot_confint
.. autofunction:: abacus.auto_ab.ABTest.test_boot_fp
.. autofunction:: abacus.auto_ab.ABTest.test_boot_welch
.. autofunction:: abacus.auto_ab.ABTest.test_buckets
.. autofunction:: abacus.auto_ab.ABTest.test_chisquare
.. autofunction:: abacus.auto_ab.ABTest.test_mannwhitney
.. autofunction:: abacus.auto_ab.ABTest.test_strat_confint
.. autofunction:: abacus.auto_ab.ABTest.test_welch
.. autofunction:: abacus.auto_ab.ABTest.test_z_proportions

VarianceReduction
-----------------

.. autoclass:: abacus.auto_ab.VarianceReduction

.. autofunction:: abacus.auto_ab.VarianceReduction._target_encoding
.. autofunction:: abacus.auto_ab.VarianceReduction._predict_target
.. autofunction:: abacus.auto_ab.VarianceReduction.cuped
.. autofunction:: abacus.auto_ab.VarianceReduction.cupac

Graphics
--------

.. autoclass:: abacus.auto_ab.Graphics

.. autofunction:: abacus.auto_ab.Graphics.plot_simulation_matrix
.. autofunction:: abacus.auto_ab.Graphics.plot_median_experiment
.. autofunction:: abacus.auto_ab.Graphics.plot_mean_experiment
.. autofunction:: abacus.auto_ab.Graphics.plot_bootstrap_confint

Params
------

.. autoclass:: abacus.auto_ab.DataParams
.. autoclass:: abacus.auto_ab.HypothesisParams
.. autoclass:: abacus.auto_ab.ABTestParams

ParallelExperiments
-------------------

.. autoclass:: abacus.auto_ab.ParallelExperiments

.. autofunction:: abacus.auto_ab.ParallelExperiments._modulo
.. autofunction:: abacus.auto_ab.ParallelExperiments._hashing

Simulation
----------

.. autoclass:: abacus.auto_ab.Simulation
   :members: _add_increment, set_increment, sample_size_simulation,
        mde_simulation, mde_hyperopt, sample_size, mde

Splitter
--------

.. autoclass:: abacus.auto_ab.Splitter
   :members: config_load, _split_data, __default_splitter,
        __clustering, __kl_divergence, __model_classify, __alpha_simulation,
        aa_test, fit
