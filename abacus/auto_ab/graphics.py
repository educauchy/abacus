import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abacus.auto_ab.params import ABTestParams


class Graphics:
    def __init__(self) -> None:
        pass

    @classmethod
    def plot_simulation_matrix(cls, log_path: str):
        """Plot log of simulation matrix.

        Axes of a matrix: ``split rate`` and ``increment``.
        Cell value: share of significant simulations.

        Args:
            log_path (str): Path to log file in .csv format.
        """
        df = pd.read_csv(log_path)
        df_pivot = df.pivot(index='split_rate', columns='increment', values='pval_sign_share')
        plt.figure(figsize=(15, 8))
        plt.title('Simulation log')
        sns.heatmap(df_pivot, cmap='Greens', annot=True)
        plt.show()
        plt.close()

    @classmethod
    def plot_mean_experiment(cls, params: ABTestParams) -> None:
        """Plot distributions of means in experiment groups.

        Args:
            params (ABTestParams): Parameters of the experiment.
        """
        bins = 100
        a_mean = np.mean(params.data_params.control)
        b_mean = np.mean(params.data_params.treatment)
        threshold = np.quantile(params.data_params.control, 0.975)
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.text(a_mean, 100, 'H0', fontsize='xx-large')
        ax.text(b_mean, 100, 'H1', fontsize='xx-large')
        ax.hist(params.data_params.control, bins, alpha=0.5, label='Control', color='Red')
        ax.hist(params.data_params.treatment, bins, alpha=0.5, label='Treatment', color='Green')
        ax.axvline(x=a_mean, color='Red')
        ax.axvline(x=b_mean, color='Green')
        ax.axvline(x=threshold, color='Blue', label='Critical value')
        ax.legend()
        plt.show()
        plt.close()

    @classmethod
    def plot_median_experiment(cls, params: ABTestParams) -> None:
        """Plot distributions of medians in experiment groups.

        Args:
            params (ABTestParams): Parameters of the experiment.
        """
        bins = 100
        a_median = np.median(params.data_params.control)
        b_median = np.median(params.data_params.treatment)
        threshold = np.quantile(params.data_params.control, 0.975)
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.hist(params.data_params.control, bins, alpha=0.5, label='Control', color='Red')
        ax.hist(params.data_params.treatment, bins, alpha=0.5, label='Treatment', color='Green')
        ax.axvline(x=a_median, color='Red')
        ax.axvline(x=b_median, color='Green')
        ax.axvline(x=threshold, color='Blue', label='Critical value')
        ax.legend(loc='upper right')
        plt.show()
        plt.close()

    @classmethod
    def plot_bootstrap_confint(cls,
                               x: np.array,
                               params: ABTestParams) -> None:
        """Plot bootstrapped metric of an experiment with its confidence
        interval and zero value.

        Args:
            x (np.array): Bootstrap metric.
            params (ABTestParams): Parameters of the experiment.
        """
        bins = 100
        ci_left, ci_right = np.quantile(x, params.hypothesis_params.alpha / 2), \
            np.quantile(x, 1 - params.hypothesis_params.alpha / 2)
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.hist(x, bins, alpha=0.5, label='Differences in metric', color='Red')
        ax.axvline(x=0, color='Red', label='No difference')
        ax.vlines([ci_left, ci_right], ymin=0, ymax=100, linestyle='--', label='Confidence interval')
        ax.legend()
        plt.show()
