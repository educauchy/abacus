import pandas as pd
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import pandas_udf
from stratification.params import SplitBuilderParams
from prepilot.prepilot_split_builder import PrepilotSplitBuilder
from prepilot.abstract_experiment_builder import AbstractExperimentBuilder
from prepilot.params import PrepilotParams
from analysis.stat_test import PeriodStatTest


_ERROR_RES_SCHEMA = T.StructType([T.StructField("group_sizes", T.StringType(), False),
                    T.StructField("metric", T.StringType(), False),
                    T.StructField("MDE", T.FloatType(), False),
                    T.StructField("is_effect_found", T.IntegerType(), False)
                    ])

class PrepilotExperimentBuilder(AbstractExperimentBuilder):
    """Calculates I and II type errors for different group sizes and injects
    """

    def __init__(self,
                 spark,
                 guests: DataFrame,
                 experiment_params: PrepilotParams,
                 stratification_params: SplitBuilderParams):
        """
        Args:
            guests: dataframe that collected by PrepilotGuestsCollector
            experiment_params: prameters for prepilot experiments

        """
        super().__init__(spark, 
                         guests,
                         experiment_params)
        self.stratification_params = stratification_params
        self._number_of_decimals = 10

    def _calc_experiment(self,
                         guests_with_splits,
                         res_schema,
                         metric_name,
                         inject=1.0):

        def calc_split_experiment_generator(stat_test_params, 
                                            metric_name, 
                                            inject=1.0):

            @pandas_udf(res_schema, F.PandasUDFType.GROUPED_MAP)
            def calc_split_experiment_pd(guests_with_splits):
                """Calculates stat test for one experiment grid element

                Args:
                    guests_with_splits: dataframe with calculated splits for experiment

                Returns: pandas DataFrame with calculated stat test and experiment parameters

                """
                group_sizes = guests_with_splits.split_group_sizes.values[0]
                control = guests_with_splits[guests_with_splits["is_control"] == 1][metric_name].values
                target = guests_with_splits[guests_with_splits["is_control"] == 0][metric_name].values * inject
                stat_test = PeriodStatTest(target, control, "", stat_test_params)
                stat_result = stat_test.calculate_period_effect()
                return pd.DataFrame({"group_sizes": [group_sizes],
                                    "metric": [metric_name],
                                    "MDE": [inject],
                                    "is_effect_found": stat_result["effect__significance"]
                                    })

            return calc_split_experiment_pd

        calculate_experiment = calc_split_experiment_generator(self.stat_test_params, metric_name, inject)
        result_df = guests_with_splits.groupBy(["split_group_sizes", "split_number" ]).apply(calculate_experiment)
        return result_df    

    def calc_alpha_error(self,
                         guests_with_splits,
                         res_schema
                         ):

        alpha_error_result = self.spark.createDataFrame([], res_schema)

        for metric_name in self.experiment_params.metrics_names:
            alpha_error_df = self._calc_experiment(guests_with_splits,
                                                   res_schema, 
                                                   metric_name)
            alpha_error_result = alpha_error_result.unionAll(alpha_error_df)
        
        alpha_agg = (alpha_error_result
            .groupBy(["metric", "group_sizes"])
            .agg(F.avg("is_effect_found").alias("alpha_error"))
        ).cache()

        return alpha_agg.groupBy("metric").pivot("group_sizes").sum("alpha_error")

    def calc_beta_error(self,
                        guests_with_splits,
                        res_schema
                        ):

        beta_error_result = self.spark.createDataFrame([], res_schema)
        for inject in self.experiment_params.injects:
            for metric_name in self.experiment_params.metrics_names:
                beta_error_df = self._calc_experiment(guests_with_splits,
                                                      res_schema,
                                                      metric_name,
                                                      inject)
                beta_error_result = beta_error_result.unionAll(beta_error_df)
        
        beta_agg = (beta_error_result
            .groupBy(["metric", "group_sizes", "MDE"])
            .agg((1.0 - F.avg("is_effect_found")).alias("beta_error"))
        ).cache()

        return beta_agg.groupBy(["metric", "MDE"]).pivot("group_sizes").sum("beta_error")


    def collect(self) -> pd.DataFrame:
        """Calculates I and II types error using prepilot parameters.

        Args:
            stratification_params: params for stratification
            full: if True function will return full dataframe with results.
            Otherwise will be returned only max calculated MDE for each size.

        Returns: pandas DataFrames with aggregated results of experiment.

        """
        prepilot_split_builder = PrepilotSplitBuilder(self.spark, 
                                                        self.guests,
                                                        self.group_sizes,
                                                        self.stratification_params,
                                                        self.experiment_params.iterations_number)

        prepilot_guests = prepilot_split_builder.collect()
        prepilot_guests = (prepilot_guests.join(self.guests, 
                                                on=self.stratification_params.customer_col,
                                                how="inner")
                        .select(prepilot_guests.columns+self.experiment_params.metrics_names)
        )

        beta = self.calc_beta_error(prepilot_guests, _ERROR_RES_SCHEMA)
        alpha = self.calc_alpha_error(prepilot_guests, _ERROR_RES_SCHEMA)
        return beta, alpha
