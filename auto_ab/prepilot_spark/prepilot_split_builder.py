from typing import List
import copy
import itertools
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import pandas_udf
from auto_ab.stratification.params import SplitBuilderParams
from auto_ab.stratification.split_builder import build_split, prepare_cat_data, assign_strata
from auto_ab.prepilot_spark.experiment_structures import BaseSplitElement


class PrepilotSplitBuilder():
    """Columns with splits and injects will be added
    """
    def __init__(self,
                 spark,
                 guests,
                 group_sizes: List[int],
                 stratification_params: SplitBuilderParams,
                 iterations_number: int = 10):
        """There is class for calculation columns with injetcs and target/control splits

        Args:
            guests: dataframe with data for calculations injects and splits
            metrics_names: list of metrics for which will be calculate injects columns
            group_sizes: list of group sizes for split building
            stratification_params: stratification parameters
            iterations_number: number of columns that will be build for each group size

        """
        self.spark = spark
        self.guests = guests.withColumn("partion", F.lit(1))
        self.iterations_number = iterations_number
        self.group_sizes = group_sizes
        self.stratification_params = copy.deepcopy(stratification_params)
        self.split_grid = self._build_splits_grid()

    def _build_splits_grid(self):
        return list(BaseSplitElement(el[0], el[1])
                    for el in itertools.product(self.group_sizes, np.arange(1, self.iterations_number+1)))
    

    def apply_strata(self):

        def strata_generator(df, stratification_params):
            schema = (copy.deepcopy(df.schema)
                    .add(T.StructField("strata", T.StringType(), False))
            )

            @pandas_udf(schema, F.PandasUDFType.GROUPED_MAP)
            def build_strata_pd(df: pd.DataFrame):

                guests_data = prepare_cat_data(df, stratification_params)
                guests_data_with_strata = assign_strata(guests_data.reset_index(drop=True), stratification_params)

                return guests_data_with_strata

            return build_strata_pd

        strata = strata_generator(self.guests, self.stratification_params)
        strata_output = self.guests.groupBy("partion").apply(strata)

        return strata_output


    def build_split_df(self, 
                    guests_with_strata,
                    split: BaseSplitElement):

        def build_split_generator(stratification_params, 
                            split: BaseSplitElement):

            schema = T.StructType([T.StructField("split_group_sizes", T.StringType(), False),
                                T.StructField("split_number", T.IntegerType(), False),
                                T.StructField(stratification_params.customer_col, T.LongType(), False),
                                T.StructField("is_control", T.IntegerType(), False)])

            @pandas_udf(schema, F.PandasUDFType.GROUPED_MAP)
            def build_split_pd(df: pd.DataFrame):
                map_group_names_to_sizes={
                    "control": split.control_group_size,
                    "target": split.target_group_size
                }

                stratification_params.map_group_names_to_sizes = map_group_names_to_sizes

                guests_groups = build_split(df, stratification_params)
                guests_groups = guests_groups.join(
                                pd.get_dummies(guests_groups["group_name"])
                                .add_prefix("is_")
                )
                
                guests_groups["split_group_sizes"] = f"{split.control_group_size}_{split.target_group_size}"
                guests_groups["split_number"] = split.split_number
                            
                result = pd.DataFrame({"split_group_sizes": guests_groups["split_group_sizes"],
                                       "split_number": guests_groups["split_number"],
                                       stratification_params.customer_col: guests_groups[stratification_params.customer_col],
                                       "is_control": guests_groups["is_control"]
                                     })

                return result
            return build_split_pd

        split = build_split_generator(self.stratification_params,
                                      split)
        split = guests_with_strata.groupBy("partion").apply(split)

        return split

    def collect(self):
        """Calculate multiple split with stratification

        Returns: DataFrame with split column

        """
        schema = T.StructType([T.StructField("split_group_sizes", T.StringType(), False),
                            T.StructField("split_number", T.IntegerType(), False),
                            T.StructField(self.stratification_params.customer_col, T.LongType(), False),
                            T.StructField("is_control", T.IntegerType(), False)])

        result = self.spark.createDataFrame([], schema)
        data_with_strata = self.apply_strata()

        for split_param in self.split_grid:
            split_data = self.build_split_df(data_with_strata, split_param)
            result = result.unionAll(split_data)
        
        return result
