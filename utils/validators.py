from typing import Set
from pyspark.sql import DataFrame
from utils.errors import AbsentColsError


def validate_df_cols(df: DataFrame, required_cols: Set[str]) -> DataFrame:
    absent_required_cols = required_cols - set(df.columns)
    is_absent_required_cols = (len(absent_required_cols) > 0)
    if is_absent_required_cols:
        raise AbsentColsError(f"Required columns {absent_required_cols} missed in df")
    return df
