from __future__ import annotations
from abc import ABC
from typing import List, Optional, Dict, Any, ClassVar
import datetime
from pydantic.dataclasses import dataclass
from pydantic import validator, root_validator


@dataclass
class PostAnalysisParams:
    test_experiment_ids: Optional[List[str]]
    days_recalculation_experiments: Optional[int]
    fin_params: FinParams
    database_params: DatabaseParams
    preperiod_stat_test_params: PreperiodStatTestParams
    period_stat_test_params: PeriodStatTestParams
    fraud_filter_params: FraudFilterParams
    communication_costs_periods: Dict[datetime.date, CommunicationCosts]
    addition_experiment_ids: Optional[List[str]]

    @classmethod
    def load(cls, raw_params: Dict[str, Any]) -> PostAnalysisParams:
        test_experiment_ids = raw_params["test_experiment_ids"]
        days_recalculation_experiments = raw_params["days_recalculation_experiments"]
        fin_params = FinParams(**raw_params["fin_params"])
        database_params = DatabaseParams(**raw_params["database_params"])
        preperiod_stat_test_params = PreperiodStatTestParams(**raw_params["preperiod_stat_test_params"])
        period_stat_test_params = PeriodStatTestParams(**raw_params["period_stat_test_params"])
        fraud_filter_params = FraudFilterParams(**raw_params["fraud_filter_params"])
        communication_costs_periods = (
            CommunicationCosts.get_map_periods_to_costs(raw_params["communication_costs_periods"])
        )
        addition_experiment_ids = raw_params["addition_experiment_ids"]
        return cls(test_experiment_ids,
                   days_recalculation_experiments,
                   fin_params,
                   database_params,
                   preperiod_stat_test_params,
                   period_stat_test_params,
                   fraud_filter_params,
                   communication_costs_periods,
                   addition_experiment_ids)


@dataclass
class FinParams:
    points_redemption_rate: float
    opex: float

    @root_validator
    @classmethod
    def values_bounds_validator(cls, values):
        for k, v in values.items():
            assert 1 > v > 0, f"{k}"
        return values


@dataclass
class DatabaseParams:
    hive_guests_metrics_table: str
    postgre_period_effects_table: str
    postgre_preperiod_tests_table: str
    test_hive_guests_metrics_table: str
    test_postgre_period_effects_table: str
    test_postgre_preperiod_tests_table: str
    postgre_input_db_uri: str
    postgre_output_db_uri: str


class StatTestParams(ABC):
    pass


@dataclass
class PreperiodStatTestParams(StatTestParams):
    ttest_ind: ClassVar[str] = "ttest_ind"
    mannwhitneyu: ClassVar[str] = "mannwhitneyu"
    chisquare: ClassVar[str] = "chisquare"
    map_test_to_metrics: Dict[str, List[str]]

    check: ClassVar[str] = "check"
    cat_plu: ClassVar[str] = "cat_plu"
    map_experiment_types_to_divergence_metrics: Dict[str, List[str]]

    pvalue: float

    @validator("pvalue")
    @classmethod
    def pvalue_validator(cls, pvalue: float):
        assert 1 > pvalue > 0
        return pvalue

    @validator("map_test_to_metrics")
    @classmethod
    def map_test_to_metrics_validator(cls,
                                      map_test_to_metrics: Dict[str, List[str]]):
        acceptable_test_names = (cls.ttest_ind, cls.mannwhitneyu, cls.chisquare)
        for test_name in map_test_to_metrics.keys():
            assert test_name in acceptable_test_names, \
                f"Acceptable tests {acceptable_test_names}"
        return map_test_to_metrics

    @validator("map_experiment_types_to_divergence_metrics")
    @classmethod
    def map_experiment_types_to_divergence_metrics_validator(cls,
                                                             map_experiment_types_to_divergence_metrics: Dict[str, List[str]]):
        acceptable_experiment_types = (cls.check, cls.cat_plu)
        for experiment_type in map_experiment_types_to_divergence_metrics.keys():
            assert experiment_type in acceptable_experiment_types, \
                f"Acceptable experiment types {map_experiment_types_to_divergence_metrics}"
        return map_experiment_types_to_divergence_metrics


@dataclass
class PeriodStatTestParams(StatTestParams):
    confidence_level: float
    bootstrap_iterations_number: int

    @validator("confidence_level")
    @classmethod
    def confidence_level_validator(cls, confidence_level: float):
        assert 1 > confidence_level > 0
        return confidence_level


@dataclass
class FraudFilterParams:
    max_rto_per_week: float
    max_count_visits_per_week: float  # = count checks
    min_fm_share_in_rto: float


@dataclass
class CommunicationCosts:
    sms: float
    viber: float
    email: float

    @classmethod
    def get_map_periods_to_costs(cls, raw_map_period_to_costs: Dict[datetime.date, Dict[str, float]]) \
        -> Dict[datetime.date, CommunicationCosts]:
        return {
            period: cls(**raw_costs)
            for period, raw_costs in raw_map_period_to_costs.items()
        }
