import logging
import pandas as pd
import hdbscan
from sklearn.preprocessing import robust_scale
from auto_ab.stratification.params import SplitBuilderParams

log = logging.getLogger(__name__)


def binnarize(df: pd.DataFrame, params: SplitBuilderParams) -> pd.DataFrame:
    lst = []
    for region in list(df[params.region_col].unique()):
        dfr = df[df[params.region_col] == region]

        # check size of selection by region to skip unreasonable split
        if len(dfr) >= params.bin_min_size * params.n_bins_rto:
            # construct rto bins
            if len(dfr[params.split_metric_col].unique()) > params.n_bins_rto:
                labels = pd.qcut(
                    dfr[params.split_metric_col], params.n_bins_rto, labels=False, duplicates="drop"
                ).astype(str)
            else:
                labels = dfr[params.split_metric_col].astype(int).astype(str)

            # construct extra columns bins
            for label in list(labels.unique()):
                res = bin_with_clustering(dfr[labels == label], region, label, params)
                lst.append(res)
        else:
            res = (dfr[[params.region_col, params.split_metric_col]]
                   .rename(columns={params.split_metric_col: f"{params.split_metric_col}_bin"})
            )
            res["cls"] = -1
            res["label"] = "outlier"
            res = res.astype(str)
            lst.append(res)

    stratas = pd.concat(lst, axis=0)
    stratas_wo_outliers = stratas.query("cls != '-1'")
    n_outliers = stratas.shape[0] - stratas_wo_outliers.shape[0]
    if n_outliers > 0:
        log.info(f"{n_outliers} outliers found")
    return stratas["label"]


def bin_with_clustering(
    df_region_labeled: pd.DataFrame, region: str, label: str, params: SplitBuilderParams
    ) -> pd.DataFrame:

    try:
        X = df_region_labeled[params.cols].values  # pylint: disable=invalid-name
        X_scaled = robust_scale(X)  # pylint: disable=invalid-name
        clusterer = hdbscan.HDBSCAN(min_cluster_size=params.bin_min_size)
        clusterer.fit(X_scaled)
        inlabels = clusterer.labels_.astype(str)
    except ValueError:
        inlabels = ["0"]

    res = pd.DataFrame(
        {
            params.region_col: region,
            f"{params.split_metric_col}_bin": label,
            "cls": inlabels
        }, index=df_region_labeled.index
    )
    res = res.assign(label=lambda x: x[params.region_col].astype(str) + x[f"{params.split_metric_col}_bin"] + x.cls)
    return res
