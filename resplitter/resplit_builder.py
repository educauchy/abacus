import pandas as pd
from resplitter.params import ResplitParams


class ResplitBuilder():
    def __init__(self,
                 df: pd.DataFrame,
                 resplit_params: ResplitParams):
        self.df = df
        self.params = resplit_params

    
    def collect(self):
        df_resplit = (self.df[self.df[self.params.group_col]==self.params.test_group_value]
                    .reset_index(drop=True)
        )
        df_strata_count = (df_resplit.groupby(self.params.strata_col)
                    .agg('count')[self.params.group_col].reset_index()
        )

        min_count = df_strata_count.min()[self.params.group_col]
        min_strata = (df_strata_count[df_strata_count[self.params.group_col]==min_count][self.params.strata_col]
                    .values[0]
        )

        df_restrata = df_resplit[df_resplit[self.params.strata_col]==min_strata]

        for strata in df_strata_count[df_strata_count[self.params.strata_col]!=min_strata][self.params.strata_col]:
            df_strata = df_resplit[df_resplit[self.params.strata_col]==strata]
            df_strata = df_strata.sample(n=min_count)
            df_restrata = pd.concat([df_restrata, df_strata])
        
        return pd.concat([df_restrata,
                        (self.df[self.df[self.params.group_col]!=self.params.test_group_value]
                        .reset_index(drop=True))])
