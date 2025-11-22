import pandas as pd

def compute_stats(df, min_reviews=5):
    grp = df.groupby(['condition','drugName'])['rating'].agg(['mean','count']).reset_index()
    grp = grp.rename(columns={'mean':'mean_rating','count':'count_reviews'})
    grp_filtered = grp[grp['count_reviews'] >= min_reviews].copy()
    return grp_filtered

def top_n_by_condition(stats_df, condition, n=5):
    cond = stats_df[stats_df['condition'].str.lower() == condition.lower()]
    if cond.empty:
        return pd.DataFrame()
    res = cond.sort_values(['mean_rating','count_reviews'], ascending=[False, False]).head(n)
    return res
