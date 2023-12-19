import pandas as pd
import numpy as np


def find_index(col_index, ticker):
    for tuple in col_index:
        if tuple[1] == ticker:
            return tuple


def plot_returns(df, col_index=None):
    if col_index is None:
        (1 + df.pct_change(1)).cumproduct(axis=0).plot()
    elif type(col_index) is int:
        (1 + df.iloc[:, col_index].pct_change(1)).cumproduct(axis=0).plot()
    elif type(col_index) is str and len(df.columns.names) == 2:
        (1 + df.loc[:, find_index(df.columns, col_index)].pct_change(1)).cumprod(axis=0).plot()
    else:
        (1 + df.loc[:, find_index(df.columns, col_index)].pct_change(1)).cumprod(axis=0).plot()


def show_data_around_time(df, ticker, date, pds=10):
    date_loc = df.index.get_loc(date)
    start = max(date_loc - pds, 0)
    end = min(date_loc + pds, len(df))
    ticker_idx = find_index(df.columns, ticker)
    df.iloc[start:end].loc[:, ticker_idx].plot(legend=True)
    return df.iloc[start:end].loc[:, ticker_idx]


def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


def get_k_rolling_stats(df, date_index, k_values=None):
    if k_values is None:
        k_values = [20, 40, 60, 120, 250, 500]
    dfs = {}
    for date in date_index:
        df_t = []
        for k in k_values:
            df_kt = df[df.index <= date].iloc[-1 * k:].agg(['mean', 'median', 'std', q1, q3])

            df_kt.index = [stat + "_" + str(k) for stat in df_kt.index]
            df_t.append(df_kt)
        df_t = pd.concat(df_t, axis=0)
        df_t.columns = df_t.columns.droplevel(0)
        dfs[date] = df_t.T
    return pd.concat(dfs.values(), keys=dfs.keys())


def exponential_weighted_average(df, alpha):
    weights = (1 - alpha) ** np.arange(len(df))[::-1]
    normalized = (df - df.mean()).fillna(0).to_numpy()
    cov = (weights * normalized.T) @ normalized / weights.sum()
    mean = df.multiply(weights, axis=0).values.sum(axis=0) / (weights.sum())

    return cov, mean


def transform_log_stats(one_day_mean, one_day_covariance, k):

    """https: // en.wikipedia.org / wiki / Log - normal_distribution"""

    cov = k * one_day_covariance
    mean = k * one_day_mean
    mean_ = np.expand_dims(mean, axis=1)
    cov_ii = np.expand_dims(np.diag(cov), axis=1)
    # convert to linear returns
    log_transformed_mean = np.expand_dims(np.e ** (mean + np.diag(cov) / 2), axis=1) - 1
    NbyN = (len(mean), len(mean))
    means_added = mean_ * np.ones(NbyN) + mean_.transpose() * np.ones(NbyN)
    cov_diags_added = (1 / 2) * (cov_ii * np.ones(NbyN)) + (1 / 2) * (cov_ii.transpose() * np.ones(NbyN))
    log_transformed_cov = (np.e ** (means_added + cov_diags_added)) * (np.e ** cov - 1)
    return log_transformed_cov, log_transformed_mean