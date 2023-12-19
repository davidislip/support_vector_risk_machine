import pandas_datareader.data as web
import time
import pandas as pd
import os
import pandas_datareader as pdr
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import wrds

# This collection of functions is meant to gather and store data from various
# sources in a Mongo DB in a consistent format
# Factors data will be in a factors table (source, frequency, factor_description, time, value)
# Asset data will be in a closing price asset table (source, frequency, ticker, metric, time, value)
# Equity Fundamentals Data (source, metric, frequency, ticker, time, value)
# source list: source, url, date_added, description
# Equity tickers table (source, ticker, start_date, end_date)
# Options Surface Data


def update_factor_id_from_famafrench(Database, fama_french_datasets_to_get, start, end):
    """
    this function updates the factor ids table from the fama
    french dataset as indicated in the list of dataset names
    denoted by fama_french_datasets_to_get

    after updating the factor id's, the function returns the downloaded
    datasets in a dictionary

    :param fama_french_datasets_to_get: list of dataset's to get
    :return:
    """
    base = Database.query("SELECT * FROM factor_id")
    dataset_dictionary = {}
    factor_names = []
    for dataset in fama_french_datasets_to_get:
        ds = web.DataReader(dataset, 'famafrench', start, end)
        dataset_dictionary[dataset] = ds[0]
        factor_names += list(ds[0].columns)
        time.sleep(5)  # take your time

    insert_list_into_lookup_table('factor_id', factor_names, base, Database)

    return dataset_dictionary


def insert_open_asset_factors(Database, open_asset_df):
    """
    inserts the open asset factors into the factor database
    :param Database:
    :param open_asset_df:
    :return:
    """
    base = Database.query("SELECT * from factors")
    source = 'zimmerman'
    frequency = 'D'

    dfs = []

    for filename, df in open_asset_df.items():
        for column in [col for col in df.columns if col.lower() != "date"]:
            dataset_to_add = df[['date', column]].dropna()
            dataset_to_add['factor'] = "OA_" + ['' + s for s in filename.split(".csv")][0] + column
            dataset_to_add['frequency'] = frequency
            dataset_to_add['source'] = source
            reordered_df = dataset_to_add[['source', 'frequency', 'factor', 'date', column]].reset_index()
            reordered_df.columns = base.columns
            reordered_df.dropna()
            dfs.append(reordered_df)
    concatenated_df = pd.concat(dfs, axis=0)
    concatenated_df['date'] = pd.to_datetime(concatenated_df.date, format="%Y-%m-%d")
    merged_df = merge_factor_df(Database, concatenated_df)
    Database.insert_from_df_by_name('factors', merged_df[base.columns], indices=[0])


def upload_fama_factor_data(Database, fama_factors_to_upload):
    """

    :param Database:
    :param fama_factors_to_upload:
    :return:
    """
    base = Database.query("SELECT * from factors")

    source = 'famafrench'

    dfs = []
    for dataset_name, df in fama_factors_to_upload.items():
        if 'weekly' in dataset_name:
            frequency = 'W'
        elif 'daily' in dataset_name:
            frequency = 'D'
        else:
            frequency = 'M'
        if df.index.dtype == 'period[M]':
            df.index = df.index.to_timestamp()

        for column in df.columns:
            dataset_to_add = df[column].reset_index()
            dataset_to_add['factor'] = column
            dataset_to_add['frequency'] = frequency
            dataset_to_add['source'] = source
            reordered_df = dataset_to_add[['source', 'frequency', 'factor', 'Date', column]].reset_index()
            reordered_df.columns = base.columns
            dfs.append(reordered_df)
    concatenated_df = pd.concat(dfs, axis=0)

    merged_df = merge_factor_df(Database, concatenated_df)

    Database.insert_from_df_by_name('factors', merged_df[base.columns], indices=[0])


def retrieve_ml_factor():
    ml_factor = pd.read_csv("local data/mlfactor/data_ml.csv")
    ml_factor['ticker'] = 'MLBook' + ml_factor.stock_id.astype('string')

    simulated_prices = []
    for stock, group in ml_factor[['ticker', 'date', 'R1M_Usd']].sort_values(by=['ticker', 'date']).groupby("ticker"):
        one_month_previous = pd.to_datetime(group.date.iloc[0], format="%Y-%m-%d") - pd.DateOffset(months=1)
        last_day_of_previous_month = pd.Timestamp(pd.Period(one_month_previous, freq='M').end_time.date())
        group.date = pd.to_datetime(group.date, format="%Y-%m-%d")
        group.index = group.date
        df_to_be_added = pd.DataFrame([stock, last_day_of_previous_month, 1.0], columns=[last_day_of_previous_month],
                                      index=group.columns).T
        group.R1M_Usd = (1 + group.R1M_Usd / 100).cumprod()
        simulated_price_df = pd.concat([df_to_be_added, group], axis=0).reset_index(drop=True)
        simulated_prices.append(simulated_price_df)
    df_prices = pd.concat(simulated_prices, axis=0)
    df_prices['priceCurrency'] = 'USD'
    df_prices['ticker_currency'] = df_prices.ticker + "-" + df_prices.priceCurrency

    df_prices = df_prices.rename(columns={"R1M_Usd": "adjClose"})
    df_prices['source'] = 'mlfactor'
    df_prices['frequency'] = 'M'
    df_prices['close'] = -999
    df_prices['high'] = -999
    df_prices['low'] = -999
    df_prices['open'] = -999
    df_prices['volume'] = -999
    df_prices['adjHigh'] = -999
    df_prices['adjLow'] = -999
    df_prices['adjOpen'] = -999
    df_prices['adjVolume'] = -999
    df_prices['divCash'] = -999
    df_prices['splitFactor'] = -999
    return df_prices


def retrieve_alpha_vantage(freq='W', ticker_info=None):
    """Load data from alpha vantage
    Inputs
    start: start date
    end: end date
    freq: data frequency (daily, weekly, monthly)
    save_results: Boolean. State whether the data should be cached for future use.
    Outputs
    X: TrainTest object with feature data split into train, validation and test subsets
    """

    if ticker_info is None:
        tick_list = ['AAPL', 'MSFT', 'AMZN', 'C', 'JPM', 'BAC', 'XOM', 'HAL', 'MCD', 'WMT', 'COST', 'CAT', 'LMT',
                     'JNJ', 'PFE', 'DIS', 'VZ', 'T', 'ED', 'NEM']
    else:
        if len(ticker_info) != len(list(ticker_info.ticker.unique())):
            print("Duplicate tickers being uploaded -- investigate")
            raise Exception
        tick_list = list(ticker_info.ticker.unique())

    ts = TimeSeries(key=os.environ['ALPHA_VANTAGE'], output_format='pandas', indexing_type='date')

    # Download asset data
    X = []
    for tick in tick_list:
        if freq == "D":
            try:
                data, _ = ts.get_daily_adjusted(symbol=tick)
            except:
                print("ticker ", tick, " Invalid")
                data = None
                data['ticker'] = tick
        elif freq == "W":
            try:
                data, _ = ts.get_weekly_adjusted(symbol=tick)
                data['ticker'] = tick
            except:
                print("ticker ", tick, " Invalid")
                data = None

        if data is not None:
            X.append(data)
        time.sleep(10.0)
    X = pd.concat(X, axis=0)
    X['source'] = 'av'
    X['frequency'] = freq
    X = X.reset_index()
    X = X.rename(columns={"1. open": "open", "2. high": "high",
                          "3. low": "low", "4. close": "close",
                          "5. adjusted close": "adjClose", "6. volume": "volume",
                          "7. dividend amount": "divCash", })
    X['splitFactor'] = X.close / X.adjClose
    X['adjHigh'] = X.high / X['splitFactor']
    X['adjLow'] = X.low / X['splitFactor']
    X['adjOpen'] = X.open / X['splitFactor']
    X['adjVolume'] = X.volume / X['splitFactor']
    X = X.merge(ticker_info, on='ticker', how='inner')  # only works because no duplicate tickers are considered
    return X.reset_index()


def retrieve_yahoo_finance(start, end, freq='W', ticker_info=None):
    """Load data from alpha vantage
    Inputs
    start: start date
    end: end date
    freq: data frequency (daily, weekly, monthly)
    save_results: Boolean. State whether the data should be cached for future use.
    Outputs
    X: TrainTest object with feature data split into train, validation and test subsets
    """
    start = pd.to_datetime(start).strftime("%Y-%m-%d")
    end = pd.to_datetime(end).strftime("%Y-%m-%d")
    if ticker_info is None:
        tick_list = ['AAPL', 'MSFT', 'AMZN', 'C', 'JPM', 'BAC', 'XOM', 'HAL', 'MCD', 'WMT', 'COST', 'CAT', 'LMT',
                     'JNJ', 'PFE', 'DIS', 'VZ', 'T', 'ED', 'NEM']
    else:
        if len(ticker_info) != len(list(ticker_info.ticker.unique())):
            print("Duplicate tickers being uploaded -- investigate")
            raise Exception
        tick_list = list(ticker_info.ticker.unique())

    if freq == "D":
        X = yf.download(tick_list, start=start, end=end, interval="1d")
    elif freq == "W":
        X = yf.download(tick_list, start=start, end=end, interval="1wk")

    if len(tick_list) > 1:
        X = X.T.unstack(level=1).T
    else:
        X['ticker'] = tick_list[0]
    X = X.reset_index()
    X['source'] = 'yahoo'
    X['frequency'] = freq
    X['divCash'] = -999  # dummy value
    X = X.rename(columns={"Date": "date", "level_1": "ticker", "Adj Close": "adjClose",
                          "Close": "close", "High": "high", "Low": "low",
                          "Open": "open", "Volume": "volume"})

    X['splitFactor'] = X.close / X.adjClose
    X['adjHigh'] = X.high / X['splitFactor']
    X['adjLow'] = X.low / X['splitFactor']
    X['adjOpen'] = X.open / X['splitFactor']
    X['adjVolume'] = X.volume / X['splitFactor']
    X = X.merge(ticker_info, on='ticker', how='inner')
    return X


def retrieve_tiingo(start, end, ticker_info):
    """
    retrieve data from tiingo
    :param start:
    :param end:
    :param tick_list:
    :return:
    """
    if ticker_info is None:
        tick_list = ['AAPL', 'MSFT', 'AMZN', 'C', 'JPM', 'BAC', 'XOM', 'HAL', 'MCD', 'WMT', 'COST', 'CAT', 'LMT',
                     'JNJ', 'PFE', 'DIS', 'VZ', 'T', 'ED', 'NEM']
    else:
        if len(ticker_info) != len(list(ticker_info.ticker.unique())):
            print("Duplicate tickers being uploaded -- investigate")
            raise Exception
        tick_list = list(ticker_info.ticker.unique())
    df_tiingo = pdr.get_data_tiingo(tick_list, api_key=os.environ['TIINGO'], start=start, end=end)
    df_tiingo['source'] = "tiingo"
    df_tiingo['frequency'] = 'D'
    df_tiingo = df_tiingo.reset_index()
    df_tiingo = df_tiingo.rename(columns={"symbol": "ticker"})
    df_tiingo = df_tiingo.merge(ticker_info, on='ticker', how='inner')
    return df_tiingo


def retrieve_tiingo_crypto_closing(database, tickers):
    """

    :param database:
    :param tickers:
    :return:
    """
    return None


def get_raw_wrds_ratios(username, ticker_info):
    """

    :param username:
    :return:
    """
    db = wrds.Connection(wrds_username=username)
    query = "select * from wrdsapps_finratio.firm_ratio where TICKER IN %(tickers)s"
    # get fundamentals data for tickers that have not been recycled - i.e. most recent ticker
    params = {"tickers": tuple(ticker_info.ticker.unique())}
    raw_data = db.raw_sql(
        query,
        params=params,
    )
    return raw_data


def process_wrds_ratio_data(raw_data, ticker_info):
    """

    :param raw_data:
    :return:
    """
    wrds_data = raw_data.drop(columns=[col for col in raw_data.columns if 'desc' in col] + ['adate', 'qdate', 'cusip'])

    wrds_data_wide = wrds_data.drop(columns=['gvkey', 'permno'])

    wrds_data_long = wrds_data_wide.melt(id_vars=['public_date', 'ticker']).dropna().drop_duplicates()
    wrds_data_long['frequency'] = 'Q'
    wrds_data_long['source'] = 'wrds'
    wrds_data_long = wrds_data_long.rename(columns={"public_date": "date", "variable": "metric"})

    group_cols = ['ticker', 'date', 'metric', 'frequency', 'source']
    de_duped_df = replace_duplicates_with_mean(group_cols, wrds_data_long)
    de_duped_df_out = de_duped_df.merge(ticker_info[['ticker', 'ticker_currency']], on='ticker')  # get ticker currency
    assert len(de_duped_df_out) == len(de_duped_df)
    de_duped_df_out.date = pd.to_datetime(de_duped_df_out.date.astype(str), format='%Y-%m-%d')
    return de_duped_df_out


def update_factor_id_from_open_asset_pricing(Database, directory='local data/zimmerman/Predictor/'):
    base = Database.query("SELECT * FROM factor_id" + " LIMIT 1")

    # assign directory

    factor_names_append = []
    dataset_dictionary = {}

    for filename in os.listdir(directory):
        factor_name = "OA_" + ['' + s for s in filename.split(".csv")][0]
        factors = pd.read_csv(directory + filename)
        factor_names = factor_name + factors.columns[1:]
        factor_names_append += list(factor_names)
        dataset_dictionary[filename] = factors

    insert_list_into_lookup_table('factor_id', factor_names_append, base, Database)

    return dataset_dictionary


def insert_list_into_lookup_table(table_name, factor_names, base, Database):
    """
    inserts the factor_names list into the table_name in the database
    the scheme information is contained in the base dataframe
    :param table_name:
    :param factor_names:
    :param base:
    :param Database:
    :return:
    """
    df = pd.DataFrame(pd.unique(factor_names)).reset_index()
    df.columns = base.columns
    new_factors = df.loc[~df.factor.isin(base.factor)]
    new_factors.id = new_factors.id + len(base)
    Database.insert_from_df_by_name(table_name, new_factors, indices=[0, 1])


def merge_factor_df(Database, concatenated_df):
    """

    :param Database:
    :param concatenated_df:
    :return:
    """
    source_id = Database.query("SELECT * FROM source_id")
    source_id.rename(columns={'id': "source_id"}, inplace=True)
    frequency_id = Database.query("SELECT * FROM frequency_id")
    frequency_id.rename(columns={'id': "frequency_id"}, inplace=True)
    factor_id = Database.query('SELECT * FROM factor_id')
    factor_id.rename(columns={'id': "factor_id"}, inplace=True)

    merged_df = concatenated_df.merge(source_id, on='source', how='inner') \
        .merge(frequency_id, on='frequency', how='inner') \
        .merge(factor_id, on='factor', how='inner')

    merged_df['id'] = merged_df.source_id.astype(str) + merged_df.frequency_id.astype(str) \
                      + merged_df.factor_id.astype(str) + merged_df.date.dt.strftime('%Y%m%d')
    merged_df['id'] = merged_df['id'].astype('int64')
    return merged_df


def merge_equities_df(Database, concatenated_df):
    """

    :param Database:
    :param concatenated_df:
    :return:
    """
    source_id = Database.query("SELECT * FROM source_id")
    source_id.rename(columns={'id': "source_id"}, inplace=True)
    frequency_id = Database.query("SELECT * FROM frequency_id")
    frequency_id.rename(columns={'id': "frequency_id"}, inplace=True)
    ticker_currency = Database.query('SELECT * FROM ticker_id')
    ticker_currency.rename(columns={'id': "ticker_id"}, inplace=True)

    merged_df = concatenated_df.merge(source_id, on='source', how='inner') \
        .merge(frequency_id, on='frequency', how='inner') \
        .merge(ticker_currency, on='ticker_currency', how='inner')

    merged_df['id'] = merged_df.source_id.astype(str) + merged_df.frequency_id.astype(str) \
                      + merged_df.ticker_id.astype(str) + merged_df.date.dt.strftime('%Y%m%d')
    merged_df['id'] = merged_df['id'].astype('int64')
    return merged_df


def merge_fundamentals_df(Database, concatenated_df):
    """

    :param Database:
    :param concatenated_df:
    :return:
    """
    source_id = Database.query("SELECT * FROM source_id")
    source_id.rename(columns={'id': "source_id"}, inplace=True)
    frequency_id = Database.query("SELECT * FROM frequency_id")
    frequency_id.rename(columns={'id': "frequency_id"}, inplace=True)
    metric_id = Database.query("SELECT * FROM metric_id")
    metric_id.rename(columns={'id': "metric_id"}, inplace=True)
    ticker_currency = Database.query('SELECT * FROM ticker_id')
    ticker_currency.rename(columns={'id': "ticker_id"}, inplace=True)

    merged_df = concatenated_df.merge(source_id, on='source', how='inner') \
        .merge(frequency_id, on='frequency', how='inner') \
        .merge(metric_id, on='metric', how='inner') \
        .merge(ticker_currency, on='ticker_currency', how='inner')

    merged_df['id'] = merged_df.source_id.astype(str) + merged_df.frequency_id.astype(str) + merged_df.metric_id.astype(str)  \
                      + merged_df.ticker_id.astype(str) + merged_df.date.dt.strftime('%Y%m%d')
    merged_df['id'] = merged_df['id'].astype('int64')
    return merged_df


def add_to_metric_id(Database, metrics, desc = ''):
    """
    add metrics to the metric table
    :param Database:
    :param metrics:
    :param desc:
    :return:
    """
    base = Database.query("Select * from metric_id LIMIT 1")
    df = pd.DataFrame(metrics)
    df = df.reset_index()
    df['desc'] = desc
    df.columns = base.columns
    Database.insert_from_df_by_name("metric_id", df[base.columns], indices=[0, 1]) #first two columns are primary key


def replace_duplicates_with_mean(group_cols, long_df):
    """
    replaces duplicate records (where duplicates are defined by the group)
    with the mean of the groups "value" column\

    long_df must have a column called value
    :param group_cols:
    :param long_df:
    :return:
    """
    duplicate_idx = long_df[group_cols].duplicated(group_cols, keep=False)

    # mean aggregation of duplicate records
    duplicate_groups = long_df.loc[duplicate_idx, :].groupby(group_cols)
    duplicate_groups_stats = duplicate_groups.agg(['mean'])
    duplicate_groups_stats.columns = ['value']
    duplicate_groups_stats = duplicate_groups_stats.reset_index()

    de_duped_df = pd.concat([long_df.loc[~duplicate_idx, duplicate_groups_stats.columns],
                             duplicate_groups_stats], axis=0)
    return de_duped_df
def add_dfs_to_db(table_name, Database, list_of_dfs, indices_of_primary_key=[0]):
    """
    this function adds the list of dataframes to the table_name
    :param table_name:
    :param Database:
    :param list_of_dfs:
    :param indices_of_primary_key:
    :return:
    """
    base = Database.query("SELECT * from " + table_name + " LIMIT 1")
    list_with_reordered_columns = []
    for df in list_of_dfs:
        list_with_reordered_columns.append(df[base.columns])
    out = pd.concat(list_with_reordered_columns, axis=0)
    Database.insert_from_df_by_name(table_name, out, indices=indices_of_primary_key)
