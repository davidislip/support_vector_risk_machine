import pandas as pd


def merge_ticker_df(Database, concatenated_df):
    """

    :param Database:
    :param concatenated_df:
    :return:
    """
    source_id = Database.query("SELECT * FROM source_id")
    source_id.rename(columns={'id': "source_id"}, inplace=True)
    ticker_currency_id = Database.query("SELECT * FROM ticker_id")
    ticker_currency_id.rename(columns={'id': "ticker_currency_id"}, inplace=True)
    equity_index_id = Database.query('SELECT * FROM equity_index_id')
    equity_index_id.rename(columns={'id': "equity_index_id"}, inplace=True)

    merged_df = concatenated_df.merge(source_id, on='source', how='inner') \
        .merge(ticker_currency_id, on='ticker_currency', how='inner') \
        .merge(equity_index_id, on='equity_index', how='inner')

    merged_df['id'] = merged_df.source_id.astype(str) + merged_df.ticker_currency_id.astype(str) \
                      + merged_df.equity_index_id.astype(str) + merged_df.startDate.dt.strftime('%Y%m%d')
    merged_df['id'] = merged_df['id'].astype('int64')
    return merged_df


def create_source_table(Database):
    """
    populates the source table with the dataframe defined below
    The sources are actually important because they will help define
    unique id's/primary keys for other tables
    :param Database:
    :return:
    """
    df = pd.DataFrame(
        {'source': ['gfd', 'mie377', 'mlfactor', 'tiingo', 'av', 'yahoo', 'nasdaq', 'famafrench', 'zimmerman', 'wrds'],
         'description': ['Global financial database', 'MIE377 dataset', 'Machine learning for factor investing',
                         'tiingo', 'Alpha Vantage', 'Yahoo Finance', 'Nasdaq Data Link', 'Fama French',
                         'Zimmerman Open Asset Pricing', 'wharton research']})
    df = df.reset_index()
    df.columns = ['id'] + list(df.columns[1:])
    Database.insert_from_df_by_name('source_id', df, indices=[0, 1])


def create_ticker_id(Database):
    """
    creates the id table for various ticker - currency pairs
    this is not perfect because sometimes tickers can be delisted and then reintroduced
    however the Tiingo api does not return data for the duplicates

    There are four csv's that are used to populate the table:
    1) A list of current tickers that constitute the Nasdaq 100
       (taken from global financial database)
    2) A list of tickers that historically constituted the S&P 500
       (taken from global financial database)
    3) A list of hypothetical tickers taking from the dataset for the book Machine Learning
       for Factor Portfolios
    4) A list of tickers supported by the Tiingo API (available for download from thier website)
    :param Database:
    :return:
    """
    base = Database.query("SELECT * FROM ticker_id")

    # read in all tickers
    gfd_nasdaq_tickers = pd.read_excel("local data/gfd_tickers/nasdaq_current.xlsx")
    gfd_nasdaq_tickers['priceCurrency'] = 'USD'
    gfd_nasdaq_tickers['ticker'] = gfd_nasdaq_tickers['Ticker'].astype('string')
    gfd_nasdaq_tickers['ticker_currency'] = gfd_nasdaq_tickers.ticker + "-" + gfd_nasdaq_tickers.priceCurrency

    gfd_spy_history_tickers = pd.read_excel("local data/gfd_tickers/SPY_history.xlsx")
    gfd_spy_history_tickers['priceCurrency'] = 'USD'
    gfd_spy_history_tickers['ticker'] = gfd_spy_history_tickers['Ticker'].astype('string')
    gfd_spy_history_tickers[
        'ticker_currency'] = gfd_spy_history_tickers.ticker + "-" + gfd_spy_history_tickers.priceCurrency

    ml_factor = pd.read_csv("local data/mlfactor/data_ml.csv")
    ml_factor_tickers = pd.unique('MLBook' + ml_factor.stock_id.astype('string'))
    ml_factor_tickers = pd.DataFrame(ml_factor_tickers)
    ml_factor_tickers.columns = ['ticker']
    ml_factor_tickers['priceCurrency'] = 'USD'
    ml_factor_tickers['ticker_currency'] = ml_factor_tickers.ticker + "-" + ml_factor_tickers.priceCurrency

    tiingo_tickers = pd.read_csv("local data/tiingotickers/supported_tickers.csv")
    tiingo_tickers['ticker_currency'] = tiingo_tickers.ticker + "-" + tiingo_tickers.priceCurrency

    all_tickers = [tiingo_tickers.loc[:, ['ticker_currency', 'ticker', 'priceCurrency']],
                   gfd_nasdaq_tickers.loc[:, ['ticker_currency', 'ticker', 'priceCurrency']],
                   gfd_spy_history_tickers.loc[:, ['ticker_currency', 'ticker', 'priceCurrency']],
                   ml_factor_tickers.loc[:, ['ticker_currency', 'ticker', 'priceCurrency']]]
    all_tickers_df = pd.concat(all_tickers, axis=0).drop_duplicates()
    all_tickers_df = all_tickers_df.fillna('')
    to_be_added = all_tickers_df.loc[~all_tickers_df.ticker_currency.isin(base.ticker_currency)].reset_index()
    to_be_added.columns = base.columns
    to_be_added.id = to_be_added.id + len(base)
    #
    Database.insert_from_df_by_name('ticker_id', to_be_added, indices=[0, 1])


def fill_tickers_from_tiingo(Database):
    """
    inserts the tiingo csv into the database
    :param Database:
    :return:
    """
    base = Database.query('SELECT * FROM tiingo_tickers' + " LIMIT 1")
    tiingo_tickers = pd.read_csv("local data/tiingotickers/supported_tickers.csv")
    tiingo_tickers['source'] = "tiingo"
    tiingo_tickers = tiingo_tickers.reset_index()
    tiingo_tickers.columns = ['id'] + list(tiingo_tickers.columns[1:])
    tiingo_tickers = tiingo_tickers[base.columns]
    tiingo_tickers.startDate = pd.to_datetime(tiingo_tickers.startDate.fillna("1900-01-01"))
    tiingo_tickers.endDate = pd.to_datetime(tiingo_tickers.endDate.fillna("1900-01-01"))
    Database.insert_from_df_by_name('tiingo_tickers', tiingo_tickers, indices=[0])


def insert_gfd_ticker_index_membership(Database):
    """
    take csv outputs from global financial database and store them in the SQLite DB
    :param Database:
    :return:
    """
    base = Database.query("SELECT * from ticker_index_membership" + " LIMIT 1")
    # read in all tickers
    gfd_nasdaq_tickers = pd.read_excel("local data/gfd_tickers/nasdaq_current.xlsx")
    gfd_nasdaq_tickers['priceCurrency'] = 'USD'
    gfd_nasdaq_tickers['ticker'] = gfd_nasdaq_tickers['Ticker'].astype('string')
    gfd_nasdaq_tickers['ticker_currency'] = gfd_nasdaq_tickers.ticker + "-" + gfd_nasdaq_tickers.priceCurrency
    gfd_nasdaq_tickers['equity_index'] = 'NASDAQ100'
    gfd_nasdaq_tickers['source'] = 'gfd'

    gfd_spy_history_tickers = pd.read_excel("local data/gfd_tickers/SPY_history.xlsx")
    gfd_spy_history_tickers['priceCurrency'] = 'USD'
    gfd_spy_history_tickers['ticker'] = gfd_spy_history_tickers['Ticker'].astype('string')
    gfd_spy_history_tickers[
        'ticker_currency'] = gfd_spy_history_tickers.ticker + "-" + gfd_spy_history_tickers.priceCurrency
    gfd_spy_history_tickers['equity_index'] = 'S&P500'
    gfd_spy_history_tickers['source'] = 'gfd'
    all_memberships = pd.concat([gfd_nasdaq_tickers, gfd_spy_history_tickers], axis=0)
    all_memberships[['startDate', 'endDate']] = all_memberships.Date.apply(lambda x: pd.Series(str(x).split("-")))

    all_memberships.endDate = pd.to_datetime(all_memberships.endDate, format="%m/%d/%Y")

    all_memberships.startDate = pd.to_datetime(all_memberships.startDate, format="%Y")
    merged_df = merge_ticker_df(Database, all_memberships)
    merged_df['ticker'] = merged_df['Ticker'].astype('string')
    Database.insert_from_df_by_name("ticker_index_membership", merged_df[base.columns], indices=[0])


def create_exchange_id(Database):
    """
    populate the list of exchanges from the tiingo tickers csv
    :param Database:
    :return:
    """
    base = Database.query("SELECT * FROM exchange_id")
    tiingo_tickers = pd.read_csv("local data/tiingotickers/supported_tickers.csv")
    updated_exchanges = pd.DataFrame(tiingo_tickers.exchange.unique()).reset_index().fillna("")
    updated_exchanges.columns = base.columns
    new_updated_exchanges = updated_exchanges.loc[~updated_exchanges.exchange.isin(base.exchange)]
    new_updated_exchanges.id = new_updated_exchanges.id + len(base)
    Database.insert_from_df_by_name('exchange_id', new_updated_exchanges, indices=[0, 1])


def create_asset_id(Database):
    """
    populate the unique id's for each asset type in the tiingo flat file
    :param Database:
    :return:
    """
    base = Database.query("SELECT * FROM asset_id")
    tiingo_tickers = pd.read_csv("local data/tiingotickers/supported_tickers.csv")
    updated_assets = pd.DataFrame(tiingo_tickers.assetType.unique()).reset_index().fillna("")
    updated_assets.columns = base.columns
    new_updated_assets = updated_assets.loc[~updated_assets.asset.isin(base.asset)]
    new_updated_assets.id = new_updated_assets.id + len(base)
    Database.insert_from_df_by_name('asset_id', new_updated_assets, indices=[0, 1])


def create_currency_id(Database):
    """
    populate the unique id's for each currency in the tiingo flat file
    :param Database:
    :return:
    """

    base = Database.query("SELECT * FROM currency_id")
    tiingo_tickers = pd.read_csv("local data/tiingotickers/supported_tickers.csv")
    updated_currency = pd.DataFrame(tiingo_tickers.priceCurrency.unique()).reset_index().fillna("")
    updated_currency.columns = base.columns
    new_updated_currency = updated_currency.loc[~updated_currency.currency.isin(base.currency)]
    new_updated_currency.id = new_updated_currency.id + len(base)
    Database.insert_from_df_by_name('currency_id', new_updated_currency, indices=[0, 1])


def create_frequencyid(Database):
    """
    hard code some values of common frequencies of data
    Daily (D)
    Weekly (W)
    Monthly (M)
    Quarterly (Q)
    Fiscal Quarter (FQ)
    SemiAnnual (SA)
    Yearly (Y)
    :param Database:
    :return:
    """
    base = Database.query("SELECT * FROM frequency_id")
    df = pd.DataFrame({'frequency': ['D', 'W', 'M', 'Q', 'FQ', 'SA', 'Y']}).reset_index()
    df.columns = base.columns
    new_freq = df.loc[~df.frequency.isin(base.frequency)]
    new_freq.id = new_freq.id + len(base)
    Database.insert_from_df_by_name('frequency_id', new_freq, indices=[0, 1])


def create_equity_index_id(Database):
    base = Database.query("SELECT * FROM equity_index_id")
    df = pd.DataFrame({'equity_index': ['NASDAQ100', 'S&P500']}).reset_index()
    df.columns = base.columns
    new_equity_index = df.loc[~df.equity_index.isin(base.equity_index)]
    new_equity_index.id = new_equity_index.id + len(base)
    Database.insert_from_df_by_name('equity_index_id', new_equity_index, indices=[0, 1])
