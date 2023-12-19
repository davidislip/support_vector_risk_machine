from pandas.tseries.offsets import MonthEnd
import pandas as pd
import warnings


def get_monthly_adjusted(tickers, start_date, end_date, Database, return_daily=False, data_column = 'adjClose'):
    """

    :param Database:
    :param tickers:
    :param start_date:
    :param end_date:
    :return:
    """
    info = Database.query("SELECT a.* FROM equities_series a "
                                "inner join ticker_id b "
                                "on a.ticker_currency = b.ticker_currency "
                                "where b.ticker in  " + str(tickers) +
                                " and a.frequency = 'D'")
    info.date = pd.to_datetime(info.date, format='%Y-%m-%d')

    data_subset = info[(start_date <= info.date) & (info.date <= end_date)].copy(deep=True)

    data_subset[['ticker', 'currency']] =data_subset.ticker_currency.str.split('-', expand=True)
    # get tiingo then yahoo - logic will have to change if alpha vantage daily data is used
    priority_sorted_data = data_subset.sort_values(by=['ticker_currency', 'date', 'source']).groupby(
        ['ticker_currency', 'date']).first()
    # pivot to form panel of prices
    data = priority_sorted_data.reset_index().pivot(index='date', columns=['ticker'], values=[
        data_column])  # will not work if there are duplicate tickers at the same time
    # get the last recorded price of the month
    if data_column != 'adjClose':
        warnings.warn("IMPORTANT: The monthly aggregation returned in the dataframe is only the last observation")

    monthly = data.groupby(pd.Grouper(freq="M")).last()
    monthly.index = monthly.index + MonthEnd(0)
    if return_daily:
        return monthly, data
    else:
        return monthly


def get_monthly_adjusted_price(tickers, start_date, end_date, Database, return_daily=False):
    """

    :param return_daily:
    :param Database:
    :param tickers:
    :param start_date:
    :param end_date:
    :return:
    """
    return get_monthly_adjusted(tickers, start_date, end_date, Database, return_daily=return_daily,
                                data_column='adjClose')


def get_monthly_factors(factors, start_date, end_date, Database):
    """

    :param factors:
    :param start_date:
    :param end_date:
    :param Database:
    :return:
    """
    factor_info = Database.query("SELECT * FROM factors where frequency = 'M' and trim(factor) in " + str(factors))
    factor_data = factor_info.pivot(index='date', columns=['factor'], values=['ret'])
    factor_data.index = pd.to_datetime(factor_data.index, format='%Y-%m-%d')
    factor_data.index = factor_data.index + MonthEnd(
        0)  # see https://stackoverflow.com/questions/37354105/find-the-end-of-the-month-of-a-pandas-dataframe-series
    factor_data_subset = factor_data.loc[start_date:end_date, :]
    return factor_data_subset


def get_mie377_data(Database, tickers, factors, start_date, end_date):
    """
    this function forms a monthly dataset of factors and prices for use in the MIE377 course and any research
    :param Database:
    :param tickers:
    :param factors:
    :param start_date:
    :param end_date:
    :return:
    """

    # check that we have all the tickers
    ticker_currency_pairs = Database.query("SELECT * FROM ticker_id where ticker in " + str(tickers))
    assert len(ticker_currency_pairs) == len(tickers)  # we have all the tickers

    factor_data_subset = get_monthly_factors(factors, start_date, end_date, Database)

    monthly_adjusted_prices = get_monthly_adjusted_price(tickers, start_date, end_date, Database)

    assert (len(monthly_adjusted_prices) == len(factor_data_subset))  # got to make sure these are the same size

    return monthly_adjusted_prices, factor_data_subset
