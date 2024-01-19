import numpy as np
from services.data_utils import *
from sklearn.covariance import ledoit_wolf


def sample_estimator(returns, factRet=None):
    # Use this function to perform naive sample estimation
    # ----------------------------------------------------------------------
    if isinstance(returns, np.ndarray):
        mu = np.expand_dims(returns.mean(axis=0), axis=1)
        Q = np.cov(returns.T)
    else:
        mu = np.expand_dims(returns.mean(axis=0).values, axis=1)
        Q = returns.cov().values

    return mu, Q


def exponential_weighted_estimator(daily_prices, k, alpha=1 - 0.985):
    # Use this function to perform naive sample estimation
    # ----------------------------------------------------------------------
    daily_log_returns = np.log(daily_prices).diff().dropna()
    one_day_covariance, one_day_mean = exponential_weighted_average(daily_log_returns, alpha)
    log_transformed_cov, log_transformed_mean = transform_log_stats(one_day_mean, one_day_covariance, k)

    return log_transformed_mean, log_transformed_cov


def exponential_weighted_estimator_shrinkage(daily_prices, k, alpha=1 - 0.985, EstNumObs=500):
    # Use this function to perform naive sample estimation
    # ----------------------------------------------------------------------
    daily_log_returns = np.log(daily_prices).diff().dropna()
    one_day_covariance, one_day_mean = exponential_weighted_average(daily_log_returns, alpha)
    one_day_covariance, alpha = ledoit_wolf(daily_log_returns.iloc[-1 * EstNumObs:])

    log_transformed_cov, log_transformed_mean = transform_log_stats(one_day_mean, one_day_covariance, k)
    # log_transformed_mean = daily_prices.pct_change().mean().values
    # print(log_transformed_mean)
    return log_transformed_mean, log_transformed_cov


def OLS(returns, factRet):
    # Use this function to perform a basic OLS regression with all factors.
    # You can modify this function (inputs, outputs and code) as much as
    # you need to.

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # Number of observations and factors
    [T, p] = factRet.shape

    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)

    # Regression coefficients
    B = np.linalg.solve(X.T @ X, X.T @ returns)

    # Separate B into alpha and betas
    a = B[0, :]
    V = B[1:, :]

    # Residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)

    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values

    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D

    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2

    return mu, Q
