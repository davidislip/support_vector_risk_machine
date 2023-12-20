import numpy as np
from services.estimators import *
from services.optimization import *
import warnings

# this file will produce portfolios as outputs from data - the strategies can be implemented as classes or functions
# if the strategies have parameters then it probably makes sense to define them as a class


def equal_weight(periodReturns):
    """
    computes the equal weight vector as the portfolio
    :param periodReturns:
    :return:x
    """
    T, n = periodReturns.shape
    x = (1 / n) * np.ones([n])
    return x


class HistoricalMeanVarianceOptimization:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns=None):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param periodReturns:
        :param factorReturns:
        :return: x
        """
        factorReturns = None  # we are not using the factor returns
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        mu = np.expand_dims(returns.mean(axis=0).values, axis=1)
        Q = returns.cov().values
        x = MVO(mu, Q)

        return x


class OLS_MVO:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = MVO(mu, Q)
        return x


class Historical_RP:
    """
    uses historical returns to estimate the covariance matrix for risk parity
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        Q = returns.cov().values
        mu = np.expand_dims(returns.mean(axis=0).values, axis=1)
        x = RP(mu, Q)
        return x


class OLS_RP:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = RP(mu, Q)
        return x


class general_strategy:
    """
    a general strategy where the estimator and optimizer are specified, makes all the other strategies obsolete
    """

    def __init__(self, estimator, optimizer, NumObs=36, investor_preferences=None):

        self.NumObs = NumObs  # number of observations to use
        self.estimator = estimator  # estimator is a function
        self.optimizer = optimizer
        self.investor_preferences = investor_preferences  # kappa, K, q, epsilon, C, turnover limit
        self.extract_estimation_info = None
        self.extract_optimization_info = None
        self.current_results = None
        self.current_estimates = None

    def execute_strategy(self, periodReturns, factorReturns, environment=None):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param additional_optimization_info:
        :param additional_estimation_info:
        :param additional_info:
        :param factorReturns:
        :param periodReturns:
        :return:x
        """

        # Estimation Step
        if self.extract_estimation_info is not None:
            estimation_params = self.extract_estimation_info(self, environment)
        else:
            estimation_params = None

        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]

        if estimation_params is None:
            estimates = self.estimator(returns, factRet)
        else:
            estimates = self.estimator(**estimation_params)

        # Optimization Step
        self.current_estimates = estimates
        if self.extract_optimization_info is not None:
            optimization_params = self.extract_optimization_info(self, environment)
        else:
            optimization_params = None

        if optimization_params is None:
            x = self.optimizer(*estimates)
            results = {'x': x}
        else:
            results = self.optimizer(**optimization_params)  # solution, optimality gap, time to solve
            if type(results) != dict:
                warnings.warn("Optimizer interaction with environment does not return a dict")
                results = {'x': results}

        self.current_results = results

        return results['x']


class e2e_strategy:
    """
    a general e2e strategy
    """

    def __init__(self, e2e_layer, NumObsTraining=None):
        self.NumObsTraining = NumObsTraining  # number of observations to use for training the e2e optimizer
        # if it is none then we use the entire accumulated history to do the hyperparameter optimization
        self.e2e_layer = e2e_layer  # must have estimate and optimize method
        if NumObsTraining is not None:
            assert self.NumObsTraining > self.e2e_layer.NumObs  # this must be true so that we can have at least one sample to base
            # the search over delta and risk_aversion on

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        if self.NumObsTraining is None:
            returns = periodReturns
            factRet = factorReturns
        else:
            returns = periodReturns.iloc[(-1) * self.NumObsTraining:, :]
            factRet = factorReturns.iloc[(-1) * self.NumObsTraining:, :]

        x = self.e2e_layer.estimate_and_optimize(returns, factorReturns)
        return x
