import numpy as np
from services.estimators import *
from services.optimization import *


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
        if investor_preferences is None:
            investor_preferences = {}
        self.NumObs = NumObs  # number of observations to use
        self.estimator = estimator  # estimator is a function
        self.optimizer = optimizer
        self.investor_preferences = investor_preferences

    def execute_strategy(self, periodReturns, factorReturns, additional_info=None):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param additional_info:
        :param factorReturns:
        :param periodReturns:
        :return:x
        """

        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        if additional_info is None:
            estimates = self.estimator(returns, factRet)
        else:
            estimates = self.estimator(**additional_info)

        x = self.optimizer(*estimates, **self.investor_preferences)
        return x




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
