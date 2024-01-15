import numpy as np
from services.estimators import *
from services.optimization import *
import warnings
import itertools


# this file will produce portfolios as outputs from data - the strategies can be implemented as classes or functions
# if the strategies have parameters then it probably makes sense to define them as a class


def product_dict(**kwargs):
    """
    returns the product of a dictionary of lists
    """
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def equal_weight(periodReturns):
    """
    computes the equal weight vector as the portfolio
    :param periodReturns:
    :return:x
    """
    T, n = periodReturns.shape
    x = (1 / n) * np.ones([n])
    return x


def evaluate_portfolio(x, t, currentVal, currentPrices, periodPrices):
    """
    Computes the return of a portfolio x (n-dim array)
    given the time index t
    using its past values in currentVal (t-dim array)
    the prices at time t are given by the currentPrices
    and are used to calculate the number of shares in the portfolio
    at time t
    the holdings are then broadcasted forward using the prices
    over the investment period: periodPrices (n x NoPeriods) array
    """
    NoShares = x * currentVal[t] / currentPrices
    periodValue = periodPrices @ NoShares.values.T
    return (periodValue.iloc[-1][0] - currentVal[t][0]) / currentVal[t][0]  # as numbers


def execute_backtest(env, Strategy, tickers, returns, factorRet, ContextualInfo,
                     adjClose, daily_adjClose, NoPeriods, testStart, testEnd, calEnd, initialVal,
                     investPeriod, hyperparam_search=False):
    """
    Executes the backtest
    """
    backtest_results = {}
    hyperparam_hist = {}
    Q = None
    # Number of assets
    n = len(tickers)
    env.n = n

    # Preallocate space for the portfolio weights (x0 will be used to calculate
    # the turnover rate)
    x = np.zeros([n, NoPeriods])
    x0 = np.zeros([n, NoPeriods])

    # Preallocate space for the portfolio per period value and turnover
    currentVal = np.zeros([NoPeriods, 1])
    turnover = np.zeros([NoPeriods, 1])

    # Initiate counter for the number of observations per investment period
    toDay = 0

    # Measure runtime: start the clock
    start_time = time.time()

    # Empty list to measure the value of the portfolio over the period
    portfValue = []

    NoShares = None

    for t in range(NoPeriods):
        # Subset the returns and factor returns corresponding to the current calibration period.
        periodReturns = returns[returns.index <= calEnd]
        periodFactRet = factorRet[factorRet.index <= calEnd]

        # take the last available contextual observations
        period_daily_adjClose = daily_adjClose[daily_adjClose.index <= calEnd]
        period_Context_idx = ContextualInfo.index.get_level_values('date') <= calEnd
        period_Context = ContextualInfo.iloc[period_Context_idx].groupby('ticker').last()

        env.periodReturns = periodReturns
        env.periodFactRet = periodFactRet
        env.period_daily_adjClose = period_daily_adjClose
        env.period_Context = period_Context
        # there should be a feature vector for each asset
        assert len(period_Context) == n
        # all the tickers should be aligned in the context and ticker dataset
        assert (periodReturns.columns == period_Context.index).all()

        current_price_idx = (calEnd - pd.offsets.DateOffset(months=1) <= adjClose.index) & (
                adjClose.index <= calEnd)
        currentPrices = adjClose[current_price_idx]

        # Subset the prices corresponding to the current out-of-sample test period.
        periodPrices_idx = (testStart <= adjClose.index) & (adjClose.index <= testEnd)
        periodPrices = adjClose[periodPrices_idx]

        assert len(periodPrices) == investPeriod
        assert len(currentPrices) == 1
        # Set the initial value of the portfolio or update the portfolio value
        if t == 0:
            currentVal[0] = initialVal
        else:
            currentVal[t] = currentPrices @ NoShares.values.T
            # Store the current asset weights (before optimization takes place)
            x0[:, t] = currentPrices.values * NoShares.values / currentVal[t]
            # update the previous periods portfolio
            env.previous_portfolio = x0[:, t]

        # ----------------------------------------------------------------------
        # Portfolio optimization
        # You must write code your own algorithmic trading function
        # ----------------------------------------------------------------------
        # add in the estimation info

        x[:, t] = Strategy.execute_strategy(periodReturns, periodFactRet,
                                            environment=env)
        # equal_weight(periodReturns) #StrategyFunction(periodReturns, periodFactRet, x0[:,t]);

        # Calculate hyperparameter tables
        if hyperparam_search:
            Q, hyperparam_hist_t = Strategy.optimize_hyperparameters(t, currentVal, currentPrices, periodPrices, Q, environment=env)
            hyperparam_hist[t] = hyperparam_hist_t
            # Strategies hyperparameters are updated
            # Dictionary of results for hyperparameter history is populated

        # Calculate the turnover rate
        if t > 0:
            turnover[t] = np.sum(np.abs(x[:, t] - x0[:, t]))

        # Number of shares your portfolio holds per stock
        NoShares = x[:, t] * currentVal[t] / currentPrices

        # Update counter for the number of observations per investment period
        fromDay = toDay
        toDay = toDay + len(periodPrices)

        # Weekly portfolio value during the out-of-sample window
        portfValue.append(periodPrices @ NoShares.values.T)

        # Update your calibration and out-of-sample test periods
        testStart = testStart + pd.offsets.DateOffset(months=investPeriod)
        testEnd = testStart + pd.offsets.DateOffset(months=investPeriod) - pd.offsets.DateOffset(days=1)
        calEnd = testStart - pd.offsets.DateOffset(days=1)

        backtest_results[t] = Strategy.current_results
        backtest_results[t]['calEnd'] = calEnd
        backtest_results[t]['mu'] = Strategy.current_estimates[0]
        backtest_results[t]['cov'] = Strategy.current_estimates[1]
        # end loop

    portfValue = pd.concat(portfValue, axis=0)
    end_time = time.time()

    return portfValue, end_time - start_time, turnover, x, backtest_results, hyperparam_hist # dict of Strategy results

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
    a general strategy where the estimator and optimizer are specified,
    makes all the other strategies obsolete
    """

    def __init__(self, estimator, optimizer, NumObs=36, investor_preferences=None):

        self.NumObs = NumObs  # number of observations to use
        self.estimator = estimator  # estimator is a function
        self.optimizer = optimizer
        self.investor_preferences = investor_preferences  # kappa, K, q, epsilon, C, turnover limit
        self.extract_estimation_info = None  # helper function
        self.extract_optimization_info = None  # helper function
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

        results = self.optimize(estimates, environment)

        self.current_results = results

        return results['x']

    def optimize(self, estimates, environment):
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
        return results

    def optimize_hyperparameters(self, t, currentVal, currentPrices, periodPrices, Q,
                                 environment=None):

        estimates = self.current_estimates

        hyperparams = self.investor_preferences['hyperparams']
        Verbose = self.investor_preferences['Verbose']

        # update q_alpha
        q_alpha = self.investor_preferences['q_alpha']
        if t + 1 <= 1 / q_alpha:
            q_alpha = 1 / (t + 1)

        # get the initial params
        names = list(hyperparams.keys())
        names.sort()

        if Verbose:
            print("-" * 20)
            print("hyperparameters ", str(names))

        initial_param_vals = tuple([self.investor_preferences[param] for param in names])
        optimization_results = {initial_param_vals: self.current_results}
        if Q is None:
            Q = {initial_param_vals:
                          evaluate_portfolio(self.current_results['x'], t, currentVal, currentPrices, periodPrices)}

        for param_dict in product_dict(**hyperparams):
            # evaluate the product of hyperparams excluding the initial params
            already_tested = True
            param_vals = tuple([param_dict[param] for param in names])

            for name in names:
                assert type(param_dict[name]) is not str
                if abs(param_dict[name] - self.investor_preferences[name]) > 10 ** (-5):
                    already_tested = False
            if not already_tested:
                for name in names:
                    self.investor_preferences[name] = param_dict[name]
                if Verbose:
                    print("Running configuation ", str(param_dict))
                results = self.optimize(estimates, environment)
                optimization_results[param_vals] = results
                reward = evaluate_portfolio(results['x'], t, currentVal, currentPrices, periodPrices)
                if t == 0:
                    Q[param_vals] = reward
                else:
                    Q[param_vals] = (1 - q_alpha) * Q[param_vals] + q_alpha * reward
        best_params = max(Q, key=Q.get)
        if Verbose:
            print("best parameters according to value ", str(best_params))
            print("-" * 20)
        # for next iteration
        for i in range(len(names)):
            name = names[i]
            self.investor_preferences[name] = best_params[i]
        # evaluate the product of hyperparams excluding the initial params
        # store the cube
        # add information to dict
        # store hyperparameters in history
        hyperparam_hist = {'previous_hyperparams': initial_param_vals,
                                   'optimization_results': optimization_results, # dictionary of optimization results
                                   'updated_hyperparams': best_params}
        return Q, hyperparam_hist


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
