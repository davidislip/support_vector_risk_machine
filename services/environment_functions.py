import math


class environment:
    """
    an object that keeps track of the variables in the backtester
    """

    def __init__(self):
        self.periodReturns = None
        self.periodFactRet = None
        self.period_daily_adjClose = None
        self.period_Context = None
        self.previous_portfolio = None
        self.n = None


# Estimators
def populate_exponential_weighted_estimator(Strategy, env):
    estimation_info = {'k': Strategy.investor_preferences['k'], 'alpha': Strategy.investor_preferences['alpha'],
                       'daily_prices': env.period_daily_adjClose}
    return estimation_info


# Standard MVO
# def populateMVO(Strategy, env):
#     optimization_info = {'mu': Strategy.current_estimates[0], 'Q': Strategy.current_estimates[1]}
#     return optimization_info


def populateMVO(Strategy, env):
    optimization_info = {'mu': Strategy.current_estimates[0], 'Q': Strategy.current_estimates[1]}
    # inspect the return constraint function
    ret_Constr_fnc = Strategy.investor_preferences['target_return_strategy']
    args = Strategy.investor_preferences['target_return_strategy_args']

    targetRetkwargs = {'mu': Strategy.current_estimates[0]}
    for arg in args:
        targetRetkwargs[arg] = Strategy.investor_preferences[arg]
    optimization_info['targetRet'] = ret_Constr_fnc(**targetRetkwargs)
    return optimization_info


# Helper to add turnover parameters
def addTurnoverConstraint(optimization_info, Strategy, env):
    optimization_info['turnover_limit'] = Strategy.investor_preferences['turnover_limit']
    optimization_info['previous_portfolio'] = env.previous_portfolio


def populateMVOTurnover(Strategy, env):
    optimization_info = populateMVO(Strategy, env)
    addTurnoverConstraint(optimization_info, Strategy, env)
    return optimization_info


# helper to add cardinality constraint parameters
def addCardinalityConstraint(optimization_info, Strategy, env):
    optimization_info['MipGap'] = Strategy.investor_preferences['MipGap']
    optimization_info['limit_time'] = Strategy.investor_preferences['limit_time']
    cardinality_ratio = Strategy.investor_preferences['cardinality_ratio']
    n = env.n
    optimization_info['K'] = math.floor(n * cardinality_ratio)


# Cardinality constraints
def populateCardMVO(Strategy, env):
    optimization_info = populateMVO(Strategy, env)
    addCardinalityConstraint(optimization_info, Strategy, env)
    return optimization_info


# Card MVO with Turnover
def populateCardMVOTurnover(Strategy, env):
    optimization_info = populateMVO(Strategy, env)
    addTurnoverConstraint(optimization_info, Strategy, env)
    addCardinalityConstraint(optimization_info, Strategy, env)
    return optimization_info


def mean_target(mu):
    return mu.mean()


def premium_target(mu, premium):
    return (1 + premium) * mu.mean()


def ticker_return_target(mu, ticker_index):
    return mu[ticker_index]
