import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time


# Cardinality Constrained Optimization
def CreateCardMVOModel(mu, targetRet, Q, K, limit_time=30, MipGap=0.01):
    n = len(mu)

    m = gp.Model("miqp")

    m.Params.timeLimit = limit_time

    m.Params.MIPGap = MipGap

    b_vars = m.addMVar(n, vtype=gp.GRB.BINARY, name="b_vars")

    x_vars = m.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="x_vars")

    m.setObjective(x_vars @ Q @ x_vars, gp.GRB.MINIMIZE)

    m.addConstr(x_vars.sum() == 1)

    m.addConstr(mu @ x_vars >= targetRet)

    m.addConstr(b_vars.sum() <= K)

    m.addConstr(x_vars <= b_vars)

    return m, x_vars, b_vars


def extractSolution(n, m, x_vars, b_vars):
    x = np.zeros(n)
    z = np.zeros(n)
    for i in range(n):
        x[i] = x_vars[i].X
        z[i] = b_vars[i].X
    obj_value = m.objVal
    gap1 = m.MIPGap
    gap2 = gap1 * 100

    end = time.time()
    return obj_value, gap2, x, z


def CardMVO(mu, Q, K, limit_time=30, MipGap=0.01):
    n = len(mu)
    mu = mu.squeeze()
    targetRet = np.mean(mu)

    start = time.time()

    m, x_vars, b_vars = CreateCardMVOModel(mu, targetRet, Q, K, limit_time, MipGap)
    m.optimize()
    obj_value, gap2, x, z = extractSolution(n, m, x_vars, b_vars)

    end = time.time()
    return {'obj_value': obj_value, 'time': end - start, 'optimality gap': gap2, 'x': x, 'z': z}


def CardMVOPremiumAdjusted(mu, Q, K, premium, limit_time=30, MipGap=0.01):
    n = len(mu)
    mu = mu.squeeze()
    targetRet = (1 + premium) * np.mean(mu)

    start = time.time()

    m, x_vars, b_vars = CreateCardMVOModel(mu, targetRet, Q, K, limit_time, MipGap)
    m.optimize()
    obj_value, gap2, x, z = extractSolution(n, m, x_vars, b_vars)

    end = time.time()

    return {'obj_value': obj_value, 'time': end - start, 'optimality gap': gap2, 'x': x, 'z': z}


def CardMVOIndexBenchmark(mu, Q, K, target_index, limit_time=30, MipGap=0.01):
    n = len(mu)
    mu = mu.squeeze()

    targetRet = mu[target_index]

    start = time.time()

    m, x_vars, b_vars = CreateCardMVOModel(mu, targetRet, Q, K, limit_time, MipGap)
    m.optimize()
    obj_value, gap2, x, z = extractSolution(n, m, x_vars, b_vars)

    end = time.time()

    return {'obj_value': obj_value, 'time': end - start, 'optimality gap': gap2, 'x': x, 'z': z}


def addTurnoverConstraints(m, x_vars, previous_portfolio, turnover_limit):
    n  = len(previous_portfolio)
    absolute_delta = m.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                               vtype=gp.GRB.CONTINUOUS, name="magntiude of portfolio changes")
    m.addConstr(absolute_delta >= x_vars - previous_portfolio)
    m.addConstr(absolute_delta >= previous_portfolio - x_vars)
    m.addConstr(absolute_delta.sum() <= turnover_limit, name='turnover')

    return absolute_delta


def smallestTurnoverModel(m, absolute_delta):
    # compute the smallest turnover portfolio instead of smallest risk with turnover constraints
    m.remove(m.getConstrByName('turnover'))
    m.reset()
    m.setObjective(absolute_delta.sum())
    m.optimize()


def CardMVOTurnover(mu, Q, K, turnover_limit, previous_portfolio, limit_time=30, MipGap=0.01):
    n = len(mu)
    mu = mu.squeeze()

    targetRet = np.mean(mu)

    start = time.time()

    m, x_vars, b_vars = CreateCardMVOModel(mu, targetRet, Q, K, limit_time, MipGap)

    if previous_portfolio is not None:
        absolute_delta = addTurnoverConstraints(m, x_vars, previous_portfolio, turnover_limit)

    m.optimize()

    # model did not solve
    if m.status in (4, 3):
        smallestTurnoverModel(m, absolute_delta)

    obj_value, gap2, x, z = extractSolution(n, m, x_vars, b_vars)

    end = time.time()

    return {'obj_value': obj_value, 'time': end - start, 'optimality gap': gap2, 'x': x, 'z': z}


def CardMVOPremiumAdjustedTurnover(mu, Q, K, premium, turnover_limit, previous_portfolio, limit_time=30, MipGap=0.01):
    n = len(mu)
    mu = mu.squeeze()

    targetRet = (1 + premium) * np.mean(mu)

    start = time.time()

    m, x_vars, b_vars = CreateCardMVOModel(mu, targetRet, Q, K, limit_time, MipGap)

    if previous_portfolio is not None:
        absolute_delta = addTurnoverConstraints(m, x_vars, previous_portfolio, turnover_limit)

    m.optimize()

    # model did not solve
    if m.status in (4, 3):
        smallestTurnoverModel(m, absolute_delta)

    obj_value, gap2, x, z = extractSolution(n, m, x_vars, b_vars)

    end = time.time()

    return {'obj_value': obj_value, 'time': end - start, 'optimality gap': gap2, 'x': x, 'z': z}


def CardMVOIndexBenchmarkTurnover(mu, Q, K, target_index, turnover_limit, previous_portfolio, limit_time=30,
                                  MipGap=0.01):
    n = len(mu)
    mu = mu.squeeze()

    targetRet = mu[target_index]

    start = time.time()

    m, x_vars, b_vars = CreateCardMVOModel(mu, targetRet, Q, K, limit_time, MipGap)

    if previous_portfolio is not None:
        absolute_delta = addTurnoverConstraints(m, x_vars, previous_portfolio, turnover_limit)

    m.optimize()

    # model did not solve
    if m.status in (4, 3):
        smallestTurnoverModel(m, absolute_delta)

    obj_value, gap2, x, z = extractSolution(n, m, x_vars, b_vars)

    end = time.time()

    return {'obj_value': obj_value, 'time': end - start, 'optimality gap': gap2, 'x': x, 'z': z}
