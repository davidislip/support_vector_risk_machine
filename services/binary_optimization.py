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

    z_vars = m.addMVar(n, vtype=gp.GRB.BINARY, name="z_vars")

    x_vars = m.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="x_vars")

    m.setObjective(x_vars @ Q @ x_vars, gp.GRB.MINIMIZE)

    m.addConstr(x_vars.sum() == 1)

    m.addConstr(mu @ x_vars >= targetRet)

    m.addConstr(z_vars.sum() <= K)

    m.addConstr(x_vars <= z_vars)

    return m, x_vars, z_vars


def extractSolution(n, m, x_vars, z_vars):
    x = np.zeros(n)
    z = np.zeros(n)
    for i in range(n):
        x[i] = x_vars[i].X
        z[i] = z_vars[i].X
    obj_value = m.objVal
    gap1 = m.MIPGap
    gap2 = gap1 * 100
    return obj_value, gap2, x, z


def CardMVO(mu, Q, K, targetRet, limit_time=30, MipGap=0.01):
    n = len(mu)
    mu = mu.squeeze()

    start = time.time()

    m, x_vars, z_vars = CreateCardMVOModel(mu, targetRet, Q, K, limit_time, MipGap)
    m.optimize()
    obj_value, gap2, x, z = extractSolution(n, m, x_vars, z_vars)

    end = time.time()
    return {'obj_value': obj_value, 'time': end - start, 'optimality gap': gap2, 'x': x, 'z': z}


def addTurnoverConstraints(m, x_vars, previous_portfolio, turnover_limit):
    n = len(previous_portfolio)
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


def CardMVOTurnover(mu, Q, K, targetRet, turnover_limit, previous_portfolio, limit_time=30, MipGap=0.01):
    n = len(mu)
    mu = mu.squeeze()

    start = time.time()

    m, x_vars, z_vars = CreateCardMVOModel(mu, targetRet, Q, K, limit_time, MipGap)

    if previous_portfolio is not None:
        absolute_delta = addTurnoverConstraints(m, x_vars, previous_portfolio, turnover_limit)

    m.optimize()

    # model did not solve
    if m.status in (4, 3):
        smallestTurnoverModel(m, absolute_delta)

    obj_value, gap2, x, z = extractSolution(n, m, x_vars, z_vars)

    end = time.time()

    return {'obj_value': obj_value, 'time': end - start, 'optimality gap': gap2, 'x': x, 'z': z}


def CreateSVMMVOModel(mu, targetRet, Q, K, q, epsilon,
                      bigM, big_w,
                      period_Context, C=1,
                      separable=False, limit_time=30, MipGap=0.01):
    n, p = period_Context.shape

    m = gp.Model("miqp")

    m.Params.timeLimit = limit_time

    m.Params.MIPGap = MipGap

    # binary variables
    z_vars = m.addMVar(n, vtype=gp.GRB.BINARY, name="z_vars")

    t_vars = m.addMVar(p, vtype=gp.GRB.BINARY, name="t_vars")

    x_vars = m.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="x_vars")

    w_vars = m.addMVar(p, lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="w_vars")

    b_var = m.addMVar(shape=1, lb=-1 * GRB.INFINITY, name="b_var", vtype=gp.GRB.CONTINUOUS)

    m.addConstr(x_vars.sum() == 1)

    m.addConstr(mu @ x_vars >= targetRet)

    m.addConstr(z_vars.sum() <= K)

    m.addConstr(x_vars <= z_vars)

    if separable:
        xi_vars = None
        m.setObjective(x_vars @ Q @ x_vars + (epsilon / 2) * (w_vars @ w_vars), gp.GRB.MINIMIZE)
        for i in range(n):
            y_i = period_Context.iloc[i].values
            m.addConstr(w_vars @ y_i + b_var <= -1 + bigM * z_vars[i], name="svm1")
            m.addConstr(w_vars @ y_i + b_var >= 1 + bigM * (1 - z_vars[i]), name="svm2")
    else:
        xi_vars = m.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="xi_vars")
        C_epsilon_by_n = C * epsilon / n
        m.setObjective(x_vars @ Q @ x_vars + (epsilon / 2) * (w_vars @ w_vars) + C_epsilon_by_n * xi_vars.sum(),
                       gp.GRB.MINIMIZE)
        for i in range(n):
            y_i = period_Context.iloc[i].values
            m.addConstr(w_vars @ y_i + b_var <= -1 + xi_vars[i] + bigM * z_vars[i], name="svm1")
            m.addConstr(w_vars @ y_i + b_var >= 1 - xi_vars[i] + bigM * (1 - z_vars[i]), name="svm2")

    m.addConstr(w_vars <= big_w * t_vars)

    m.addConstr(w_vars >= -1 * big_w * t_vars)

    m.addConstr(t_vars.sum() <= q)

    return m, x_vars, z_vars, w_vars, b_var, t_vars, xi_vars


def corollary1(period_Context, q, big_w, big_xi):
    """
    Follows Bianco et.al and assumes ||w, b|| \leq 1
    big_w = 1
    big_b = 1
    bigM = big_w + big_b + \max{i = 1...N} ||\boldsymbol{y}^{(i)}||_1
    """
    n, p = period_Context.shape
    largest_abs = 0
    pairs = period_Context.values - period_Context.values[:, None]
    pairs = np.abs(pairs)
    for i in range(n):
        for j in range(i):
            candidate = np.sort(pairs[i, j])[-1 * q:].sum()
            if candidate > largest_abs:
                largest_abs = candidate
    bigM = big_w * largest_abs + max(2, big_xi)

    return bigM


def corollary2(period_Context, big_w, big_xi):
    """
    Follows Bianco et.al and assumes ||w, b|| \leq 1
    big_w = 1
    big_b = 1
    bigM = big_w + big_b + \max{i = 1...N} ||\boldsymbol{y}^{(i)}||_1
    """
    n, p = period_Context.shape
    largest_abs = 0
    pairs = period_Context.values - period_Context.values[:, None]
    pairs = np.abs(pairs)
    largest_abs = np.abs(pairs).sum(axis=-1).max()
    bigM = big_w * largest_abs + max(2, big_xi)

    return bigM


def corollary3(period_Context, q, big_w, big_b):
    """
    Follows Bianco et.al and assumes ||w, b|| \leq 1
    big_w = 1
    big_b = 1
    bigM = big_w + big_b + \max{i = 1...N} ||\boldsymbol{y}^{(i)}||_1
    """
    n, p = period_Context.shape
    largest_abs = 0
    for i in range(n):
        candidate = period_Context.iloc[i].abs().sort_values(ascending=False).iloc[:q].sum()
        if candidate > largest_abs:
            largest_abs = candidate
    bigM = 1 + big_b + big_w * largest_abs

    return bigM


def corollary4(period_Context, big_w, big_b):
    """
    Follows Bianco et.al and assumes ||w, b|| \leq 1
    big_w = 1
    big_b = 1
    bigM = big_w + big_b + \max{i = 1...N} ||\boldsymbol{y}^{(i)}||_1
    """
    bigM = 1 + big_b + big_w * period_Context.abs().sum(axis=1).max()
    return bigM


def naivebigMStrategyCorollary4(period_Context):
    """
    Follows Bianco et.al and assumes ||w, b|| \leq 1
    big_w = 1
    big_b = 1
    bigM = big_w + big_b + \max{i = 1...N} ||\boldsymbol{y}^{(i)}||_1
    """
    big_w = 1
    big_b = 1
    bigM = corollary4(period_Context, big_w, big_b)
    big_xi = None
    return bigM, big_w, big_b, big_xi


def naivebigMStrategyCorollary3(period_Context, q):
    """
    Follows Bianco et.al and assumes ||w, b|| \leq 1
    big_w = 1
    big_b = 1
    bigM = big_w + big_b + \max{i = 1...N} ||\boldsymbol{y}^{(i)}||_1
    """
    big_w = 1
    big_b = 1
    bigM = corollary3(period_Context, q, big_w, big_b)
    big_xi = None
    return bigM, big_w, big_b, big_xi

def ConstructFeasibleSolution():
    ObjSVM = 10
    return ObjSVM

def objectiveBigMStrategy():
    ObjSVM = ConstructFeasibleSolution()

    big_w = ObjSVM
    big_xi = ObjSVM
    bigM = corollary1()
    big_b = None
    return bigM, big_w, big_b, big_xi

def objectiveBigMStrategyTightening():
    ObjSVM = ConstructFeasibleSolution()

    big_w = ObjSVM
    big_xi = ObjSVM
    bigM = corollary1()
    # min max big_b
    # update big_b
    # min max big xi with bound on b added
    # update big_xi
    # check using corrollary 3
    # update M
    # Solve SVMMVO using Gurobi?
    # update big_w, big_xi
    # bounds improved? Keep going

def SVMMVO(mu, targetRet, Q, K, q, epsilon, period_Context, C,
           separable, bigMStrategy, bigMStrategy_args, limit_time=30, MipGap=0.01):
    n = len(mu)
    mu = mu.squeeze()

    start = time.time()
    # compute big M here

    bigM, big_w, big_b, big_xi = bigMStrategy(mu, targetRet, Q, K, q, epsilon, period_Context, C, separable,
                                              limit_time=limit_time, MipGap=MipGap)

    m, x_vars, z_vars, w_vars, b_var, t_vars, xi_vars = CreateSVMMVOModel(mu, targetRet, Q, K, q, epsilon,
                                                                          bigM, big_w,
                                                                          period_Context, C=C,
                                                                          separable=separable, limit_time=limit_time,
                                                                          MipGap=MipGap)
    m.optimize()
    obj_value, gap2, x, z = extractSolution(n, m, x_vars, z_vars)

    end = time.time()
    return {'obj_value': obj_value, 'time': end - start, 'optimality gap': gap2, 'x': x, 'z': z}
