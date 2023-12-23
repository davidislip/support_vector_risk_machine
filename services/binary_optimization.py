import math
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


def CardMVO(limit_time=30, MipGap=0.01, SolutionLimit=GRB.MAXINT, **kwargs):
    mu, Q, K, targetRet = kwargs['mu'], kwargs['Q'], kwargs['K'], kwargs['targetRet']
    turnover_constraints = kwargs['turnover_constraints']
    previous_portfolio = kwargs['previous_portfolio']
    if turnover_constraints:
        turnover_limit = kwargs['turnover_limit']

    feasible_solution = True

    n = len(mu)
    mu = mu.squeeze()

    start = time.time()

    m, x_vars, z_vars = CreateCardMVOModel(mu, targetRet, Q, K, limit_time, MipGap)

    m.Params.SolutionLimit = SolutionLimit

    if previous_portfolio is not None and turnover_constraints:
        absolute_delta = addTurnoverConstraints(m, x_vars, previous_portfolio, turnover_limit)

    m.optimize()

    # model did not solve
    if m.status in (4, 3):
        feasible_solution = False
        if previous_portfolio is not None and turnover_constraints:
            smallestTurnoverModel(m, absolute_delta)

    obj_value, gap2, x, z = extractSolution(n, m, x_vars, z_vars)

    end = time.time()

    return {'obj_value': obj_value, 'time': end - start,
            'optimality gap': gap2, 'x': x, 'z': z, 'feasible_solution': feasible_solution}


def CreateSVMMVOModel(mu, targetRet, Q, K, q, epsilon,
                      bigM, big_w,
                      period_Context, C, separable, limit_time, MipGap):
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
            m.addConstr(w_vars @ y_i + b_var >= 1 - xi_vars[i] - bigM * (1 - z_vars[i]), name="svm2")

    m.addConstr(w_vars <= big_w * t_vars)

    m.addConstr(w_vars >= -1 * big_w * t_vars)

    m.addConstr(t_vars.sum() <= q)

    return m, x_vars, z_vars, w_vars, b_var, t_vars, xi_vars


def corollary1(period_Context, q, big_w, big_xi):
    """
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
    """
    bigM = 1 + big_b + big_w * period_Context.abs().sum(axis=1).max()
    return bigM


def CreateSVMModel(period_Context, z_vals, C, separable, limit_time):
    n, p = period_Context.shape

    m = gp.Model("miqp")

    m.Params.timeLimit = limit_time

    w_vars = m.addMVar(p, lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="w_vars")

    b_var = m.addMVar(shape=1, lb=-1 * GRB.INFINITY, name="b_var", vtype=gp.GRB.CONTINUOUS)

    if separable:
        xi_vars = None
        m.setObjective((1 / 2) * (w_vars @ w_vars), gp.GRB.MINIMIZE)
        for i in range(n):
            y_i = period_Context.iloc[i].values
            m.addConstr((2 * z_vals[i] - 1) * (y_i @ w_vars + b_var) >= 1)
    else:
        xi_vars = m.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="xi_vars")
        C_by_n = C / n
        m.setObjective((1 / 2) * (w_vars @ w_vars) + C_by_n * xi_vars.sum(), gp.GRB.MINIMIZE)
        for i in range(n):
            y_i = period_Context.iloc[i].values
            m.addConstr((2 * z_vals[i] - 1) * (y_i @ w_vars + b_var) + xi_vars[i] >= 1)
    return m, w_vars, b_var, xi_vars


def extractSVMSolution(n, p, m, w_vars, b_var, xi_vars):
    w = np.zeros(p)
    b = np.zeros(1)
    xi = np.zeros(n)
    for i in range(p):
        w[i] = w_vars[i].X
    b[0] = b_var[0].X

    for i in range(n):
        if xi_vars is not None:
            xi[i] = xi_vars[i].X

    obj_value = m.objVal

    return obj_value, w, b, xi


def SVM(period_Context, z_vals, C, separable, limit_time):
    n, p = period_Context.shape

    start = time.time()

    m, w_vars, b_var, xi_vars = CreateSVMModel(period_Context, z_vals, C, separable, limit_time)

    m.optimize()

    obj_value, w, b, xi = extractSVMSolution(n, p, m, w_vars, b_var, xi_vars)

    end = time.time()
    return {'obj_value': obj_value, 'time': end - start, 'w': w, 'b': b, 'xi': xi}


# Big M strategies begin

def naivebigMStrategyCorollary3(**kwargs):
    """
    Follows Bianco et.al and assumes ||w, b|| \leq 1
    big_w = 1
    big_b = 1
    bigM = big_w + big_b +
    \max_{\mathcal{T} \subset [p]: |\mathcal{T}| \leq q} \max{i = 1...N} ||\boldsymbol{y}_{\mathcal{T}}^{(i)}||_1
    """
    period_Context, q = kwargs['period_Context'], kwargs['q']
    big_w = 1
    big_b = 1
    bigM = corollary3(period_Context, q, big_w, big_b)
    big_xi = None
    return bigM, big_w, big_b, big_xi


def naivebigMStrategyCorollary4(**kwargs):
    """
    Follows Bianco et.al and assumes ||w, b|| \leq 1
    big_w = 1
    big_b = 1
    bigM = big_w + big_b + \max{i = 1...N} ||\boldsymbol{y}^{(i)}||_1
    """
    period_Context = kwargs['period_Context']
    big_w = 1
    big_b = 1
    bigM = corollary4(period_Context, big_w, big_b)
    big_xi = None
    return bigM, big_w, big_b, big_xi


def ConstructFeasibleSolution(bigM_limit_time=10, bigM_MipGap=0.1, bigM_SolutionLimit=10, **kwargs):
    period_Context, C, separable = kwargs['period_Context'], kwargs['C'], kwargs['separable']
    q, epsilon = kwargs['q'], kwargs['epsilon']

    # take out args into new dict

    bigM_kwargs_forCardMVO = kwargs.copy()
    bigM_kwargs_forCardMVO['limit_time'] = bigM_limit_time
    bigM_kwargs_forCardMVO['MipGap'] = bigM_MipGap
    bigM_kwargs_forCardMVO['SolutionLimit'] = bigM_SolutionLimit

    # card mvo
    card_mvo_results = CardMVO(**bigM_kwargs_forCardMVO)
    z_vals = card_mvo_results['z']
    # SVM
    svm_phase1_results = SVM(period_Context, z_vals, C, separable, bigM_limit_time)
    # sort and clip w
    w_vals = svm_phase1_results['w']
    abs_w = np.abs(w_vals)
    q_largest = np.argpartition(abs_w, (-1) * q)[-q:]
    # restrict indices
    period_Context_subset = period_Context.iloc[:, q_largest]
    # SVM again
    svm_phase2_results = SVM(period_Context_subset, z_vals, C, separable, bigM_limit_time)
    # Calculate Objective Value
    ObjSVM = svm_phase2_results['obj_value']
    ObjMVO = card_mvo_results['obj_value']

    return ObjMVO + epsilon * ObjSVM, card_mvo_results['feasible_solution']


def objectiveBigMStrategy(**kwargs):
    period_Context, epsilon, C, q = kwargs['period_Context'], kwargs['epsilon'], kwargs['C'], kwargs['q']

    ObjSVMMVO, feasible_solution = ConstructFeasibleSolution(
        **kwargs)  # kwargs may or may not have big M limit times etc

    n, p = period_Context.shape
    big_w = math.sqrt(ObjSVMMVO / epsilon)
    big_xi = n * ObjSVMMVO / (epsilon * C)
    bigM = corollary1(period_Context, q, big_w, big_xi)
    big_b = None
    return bigM, big_w, big_b, big_xi


def objectiveBigMStrategyTightening():
    # ObjSVM = ConstructFeasibleSolution()
    #
    # big_w = ObjSVM
    # big_xi = ObjSVM
    # bigM = corollary1()
    return None
    # min max big_b
    # update big_b
    # min max big xi with bound on b added
    # update big_xi
    # check using corrollary 3
    # update M
    # Solve SVMMVO using Gurobi?
    # update big_w, big_xi
    # bounds improved? Keep going


def extractSVMMVOSolution(n, p, m, x_vars, z_vars, w_vars, t_vars, b_var, xi_vars):
    x = np.zeros(n)
    z = np.zeros(n)
    w = np.zeros(p)
    t = np.zeros(p)
    b = np.zeros(1)
    xi = np.zeros(n)

    for i in range(n):
        x[i] = x_vars[i].X
        z[i] = z_vars[i].X
        if xi_vars is not None:
            xi[i] = xi_vars[i].X

    for i in range(p):
        w[i] = w_vars[i].X
        t[i] = t_vars[i].X
    b[0] = b_var[0].X

    obj_value = m.objVal
    gap1 = m.MIPGap
    gap2 = gap1 * 100
    return obj_value, gap2, x, z, w, t, b, xi


def smallestTurnoverModelSVMMVO(m, n, absolute_delta, separable, w_vars, xi_vars, epsilon, C):
    # compute the smallest turnover portfolio instead of smallest risk with turnover constraints

    m.remove(m.getConstrByName('turnover'))
    m.reset()

    if separable:
        m.setObjective(absolute_delta.sum() + (epsilon / 2) * (w_vars @ w_vars), gp.GRB.MINIMIZE)
    else:
        C_epsilon_by_n = C * epsilon / n
        m.setObjective(absolute_delta.sum() + (epsilon / 2) * (w_vars @ w_vars) + C_epsilon_by_n * xi_vars.sum(),
                       gp.GRB.MINIMIZE)

    m.optimize()


def SVMMVO(limit_time=30, MipGap=0.01, **kwargs):  # if kwargs does not have limit time and mipgap then

    mu, targetRet, Q, K = kwargs['mu'], kwargs['targetRet'], kwargs['Q'], kwargs['K']
    q, epsilon, period_Context = kwargs['q'], kwargs['epsilon'], kwargs['period_Context']
    C, separable = kwargs['C'], kwargs['separable']
    bigMStrategy = kwargs['bigMStrategy']
    turnover_constraints = kwargs['turnover_constraints']
    previous_portfolio = kwargs['previous_portfolio']
    if turnover_constraints:
        turnover_limit = kwargs['turnover_limit']

    feasible_solution = True

    n, p = period_Context.shape
    mu = mu.squeeze()

    start = time.time()
    # compute big M here
    bigM, big_w, big_b, big_xi = bigMStrategy(**kwargs)

    bigM_finish_time = time.time()

    m, x_vars, z_vars, w_vars, b_var, t_vars, xi_vars = CreateSVMMVOModel(mu, targetRet, Q, K, q, epsilon,
                                                                          bigM, big_w,
                                                                          period_Context, C=C,
                                                                          separable=separable, limit_time=limit_time,
                                                                          MipGap=MipGap)

    if previous_portfolio is not None and turnover_constraints:  # add turnover constraints
        absolute_delta = addTurnoverConstraints(m, x_vars, previous_portfolio, turnover_limit)

    m.optimize()  # try to solve
    # model did not solve
    if m.status in (4, 3):
        feasible_solution = False

        if previous_portfolio is not None and turnover_constraints:
            smallestTurnoverModelSVMMVO(m, n, absolute_delta, separable, w_vars, xi_vars, epsilon, C)

    obj_value, gap2, x, z, w, t, b, xi = extractSVMMVOSolution(n, p, m, x_vars, z_vars, w_vars, t_vars, b_var, xi_vars)

    end = time.time()
    return {'obj_value': obj_value, 'time': end - start, 'bigM_time': bigM_finish_time - start, 'optimality gap': gap2,
            'x': x, 'z': z, 'w': w, 't': t, 'b': b, 'xi': xi, 'feasible_solution': feasible_solution}
