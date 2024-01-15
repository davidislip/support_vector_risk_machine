import gurobipy as gp
from gurobipy import GRB
import time
from services.data_utils import *


# Cardinality Constrained Optimization
def CreateCardMVOModel(mu, targetRet, Q, K, limit_time=30, MipGap=0.01, LogToConsole=True):
    n = len(mu)

    m = gp.Model("miqp")

    m.Params.timeLimit = limit_time

    m.Params.MIPGap = MipGap

    m.Params.LogToConsole = int(LogToConsole)

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
                               vtype=gp.GRB.CONTINUOUS, name="magnitude of portfolio changes")
    m.addConstr(absolute_delta >= x_vars - previous_portfolio)
    m.addConstr(absolute_delta >= previous_portfolio - x_vars)
    m.addConstr(absolute_delta.sum() <= turnover_limit, name='turnover')
    m.update()
    return absolute_delta


def smallestTurnoverModel(m, absolute_delta):
    # compute the smallest turnover portfolio instead of the smallest risk with turnover constraints
    m.remove(m.getConstrByName('turnover'))
    m.reset()
    m.setObjective(absolute_delta.sum())
    m.optimize()


def CardMVO(limit_time=30, MipGap=0.01, SolutionLimit=GRB.MAXINT,
            LogToConsole=True, Verbose=True, **kwargs):
    mu, Q, K, targetRet = kwargs['mu'], kwargs['Q'], kwargs['K'], kwargs['targetRet']
    turnover_constraints = kwargs['turnover_constraints']
    previous_portfolio = kwargs['previous_portfolio']
    if turnover_constraints:
        turnover_limit = kwargs['turnover_limit']

    feasible_solution = True

    n = len(mu)
    mu = mu.squeeze()

    start = time.time()

    m, x_vars, z_vars = CreateCardMVOModel(mu, targetRet, Q, K, limit_time, MipGap, LogToConsole)

    m.Params.SolutionLimit = SolutionLimit

    if previous_portfolio is not None and turnover_constraints:
        absolute_delta = addTurnoverConstraints(m, x_vars, previous_portfolio, turnover_limit)

    if Verbose:
        print("-"*20)
        print("Solving Card MVO...")
    m.optimize()

    # model did not solve
    if m.status in (4, 3):
        feasible_solution = False
        if previous_portfolio is not None and turnover_constraints:
            if Verbose:
                print("Computing lowest turnover feasible solution...")
            smallestTurnoverModel(m, absolute_delta)

    obj_value, gap2, x, z = extractSolution(n, m, x_vars, z_vars)

    end = time.time()
    if Verbose:
        print("Card MVO Finished...")
        print("-" * 20)

    return {'obj_value': obj_value, 'time': end - start,
            'optimality gap': gap2, 'x': x, 'z': z, 'feasible_solution': feasible_solution}


def CreateSVMMVOModel(mu, targetRet, Q, K, q, epsilon,
                      bigM, big_w_inf,
                      period_Context, C, separable, limit_time, MipGap, LogToConsole):
    n, p = period_Context.shape

    m = gp.Model("miqp")

    m.Params.timeLimit = limit_time

    m.Params.MIPGap = MipGap

    m.Params.LogToConsole = int(LogToConsole)

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

    m.addConstr(w_vars <= big_w_inf * t_vars)

    m.addConstr(w_vars >= -1 * big_w_inf * t_vars)

    m.addConstr(t_vars.sum() <= q)

    return m, x_vars, z_vars, w_vars, b_var, t_vars, xi_vars


def CreateSVMModel(period_Context, z_vals, C, separable, limit_time, LogToConsole):
    n, p = period_Context.shape

    m = gp.Model("miqp")

    m.Params.timeLimit = limit_time

    m.Params.LogToConsole = int(LogToConsole)

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


def SVM(period_Context, z_vals, C, separable, limit_time, LogToConsole):
    n, p = period_Context.shape

    start = time.time()

    m, w_vars, b_var, xi_vars = CreateSVMModel(period_Context, z_vals, C, separable, limit_time, LogToConsole)

    m.optimize()

    obj_value, w, b, xi = extractSVMSolution(n, p, m, w_vars, b_var, xi_vars)

    end = time.time()
    return {'obj_value': obj_value, 'time': end - start, 'w': w, 'b': b, 'xi': xi}


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


def SVMMVO(limit_time=30, MipGap=0.01, LogToConsole=True, Verbose=True, SolutionLimit=GRB.MAXINT, user_big_m=None,
           **kwargs):  # if kwargs does not have limit time and mipgap then

    mu, targetRet, Q, K, q, epsilon, period_Context, C, separable = unpack_kwargs(kwargs)
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
    if user_big_m is None:
        bigM_kwargs = kwargs.copy()
        bigM_kwargs['LogToConsole'] = LogToConsole
        bigM_kwargs['Verbose'] = Verbose

        big_M_results = bigMStrategy(**bigM_kwargs)
        bigM, big_w_inf, big_b, big_xi = (big_M_results['bigM'], big_M_results['big_w_inf'],
                                          big_M_results['big_b'], big_M_results['big_xi'])
    else:
        assert type(user_big_m) == dict
        big_M_results = user_big_m
        bigM, big_w_inf, big_b, big_xi = user_big_m['bigM'], user_big_m['big_w_inf'], user_big_m['big_b'], user_big_m[
            'big_xi']

    # if the big M strategy yields feasibility information then
    # update the feasible solution flag
    if 'feasible_solution' in big_M_results.keys():
        feasible_solution = big_M_results['feasible_solution']

    if Verbose:
        print("Calculated Big M ", bigM)
        print("Calculated big W", big_w_inf)
        print("Calculated big b", big_b)
        print("Calculated big xi", big_xi)

    bigM_finish_time = time.time()

    m, x_vars, z_vars, w_vars, b_var, t_vars, xi_vars = CreateSVMMVOModel(mu, targetRet, Q, K, q, epsilon,
                                                                          bigM, big_w_inf,
                                                                          period_Context, C=C,
                                                                          separable=separable, limit_time=limit_time,
                                                                          MipGap=MipGap, LogToConsole=LogToConsole)
    m.Params.SolutionLimit = SolutionLimit

    if previous_portfolio is not None and turnover_constraints:  # add turnover constraints
        absolute_delta = addTurnoverConstraints(m, x_vars, previous_portfolio, turnover_limit)

    if feasible_solution:  # if we are feasible to the best of our knowledge
        m.optimize()  # try to solve
        # model did not solve
        if m.status in (4, 3):  # ah turns out we are not feasible
            feasible_solution = False  # update the flag
    else:  # feasible_solution is already set to false
        # it must be because of the turnover constraint
        # compute the min turnover portfolio that satisfies the constraint
        if previous_portfolio is not None and turnover_constraints:
            if Verbose:
                print("Calculating the closest turnover portfolio...")
            smallestTurnoverModelSVMMVO(m, n, absolute_delta, separable, w_vars, xi_vars, epsilon, C)

    obj_value, gap2, x, z, w, t, b, xi = extractSVMMVOSolution(n, p, m, x_vars, z_vars, w_vars, t_vars, b_var, xi_vars)
    if Verbose:
        print("SVM MVO Objective Value ", obj_value)
        print("Norm of w ", np.power(w, 2).sum())
        print("Classification errors ", np.sum(xi))
    end = time.time()
    return {'obj_value': obj_value, 'time': end - start, 'bigM_time': bigM_finish_time - start, 'optimality gap': gap2,
            'x': x, 'z': z, 'w': w, 't': t, 'b': b, 'xi': xi, 'feasible_solution': feasible_solution}
