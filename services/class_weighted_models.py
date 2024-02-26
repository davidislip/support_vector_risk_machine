import gurobipy as gp
from gurobipy import GRB
import time
from services.data_utils import *
from sklearn.svm import SVC
import warnings
from sklearn.metrics import hinge_loss


def CreateClassWgtSVMMVOModel(mu, targetRet, Q, K, q, epsilon,
                              bigM, big_w_inf,
                              period_Context, C, separable, limit_time, MipGap, LogToConsole, class_weights):
    n, p = period_Context.shape

    m = gp.Model("miqp")

    m.Params.timeLimit = limit_time
    m.Params.FeasibilityTol = 1e-8
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
        xi_plus_vars = None
        xi_neg_vars = None
        m.setObjective(x_vars @ Q @ x_vars + (epsilon / 2) * (w_vars @ w_vars), gp.GRB.MINIMIZE)
        for i in range(n):
            y_i = period_Context.iloc[i].values
            m.addConstr(w_vars @ y_i + b_var <= -1 + bigM * z_vars[i], name="svm1")
            m.addConstr(w_vars @ y_i + b_var >= 1 + bigM * (1 - z_vars[i]), name="svm2")
    else:
        xi_plus_vars = m.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="xi_plus_vars")
        xi_neg_vars = m.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="xi_neg_vars")
        C_epsilon_by_n = C * epsilon / n
        m.setObjective(x_vars @ Q @ x_vars + (epsilon / 2) * (w_vars @ w_vars) + C_epsilon_by_n * class_weights[
            1] * xi_plus_vars.sum()
                       + class_weights[0] * C_epsilon_by_n * xi_neg_vars.sum(),
                       gp.GRB.MINIMIZE)
        for i in range(n):
            y_i = period_Context.iloc[i].values
            m.addConstr(w_vars @ y_i + b_var <= -1 + xi_neg_vars[i] + bigM * z_vars[i], name="svm1")
            m.addConstr(w_vars @ y_i + b_var >= 1 - xi_plus_vars[i] - bigM * (1 - z_vars[i]), name="svm2")

    m.addConstr(w_vars <= big_w_inf * t_vars)

    m.addConstr(w_vars >= -1 * big_w_inf * t_vars)

    m.addConstr(t_vars.sum() <= q)

    return m, x_vars, z_vars, w_vars, b_var, t_vars, xi_plus_vars, xi_neg_vars


def CreateClassWgtSVMModel(period_Context, z_vals, C, separable, limit_time, LogToConsole, class_weights):
    n, p = period_Context.shape

    m = gp.Model("miqp")

    m.Params.timeLimit = limit_time
    m.Params.FeasibilityTol = 1e-8
    m.Params.LogToConsole = int(LogToConsole)

    w_vars = m.addMVar(p, lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="w_vars")

    b_var = m.addMVar(shape=1, lb=-1 * GRB.INFINITY, name="b_var", vtype=gp.GRB.CONTINUOUS)

    if separable:
        xi_plus_vars = None
        xi_neg_vars = None
        m.setObjective((1 / 2) * (w_vars @ w_vars), gp.GRB.MINIMIZE)
        for i in range(n):
            y_i = period_Context.iloc[i].values
            m.addConstr((2 * z_vals[i] - 1) * (y_i @ w_vars + b_var) >= 1)
    else:
        xi_plus_vars = m.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="xi_plus_vars")
        xi_neg_vars = m.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="xi_neg_vars")
        C_by_n = C / n
        m.setObjective((1 / 2) * (w_vars @ w_vars) + C_by_n * class_weights[1] * xi_plus_vars.sum()
                       + class_weights[0] * C_by_n * xi_neg_vars.sum(), gp.GRB.MINIMIZE)
        for i in range(n):
            y_i = period_Context.iloc[i].values
            if z_vals[i] > 0.9:
                m.addConstr((2 * z_vals[i] - 1) * (y_i @ w_vars + b_var) + xi_plus_vars[i] >= 1)
            else:
                m.addConstr((2 * z_vals[i] - 1) * (y_i @ w_vars + b_var) + xi_neg_vars[i] >= 1)
    return m, w_vars, b_var, xi_plus_vars, xi_neg_vars


def extractSVMSolution(n, p, m, w_vars, b_var, xi_plus_vars, xi_neg_vars):
    w = np.zeros(p)
    b = np.zeros(1)
    xi_plus = np.zeros(n)
    xi_neg = np.zeros(n)
    for i in range(p):
        w[i] = w_vars[i].X
    b[0] = b_var[0].X

    for i in range(n):
        if xi_plus_vars is not None:
            xi_plus[i] = xi_plus_vars[i].X
        if xi_neg_vars is not None:
            xi_neg[i] = xi_neg_vars[i].X
    obj_value = m.objVal

    return obj_value, w, b, xi_plus, xi_neg


def ClassWgtSVM(period_Context, K, z_vals, C, separable, limit_time, LogToConsole):
    n, p = period_Context.shape
    class_weights = {0: K / n, 1: (n - K) / n}
    m, w_vars, b_var, xi_plus_vars, xi_neg_vars = CreateClassWgtSVMModel(period_Context, z_vals, C, separable,
                                                                         limit_time,
                                                                         LogToConsole, class_weights)
    start = time.time()
    m.optimize()
    end = time.time()
    obj_value, w, b, xi_plus, xi_neg = extractSVMSolution(n, p, m, w_vars, b_var, xi_plus_vars, xi_neg_vars)

    return {'obj_value': obj_value, 'time': end - start, 'w': w, 'b': b, 'xi_plus': xi_plus, 'xi_neg': xi_neg}


def sklearn_ClassWgtSVM(period_Context, K, z_vals, C, separable):
    if separable is True:
        warnings.warn("SVM is no longer separable")

    n, p = period_Context.shape
    class_weights = {-1: K / n, 1: (n - K) / n}
    svc = SVC(C=C / n, kernel='linear', class_weight=class_weights)
    u = 2 * z_vals - 1
    start = time.time()
    svc.fit(period_Context, u)
    end = time.time()
    w = np.squeeze(svc.coef_)
    b = svc.intercept_
    pred_decision = svc.decision_function(period_Context)
    margin = u * pred_decision
    xi = np.maximum(0, 1 - margin)
    pos_indices = z_vals > 0.9
    neg_indices = z_vals <= 0.9
    obj_value = (1 / 2) * np.power(w, 2).sum() + C / n * class_weights[1] * np.mean(xi[pos_indices]) \
                + C / n * class_weights[-1] * np.mean(xi[neg_indices])

    return {'obj_value': obj_value, 'time': end - start, 'w': w, 'b': b, 'xi': xi, 'svc': svc}


def CreateBestSubsetClassWgtSVMModel(period_Context, z_vals, C, separable, q, big_w2, limit_time, LogToConsole,
                                     class_weights):
    n, p = period_Context.shape

    m = gp.Model("miqp")

    m.Params.timeLimit = limit_time
    m.Params.FeasibilityTol = 1e-8
    m.Params.LogToConsole = int(LogToConsole)

    w_vars = m.addMVar(p, lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="w_vars")
    t_vars = m.addMVar(p, vtype=gp.GRB.BINARY, name="t_vars")

    b_var = m.addMVar(shape=1, lb=-1 * GRB.INFINITY, name="b_var", vtype=gp.GRB.CONTINUOUS)

    if separable:
        xi_plus_vars = None
        xi_neg_vars = None
        m.setObjective((1 / 2) * (w_vars @ w_vars), gp.GRB.MINIMIZE)
        for i in range(n):
            y_i = period_Context.iloc[i].values
            m.addConstr((2 * z_vals[i] - 1) * (y_i @ w_vars + b_var) >= 1)
    else:
        xi_plus_vars = m.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="xi_plus_vars")
        xi_neg_vars = m.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="xi_neg_vars")
        C_by_n = C / n
        m.setObjective((1 / 2) * (w_vars @ w_vars) + C_by_n * class_weights[1] * xi_plus_vars.sum()
                       + class_weights[0] * C_by_n * xi_neg_vars.sum(), gp.GRB.MINIMIZE)
        for i in range(n):
            y_i = period_Context.iloc[i].values
            if z_vals[i] > 0.9:
                m.addConstr((2 * z_vals[i] - 1) * (y_i @ w_vars + b_var) + xi_plus_vars[i] >= 1)
            else:
                m.addConstr((2 * z_vals[i] - 1) * (y_i @ w_vars + b_var) + xi_neg_vars[i] >= 1)

    m.addConstr(t_vars.sum() <= q)

    m.addConstr(w_vars <= big_w2 * t_vars)
    m.addConstr((-1) * w_vars <= big_w2 * t_vars)

    return m, w_vars, b_var, xi_plus_vars, xi_neg_vars, t_vars


def BestSubsetClassWgtSVM(period_Context, K, z_vals, C, separable, q, big_w2, limit_time, LogToConsole, warm_start):
    n, p = period_Context.shape
    class_weights = {0: K / n, 1: (n - K) / n}
    m, w_vars, b_var, xi_plus_vars, xi_neg_vars, t_vars = CreateBestSubsetClassWgtSVMModel(period_Context, z_vals, C,
                                                                                           separable,
                                                                                           q, big_w2,
                                                                                           limit_time, LogToConsole,
                                                                                           class_weights)

    w_vars.Start = warm_start['w_vals']
    b_var.Start = warm_start['b_val']
    t_vars.Start = warm_start['t_vals']

    start = time.time()
    m.optimize()
    end = time.time()
    obj_value, w, b, xi_plus, xi_neg = extractSVMSolution(n, p, m, w_vars, b_var, xi_plus_vars, xi_neg_vars)
    t = np.zeros(p)
    for i in range(p):
        t[i] = t_vars[i].X

    return {'obj_value': obj_value, 'time': end - start, 'w': w, 'b': b, 'xi_plus': xi_plus, 'xi_neg': xi_neg, 't': t}


def extractClassWgtSVMMVOSolution(n, p, m, x_vars, z_vars, w_vars, t_vars, b_var, xi_plus_vars, xi_neg_vars):
    x = np.zeros(n)
    z = np.zeros(n)
    w = np.zeros(p)
    t = np.zeros(p)
    b = np.zeros(1)
    xi_plus = np.zeros(n)
    xi_neg = np.zeros(n)
    for i in range(n):
        x[i] = x_vars[i].X
        z[i] = z_vars[i].X
        if xi_plus_vars is not None:
            xi_plus[i] = xi_plus_vars[i].X
        if xi_neg_vars is not None:
            xi_neg[i] = xi_neg_vars[i].X

    for i in range(p):
        w[i] = w_vars[i].X
        t[i] = t_vars[i].X
    b[0] = b_var[0].X

    obj_value = m.objVal
    gap1 = m.MIPGap
    gap2 = gap1 * 100
    return obj_value, gap2, x, z, w, t, b, xi_plus, xi_neg


def smallestTurnoverModelClassWgtSVMMVO(m, n, absolute_delta, separable, w_vars, xi_plus_vars, xi_neg_vars, epsilon, C,
                                        class_weights):
    # compute the smallest turnover portfolio instead of smallest risk with turnover constraints

    m.remove(m.getConstrByName('turnover'))
    m.reset()

    if separable:
        m.setObjective(absolute_delta.sum() + (epsilon / 2) * (w_vars @ w_vars), gp.GRB.MINIMIZE)
    else:
        C_epsilon_by_n = C * epsilon / n
        m.setObjective(absolute_delta.sum() + (epsilon / 2) * (w_vars @ w_vars) + C_epsilon_by_n * class_weights[
            1] * xi_plus_vars.sum()
                       + class_weights[0] * C_epsilon_by_n * xi_neg_vars.sum(),
                       gp.GRB.MINIMIZE)
    m.optimize()


from services.binary_optimization import SVMMVO_check_bigM, addTurnoverConstraints


def ClassWgtSVMMVO(limit_time=30, MipGap=0.01, LogToConsole=True, Verbose=True, SolutionLimit=GRB.MAXINT,
                   user_big_m=None,
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

    class_weights = {0: K / n, 1: (n - K) / n}
    if Verbose:
        print("Class weighted SVM")

    start = time.time()

    # compute big M here
    big_M_results, bigM, big_w_inf, big_b, big_xi = SVMMVO_check_bigM(user_big_m, LogToConsole,
                                                                      bigMStrategy, Verbose, kwargs)

    # if epsilon and C in big_M_results --> update epsilon and C: kappa
    if 'epsilon' in big_M_results.keys() and 'C' in big_M_results.keys():
        C, epsilon = big_M_results['C'], big_M_results['epsilon']
        if Verbose:
            print("C and epsilon updated from big M")

    if 'q' in big_M_results.keys():
        q = big_M_results['q']
        if Verbose:
            print("q updated from big M", q)

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

    m, x_vars, z_vars, w_vars, b_var, t_vars, xi_plus_vars, xi_neg_vars = CreateClassWgtSVMMVOModel(mu, targetRet, Q, K,
                                                                                                    q,
                                                                                                    epsilon,
                                                                                                    bigM, big_w_inf,
                                                                                                    period_Context, C=C,
                                                                                                    separable=separable,
                                                                                                    limit_time=limit_time,
                                                                                                    MipGap=MipGap,
                                                                                                    LogToConsole=LogToConsole,
                                                                                                    class_weights=class_weights)

    m.Params.SolutionLimit = SolutionLimit
    if 'warm_start' in big_M_results.keys():
        warm_start = big_M_results['warm_start']
        x_vars.Start = warm_start['x_vals']
        z_vars.Start = warm_start['z_vals']
        w_vars.Start = warm_start['w_vals']
        b_var.Start = warm_start['b_val']
        t_vars.Start = warm_start['t_vals']
        # xi_vars.Start = warm_start['xi_vals']
        # print("Warm Start")
        # m.Params.LogToConsole = True

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
            smallestTurnoverModelClassWgtSVMMVO(m, n, absolute_delta, separable, w_vars,
                                                xi_plus_vars, xi_neg_vars, epsilon, C, class_weights)

    obj_value, gap2, x, z, w, t, b, xi_plus, xi_neg = extractClassWgtSVMMVOSolution(n, p, m, x_vars, z_vars, w_vars,
                                                                                    t_vars, b_var, xi_plus_vars,
                                                                                    xi_neg_vars)
    if Verbose:
        print("SVM MVO Objective Value ", obj_value)
        print("Norm of w ", np.power(w, 2).sum())
        print("Positive Classification errors ", np.sum(xi_plus))
        print("Negative Classification errors ", np.sum(xi_neg))
    end = time.time()
    # m.Params.LogToConsole = False
    return {'obj_value': obj_value, 'time': end - start, 'bigM_time': bigM_finish_time - start, 'optimality gap': gap2,
            'x': x, 'z': z, 'w': w, 't': t, 'b': b, 'xi_plus': xi_plus, 'xi_neg': xi_neg,
            'feasible_solution': feasible_solution,
            'C': C, 'epsilon': epsilon, 'q':q}
