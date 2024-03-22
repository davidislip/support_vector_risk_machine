from services.binary_optimization import *
from services.data_utils import *
import warnings
import math
from sklearn.model_selection import KFold


# Big M strategies begin
def ConstructFeasibleMVO(bigM_limit_time=10, bigM_MipGap=0.1, bigM_SolutionLimit=10, LogToConsole=True,
                         Verbose=True, **kwargs):
    # take out args into new dict
    if Verbose:
        print("-" * 20)
        print("Constructing Feasible Solution...")
    bigM_kwargs_forCardMVO = kwargs.copy()
    bigM_kwargs_forCardMVO['limit_time'] = bigM_limit_time
    bigM_kwargs_forCardMVO['MipGap'] = bigM_MipGap
    bigM_kwargs_forCardMVO['SolutionLimit'] = bigM_SolutionLimit
    bigM_kwargs_forCardMVO['LogToConsole'] = LogToConsole
    bigM_kwargs_forCardMVO['Verbose'] = Verbose
    # card mvo
    if Verbose:
        print("Extracting a cardinality constrained portfolio...")
    card_mvo_results = CardMVO(**bigM_kwargs_forCardMVO)
    z_vals = card_mvo_results['z']
    x_vals = card_mvo_results['x']
    ObjMVO = card_mvo_results['obj_value']

    return ObjMVO, card_mvo_results['feasible_solution'], z_vals, x_vals


def ConstructFeasibleSolutionSVM(z_vals, period_Context, C, separable, q, bigM_limit_time, LogToConsole, Verbose):
    # take out args into new dict2
    # SVM

    if Verbose:
        print("Phase 1 SVM...")
    # svm_phase1_results = SVM(period_Context, z_vals, C, separable, bigM_limit_time, LogToConsole)
    # print(svm_phase1_results['obj_value'])
    # print(np.power(svm_phase1_results['w'], 2).sum())
    # print(svm_phase1_results['xi'].sum())
    # print(svm_phase1_results['time'])
    svm_phase1_results = sklearn_SVM(period_Context, z_vals, C, separable)
    # print("SKLEARN")
    # print(svm_phase1_results['obj_value'])
    # print(np.power(svm_phase1_results['w'], 2).sum())
    # print(svm_phase1_results['xi'].sum())
    print(svm_phase1_results['time'])
    # sort and clip w
    w_vals = svm_phase1_results['w']
    abs_w = np.abs(w_vals)
    q_largest = np.argpartition(abs_w, (-1) * q)[-q:]
    # restrict indices
    period_Context_subset = period_Context.iloc[:, q_largest]
    # SVM again
    if Verbose:
        print("Phase 2 SVM...")
    svm_phase2_results = sklearn_SVM(period_Context_subset, z_vals, C, separable)
    if Verbose:
        print("Feasible Solution Constructed")
        print("-" * 20)
    # Calculate Objective Value
    ObjSVM = svm_phase2_results['obj_value']
    w = svm_phase2_results['w']
    b = svm_phase2_results['b']

    return ObjSVM, w, b


def unpack_bigM_params(**kwargs):
    if 'bigM_limit_time' in kwargs.keys():
        bigM_limit_time = kwargs['bigM_limit_time']
    else:
        bigM_limit_time = 10
    if 'LogToConsole' in kwargs.keys():
        LogToConsole = kwargs['LogToConsole']
    else:
        LogToConsole = True
    if 'Verbose' in kwargs.keys():
        Verbose = kwargs['Verbose']
    else:
        Verbose = True
    return bigM_limit_time, LogToConsole, Verbose


def ConstructFeasibleSolution(**kwargs):
    period_Context, C, separable = kwargs['period_Context'], kwargs['C'], kwargs['separable']
    q, epsilon = kwargs['q'], kwargs['epsilon']

    # Do Card MVO
    ObjMVO, feasible_solution, z_vals, x_vals = ConstructFeasibleMVO(**kwargs)

    # SVM
    bigM_limit_time, LogToConsole, Verbose = unpack_bigM_params(**kwargs)

    ObjSVM, w, b = ConstructFeasibleSolutionSVM(z_vals, period_Context, C, separable, q,
                                                bigM_limit_time, LogToConsole, Verbose)

    return ObjMVO + epsilon * ObjSVM, feasible_solution


def ConstructFeasibleSolutionandHyperParams(**kwargs):
    period_Context, separable = kwargs['period_Context'], kwargs['separable']

    q = kwargs['q']
    kappa = kwargs['kappa']

    n, p = period_Context.shape
    # do card MVO
    ObjMVO, feasible_solution, z_vals, x_vals = ConstructFeasibleMVO(**kwargs)
    z_vals = np.rint(z_vals)
    u = 2 * z_vals - 1
    # SVM
    bigM_limit_time, LogToConsole, Verbose = unpack_bigM_params(**kwargs)

    # find the best C using k fold
    Cs = np.geomspace(2 ** (-12), 2 ** 12, 25)
    bestC = 2 ** (-12)
    lowest_error = n  # this is an upper bound on the error
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=20)
    error_list = []
    w_list = []
    norm_w = 0
    for C in Cs:
        errors = np.zeros(n_splits)
        for i, (train_index, test_index) in enumerate(kf.split(period_Context)):
            # print(f"Fold {i}:")
            # print(f"  Train: index={train_index}")
            # print(f"  Test:  index={test_index}")
            # solve SVM
            try:
                svm_phase1_results = sklearn_SVM(period_Context.iloc[train_index], z_vals[train_index], C, separable)
            except ValueError:
                if Verbose:
                    print("Value Error - sklearn . . . using Gurobi")
                svm_phase1_results = SVM(period_Context.iloc[train_index], z_vals[train_index], C, separable,
                                         bigM_limit_time, LogToConsole)
            w_vals = svm_phase1_results['w']
            abs_w = np.abs(w_vals)
            q_largest = np.argpartition(abs_w, (-1) * q)[-q:]
            # restrict indices
            period_Context_subset = period_Context.iloc[:, q_largest]
            try:
                svm_phase2_results = sklearn_SVM(period_Context_subset.iloc[train_index], z_vals[train_index], C,
                                                 separable)
                svc = svm_phase2_results['svc']
                pred_decision = svc.decision_function(period_Context_subset.iloc[test_index])
            except ValueError:
                svm_phase2_results = SVM(period_Context_subset.iloc[train_index], z_vals[train_index], C, separable,
                                         bigM_limit_time, LogToConsole)
                w, b = svm_phase2_results['w'], svm_phase2_results['b']
                pred_decision = period_Context_subset.iloc[test_index].values @ svm_phase2_results['w'] + b
            margin = u[test_index] * pred_decision
            xi_test = np.maximum(0, 1 - margin)
            errors[i] = np.sum(xi_test)
        error_list.append(errors.mean())
        w_list.append(np.power(svm_phase2_results['w'], 2).sum())
        if errors.mean() < lowest_error:
            bestC = C
            if Verbose:
                print("New Best C ", bestC)
                print("Test Error ", errors.mean())
                print("Selected Features ", period_Context_subset.columns)
            lowest_error = errors.mean()
            norm_w = np.power(svm_phase2_results['w'], 2).sum()

    if norm_w > 10 ** (-7) and Verbose:
        print("Non degenerate solution found")
    # calculate ObjSVM and construct bound on ||w||_1
    try:
        svm_phase1_results = sklearn_SVM(period_Context, z_vals, bestC, separable)
    except ValueError:
        svm_phase1_results = SVM(period_Context, z_vals, bestC, separable, bigM_limit_time, LogToConsole)
    w_vals = svm_phase1_results['w']
    abs_w = np.abs(w_vals)
    q_largest = np.argpartition(abs_w, (-1) * q)[-q:]
    # restrict indices
    period_Context_subset = period_Context.iloc[:, q_largest]
    try:
        svm_phase2_results = sklearn_SVM(period_Context_subset, z_vals, bestC, separable)
    except ValueError:
        svm_phase2_results = SVM(period_Context_subset, z_vals, bestC, separable, bigM_limit_time, LogToConsole)
    # sqrt(2 ObjSVM) is the soln is big_w2 and hence big w
    ObjFeasibleSVM = svm_phase2_results['obj_value']
    big_w2 = min(math.sqrt(2 * ObjFeasibleSVM), math.sqrt(2 * bestC))
    # solve BSS SVM
    w_vals = np.zeros(p)
    t_vals = np.zeros(p)
    w_vals[q_largest] = svm_phase2_results['w']
    t_vals[q_largest] = np.ones(len(q_largest))
    #xi_vals = svm_phase2_results['xi'] + 10 ** (-8)
    b_val = svm_phase2_results['b']

    warm_start = {'w_vals': w_vals, 'b_val': b_val, 't_vals': t_vals}
    BestSubsetSVM_results = BestSubsetSVM(period_Context, z_vals, bestC, separable, q, big_w2, bigM_limit_time,
                                          LogToConsole, warm_start)
    ObjSVM = BestSubsetSVM_results['obj_value']
    # set epsilon so that the risk guarantee is satisfied
    epsilon = (kappa * ObjMVO * 1000) / (ObjSVM * 1000)  # this is the big line
    if Verbose:
        print("Largest epsilon value guaranteeing ", 1 + kappa, " risk: ", epsilon)
    warm_start = {'x_vals': x_vals, 'z_vals': z_vals,
                  'w_vals': BestSubsetSVM_results['w'], 't_vals': np.rint(BestSubsetSVM_results['t']),
                  'b_val': BestSubsetSVM_results['b'], 'xi_vals': BestSubsetSVM_results['xi']}
    return ObjMVO + epsilon * ObjSVM, feasible_solution, bestC, epsilon, warm_start


def size_of_largest_feature(period_Context, q):
    """
    M_{\mathcal{Y}}
    """
    n, p = period_Context.shape
    largest_abs = 0
    for i in range(n):
        candidate = math.sqrt(np.power(period_Context.iloc[i], 2).sort_values(ascending=False).iloc[:q].sum())
        if candidate > largest_abs:
            largest_abs = candidate
    return largest_abs


def largest_pairwise_distance(period_Context, q):
    """
    """
    n, p = period_Context.shape
    largest_abs_pdist = 0
    pairs = period_Context.values - period_Context.values[:, None]
    pairs = np.power(pairs, 2)
    for i in range(n):
        for j in range(i):
            candidate = math.sqrt(np.sort(pairs[i, j])[-1 * q:].sum())
            if candidate > largest_abs_pdist:
                largest_abs_pdist = candidate

    return largest_abs_pdist


def theorem1(ObjSVMMVO, n, epsilon, C, largest_abs):
    big_w_inf = min(math.sqrt(2*ObjSVMMVO / epsilon), math.sqrt(2 * C))
    big_w_2 = min(math.sqrt(2*ObjSVMMVO / epsilon), math.sqrt(2 * C))
    big_b = 1 + largest_abs * min(math.sqrt(2*ObjSVMMVO / epsilon), math.sqrt(2 * C))

    big_xi = n * ObjSVMMVO / (epsilon * C)
    xi_str = 'Lemma 2'
    if n <= big_xi:
        big_xi = n
        xi_str = 'Lemma 3'
    if 2 + 2 * largest_abs * math.sqrt(2*ObjSVMMVO / epsilon) <= big_xi:
        big_xi = 2 + 2 * largest_abs * math.sqrt(2*ObjSVMMVO / epsilon)
        xi_str = 'Lemma 5'

    return big_w_inf, big_w_2, big_b, big_xi, xi_str


def theorem2(largest_abs_pdist, big_w_2, big_xi):
    bigM = big_w_2 * largest_abs_pdist + max(2, big_xi)
    return bigM


def theorem3(largest_abs, big_b, big_w_2):
    bigM = 1 + big_b + big_w_2 * largest_abs
    return bigM


def naive_bigMStrategy(**kwargs):
    """
    Follows Bianco et.al and assumes ||w, b|| \leq 1
    big_w_inf = 1
    big_b = 1
    bigM = big_w_inf + big_b +
    \max_{\mathcal{T} \subset [p]: |\mathcal{T}| \leq q} \max{i = 1...N} ||\boldsymbol{y}_{\mathcal{T}}^{(i)}||_1
    """
    period_Context, q = kwargs['period_Context'], kwargs['q']
    big_w_inf = 1
    big_w_2 = 1
    big_b = 1
    largest_abs = size_of_largest_feature(period_Context, q)
    bigM = theorem3(largest_abs, big_b, big_w_2)
    big_xi = None
    return {'bigM': bigM, 'big_w_inf': big_w_inf, 'big_w_2': big_w_2, 'big_b': big_b, 'big_xi': big_xi}


def objectiveBigMStrategy(**kwargs):
    period_Context, epsilon, C, q = kwargs['period_Context'], kwargs['epsilon'], kwargs['C'], kwargs['q']
    Verbose = kwargs['Verbose']
    theorem3_bool = False

    if Verbose:
        print("-" * 20)
        print("Calculating Big M")
    start = time.time()
    ObjSVMMVO, feasible_solution = ConstructFeasibleSolution(
        **kwargs)  # kwargs may or may not have big M limit times etc
    # ObjSVMMVO, feasible_solution, C, epsilon = ConstructFeasibleSolutionandHyperParams(**kwargs)
    end = time.time()
    if Verbose:
        print("Feasible solution constructed in ", end - start, " seconds")
    n, p = period_Context.shape
    largest_abs = size_of_largest_feature(period_Context, q)
    largest_abs_pdist = largest_pairwise_distance(period_Context, q)
    big_w_inf, big_w_2, big_b, big_xi, xi_str = theorem1(ObjSVMMVO, n, epsilon, C, largest_abs)
    bigM = theorem2(largest_abs_pdist, big_w_2, big_xi)
    bigM_cand = theorem3(largest_abs, big_b, big_w_2)
    if Verbose:
        print("Xi Lemma ", xi_str)
        print("Theorem 2 big M ", bigM)
        print("Theorem 3 big M ", bigM_cand)
    if bigM_cand < bigM:
        theorem3_bool = True
        bigM = bigM_cand

    return {'bigM': bigM, 'big_w_inf': big_w_inf, 'big_w_2': big_w_2, 'big_b': big_b,
            'big_xi': big_xi, 'feasible_solution': feasible_solution, 'Theorem': theorem3_bool,
            'xi lemma': xi_str}


def get_SVMMVO_bigM_Vars(**kwargs):
    if 'SVMMVO_bigM_time_limit' in kwargs.keys():
        SVMMVO_bigM_time_limit = kwargs['SVMMVO_bigM_time_limit']
    else:
        SVMMVO_bigM_time_limit = 10

    if 'SVMMVO_MipGap' in kwargs.keys():
        SVMMVO_MipGap = kwargs['SVMMVO_MipGap']
    else:
        SVMMVO_MipGap = 0.1

    if 'SVMMVO_SolutionLimit' in kwargs.keys():
        SVMMVO_SolutionLimit = kwargs['SVMMVO_SolutionLimit']
    else:
        SVMMVO_SolutionLimit = 10

    return SVMMVO_bigM_time_limit, SVMMVO_MipGap, SVMMVO_SolutionLimit


def get_tightening_kwargs(**kwargs):
    if 'SkipSOCP' in kwargs.keys():
        SkipSOCP = kwargs['SkipSOCP']
    else:
        SkipSOCP = True
    if 'tightening_iter_lim' in kwargs.keys():
        tightening_iter_lim = kwargs['tightening_iter_lim']
    else:
        tightening_iter_lim = 1
    return SkipSOCP, tightening_iter_lim


def SOCP_step(bigM, big_w_inf, big_w_2, big_b,
              big_xi, ObjSVMMVO, feasible_solution, largest_abs_pdist, largest_abs,
              bigM_kwargs_for_SOCP):
    big_b_candidate, big_xi_candidate, SOCP_time_b, SOCP_time_xi = bounding_b_and_xi_socp(bigM, big_w_inf, big_b,
                                                                                          big_xi,
                                                                                          ObjSVMMVO, feasible_solution,
                                                                                          **bigM_kwargs_for_SOCP)
    bounds_improved_socp = False
    if big_b_candidate < big_b:
        big_b = big_b_candidate
        bounds_improved_socp = True
    if big_xi_candidate < big_xi:
        big_xi = big_xi_candidate
        bounds_improved_socp = True

    # update M
    if bounds_improved_socp:
        bigM = theorem2(largest_abs_pdist, big_w_2, big_xi)
        bigM_cand = theorem3(largest_abs, big_b, big_w_2)
        if bigM_cand < bigM:
            bigM = bigM_cand

    return bigM, big_b, big_xi, bounds_improved_socp


def relax_binary(vars):
    vars.setAttr('vtype', 'C')
    vars.setAttr('lb', 0)
    vars.setAttr('ub', 1)


def bounding_b_and_xi_socp(bigM, big_w_inf, big_b, big_xi, ObjSVMMVO, feasible_solution,
                           LogToConsole=True, MipGap=0.1, **kwargs):
    mu, targetRet, Q, K, q, epsilon, period_Context, C, separable = unpack_kwargs(kwargs)
    Verbose = kwargs['Verbose']

    if 'SOCP_limit_time' in kwargs.keys():
        limit_time = kwargs['SOCP_limit_time']
    else:
        limit_time = 30
        if Verbose:
            print("SOCP default solve time 30s...")

    turnover_constraints = kwargs['turnover_constraints']
    previous_portfolio = kwargs['previous_portfolio']

    if turnover_constraints:
        turnover_limit = kwargs['turnover_limit']

    mu = mu.squeeze()
    n = len(mu)
    m, x_vars, z_vars, w_vars, b_var, t_vars, xi_vars = CreateSVMMVOModel(mu, targetRet, Q, K, q, epsilon,
                                                                          bigM, big_w_inf,
                                                                          period_Context, C=C,
                                                                          separable=separable, limit_time=limit_time,
                                                                          MipGap=MipGap, LogToConsole=LogToConsole)
    # add b constraints and xi constraints
    if xi_vars is not None:
        for i in range(n):
            m.addConstr(xi_vars[i] <= big_xi, name="xi_bnd")
    m.addConstr(b_var <= big_b, name="b_bnd")
    m.addConstr(-1 * big_b <= b_var, name="b_bnd")

    if previous_portfolio is not None and turnover_constraints:  # add turnover constraints
        absolute_delta = addTurnoverConstraints(m, x_vars, previous_portfolio, turnover_limit)

    # relax variables
    relax_binary(z_vars)
    relax_binary(t_vars)

    C_epsilon_by_n = C * epsilon / n

    if feasible_solution:
        if not separable:
            m.addConstr(
                x_vars @ Q @ x_vars + (epsilon / 2) * (w_vars @ w_vars) + C_epsilon_by_n * xi_vars.sum() <= ObjSVMMVO,
                name="suboptimality")
        else:
            m.addConstr(x_vars @ Q @ x_vars + (epsilon / 2) * (w_vars @ w_vars) <= ObjSVMMVO, name="suboptimality")
    else:
        m.remove(m.getConstrByName('turnover'))
        if not separable:
            m.addConstr(
                absolute_delta.sum() + (epsilon / 2) * (w_vars @ w_vars) + C_epsilon_by_n * xi_vars.sum() <= ObjSVMMVO,
                name="suboptimality")
        else:
            m.addConstr(absolute_delta.sum() + (epsilon / 2) * (w_vars @ w_vars) <= ObjSVMMVO, name="suboptimality")

    m.setObjective(b_var, gp.GRB.MAXIMIZE)
    m.reset()
    start = time.time()
    m.optimize()
    max_b = m.objVal

    m.setObjective(b_var, gp.GRB.MINIMIZE)
    m.reset()
    m.optimize()
    min_b = m.objVal

    big_b = max(abs(max_b), abs(min_b))

    end_b = time.time()
    m.setObjective(xi_vars.sum(), gp.GRB.MAXIMIZE)
    m.reset()
    m.optimize()
    big_xi = m.objVal
    end_xi = time.time()

    return big_b, big_xi, end_b - start, end_xi - end_b


def objectiveBigMStrategyTightening(**kwargs):
    period_Context, epsilon, C, q = kwargs['period_Context'], kwargs['epsilon'], kwargs['C'], kwargs['q']
    theorem3_bool = False
    Verbose = kwargs['Verbose']
    if Verbose:
        print("-" * 20)
        print("Calculating Big M")
    bounds_improved_grb = True
    ObjSVMMVO, feasible_solution = ConstructFeasibleSolution(
        **kwargs)  # kwargs may or may not have big M limit times etc

    largest_abs = size_of_largest_feature(period_Context, q)
    largest_abs_pdist = largest_pairwise_distance(period_Context, q)

    n, p = period_Context.shape
    big_w_inf, big_w_2, big_b, big_xi, xi_str = theorem1(ObjSVMMVO, n, epsilon, C, largest_abs)
    bigM = theorem2(largest_abs_pdist, big_w_2, big_xi)
    bigM_cand = theorem3(largest_abs, big_b, big_w_2)

    if bigM_cand < bigM:
        theorem3_bool = True
        bigM = bigM_cand

    bigM_kwargs_for_SOCP = kwargs.copy()
    bigMs = [bigM]

    SVMMVO_bigM_time_limit, SVMMVO_MipGap, SVMMVO_SolutionLimit = get_SVMMVO_bigM_Vars(**kwargs)
    SkipSOCP, tightening_iter_lim = get_tightening_kwargs(**kwargs)
    iter_count = 1
    while bounds_improved_grb and iter_count <= tightening_iter_lim:
        if not SkipSOCP:
            bigM, big_b, big_xi, bounds_improved_socp = SOCP_step(bigM, big_w_inf, big_w_2, big_b,
                                                                  big_xi, ObjSVMMVO, feasible_solution,
                                                                  largest_abs_pdist,
                                                                  largest_abs,
                                                                  bigM_kwargs_for_SOCP)

        # Solve SVMMVO using Gurobi?
        user_big_m = {'bigM': bigM, 'big_w_inf': big_w_inf,
                      'big_b': big_b, 'big_xi': big_xi,
                      'feasible_solution': feasible_solution}
        # update big_w_inf, big_xi
        try:
            SVMMVO_results = SVMMVO(limit_time=SVMMVO_bigM_time_limit, MipGap=SVMMVO_MipGap,
                                    user_big_m=user_big_m, SolutionLimit=SVMMVO_SolutionLimit, **kwargs)
        except:
            warnings.warn("No heuristic solution found for SVMMVO")
            SVMMVO_results = {'obj_value': ObjSVMMVO}
        # bounds improved? Keep going

        if SVMMVO_results['obj_value'] < ObjSVMMVO:
            bounds_improved_grb = True
            ObjSVMMVO = SVMMVO_results['obj_value']

            big_w_inf_cand, big_w_2_cand, big_b_cand, big_xi_cand, xi_str_cand = theorem1(ObjSVMMVO, n, epsilon, C,
                                                                                          largest_abs)
            # update the best bounds
            big_w_inf, big_w_2 = min(big_w_inf, big_w_inf_cand), min(big_w_2, big_w_2_cand)
            big_b, big_xi = min(big_b, big_b_cand), min(big_xi, big_xi_cand)

            # recalculate bounds according to theorem
            bigM = theorem2(largest_abs_pdist, big_w_2, big_xi)
            bigM_cand = theorem3(largest_abs, big_b, big_w_2)
            # take the best M
            if bigM_cand < bigM:
                theorem3_bool = True
                bigM = bigM_cand
            bigMs.append(bigM)
        else:
            bounds_improved_grb = False
        iter_count += 1
    return {'bigM': bigM, 'big_w_inf': big_w_inf, 'big_w_2': big_w_2, 'big_b': big_b,
            'big_xi': big_xi, 'feasible_solution': feasible_solution, 'Theorem': theorem3_bool,
            'xi lemma': xi_str}


def HyperparameterBigMStrategy(**kwargs):
    period_Context, epsilon, C, q = kwargs['period_Context'], kwargs['epsilon'], kwargs['C'], kwargs['q']
    Verbose = kwargs['Verbose']
    theorem3_bool = False

    if Verbose:
        print("-" * 20)
        print("Calculating Big M")

    start = time.time()
    ObjSVMMVO, feasible_solution, C, epsilon, warm_start = ConstructFeasibleSolutionandHyperParams(**kwargs)
    end = time.time()
    if Verbose:
        print("Feasible solution constructed in ", end - start, " seconds")
        print("Feasible solution objective value ", ObjSVMMVO)
    n, p = period_Context.shape
    largest_abs = size_of_largest_feature(period_Context, q)
    largest_abs_pdist = largest_pairwise_distance(period_Context, q)
    big_w_inf, big_w_2, big_b, big_xi, xi_str = theorem1(ObjSVMMVO, n, epsilon, C, largest_abs)
    bigM = theorem2(largest_abs_pdist, big_w_2, big_xi)
    bigM_cand = theorem3(largest_abs, big_b, big_w_2)
    if Verbose:
        print("Xi Lemma ", xi_str)
        print("Theorem 2 big M ", bigM)
        print("Theorem 3 big M ", bigM_cand)
    if bigM_cand < bigM:
        theorem3_bool = True
        bigM = bigM_cand

    return {'bigM': bigM, 'big_w_inf': big_w_inf, 'big_w_2': big_w_2, 'big_b': big_b,
            'big_xi': big_xi, 'feasible_solution': feasible_solution, 'Theorem': theorem3_bool,
            'xi lemma': xi_str, 'C': C, 'epsilon': epsilon, 'warm_start': warm_start}
