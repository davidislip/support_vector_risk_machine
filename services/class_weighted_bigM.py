from services.class_weighted_models import *
from services.data_utils import *
import warnings
import math
from sklearn.model_selection import KFold
from services.big_m_strategies import ConstructFeasibleMVO, unpack_bigM_params
from services.big_m_strategies import size_of_largest_feature, largest_pairwise_distance, theorem1, theorem2, theorem3
from itertools import product


def ConstructFeasibleSolutionSVM(z_vals, K, period_Context, C, separable, q, bigM_limit_time, LogToConsole, Verbose):
    # take out args into new dict2
    # SVM

    if Verbose:
        print("Phase 1 SVM...")
    # svm_phase1_results = SVM(period_Context, z_vals, C, separable, bigM_limit_time, LogToConsole)
    # print(svm_phase1_results['obj_value'])
    # print(np.power(svm_phase1_results['w'], 2).sum())
    # print(svm_phase1_results['xi'].sum())
    # print(svm_phase1_results['time'])
    svm_phase1_results = sklearn_ClassWgtSVM(period_Context, K, z_vals, C, separable)
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
    svm_phase2_results = sklearn_ClassWgtSVM(period_Context_subset, K, z_vals, C, separable)
    if Verbose:
        print("Feasible Solution Constructed")
        print("-" * 20)
    # Calculate Objective Value
    ObjSVM = svm_phase2_results['obj_value']
    w = svm_phase2_results['w']
    b = svm_phase2_results['b']

    return ObjSVM, w, b


def ConstructFeasibleSolution(**kwargs):
    period_Context, C, separable = kwargs['period_Context'], kwargs['C'], kwargs['separable']
    q, epsilon = kwargs['q'], kwargs['epsilon']

    # Do Card MVO
    ObjMVO, feasible_solution, z_vals, x_vals = ConstructFeasibleMVO(**kwargs)

    # SVM

    bigM_limit_time, LogToConsole, Verbose = unpack_bigM_params(**kwargs)
    K = kwargs['K']
    ObjSVM, w, b = ConstructFeasibleSolutionSVM(z_vals, K, period_Context, C, separable, q,
                                                bigM_limit_time, LogToConsole, Verbose)

    return ObjMVO + epsilon * ObjSVM, feasible_solution


def ClassWgtConstructFeasibleSolutionandHyperParams(**kwargs):
    period_Context, separable = kwargs['period_Context'], kwargs['separable']

    q = kwargs['q']
    kappa = kwargs['kappa']

    n, p = period_Context.shape
    # do card MVO
    K = kwargs['K']
    class_weights = {-1: K / n, 1: (n - K) / n}
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
                svm_phase1_results = sklearn_ClassWgtSVM(period_Context.iloc[train_index], K, z_vals[train_index], C,
                                                         separable)
            except ValueError:
                if Verbose:
                    print("Value Error - sklearn . . . using Gurobi")
                svm_phase1_results = ClassWgtSVM(period_Context.iloc[train_index], K, z_vals[train_index], C, separable,
                                                 bigM_limit_time, LogToConsole)
            w_vals = svm_phase1_results['w']
            abs_w = np.abs(w_vals)
            q_largest = np.argpartition(abs_w, (-1) * q)[-q:]
            # restrict indices
            period_Context_subset = period_Context.iloc[:, q_largest]
            try:
                svm_phase2_results = sklearn_ClassWgtSVM(period_Context_subset.iloc[train_index], K,
                                                         z_vals[train_index], C,
                                                         separable)
                svc = svm_phase2_results['svc']
                pred_decision = svc.decision_function(period_Context_subset.iloc[test_index])
            except ValueError:
                svm_phase2_results = ClassWgtSVM(period_Context_subset.iloc[train_index], K, z_vals[train_index], C,
                                                 separable,
                                                 bigM_limit_time, LogToConsole)
                w, b = svm_phase2_results['w'], svm_phase2_results['b']
                pred_decision = period_Context_subset.iloc[test_index].values @ svm_phase2_results['w'] + b
            margin = u[test_index] * pred_decision
            xi_test = np.maximum(0, 1 - margin)
            errors[i] = class_weights[-1] * np.sum(xi_test[u[test_index] < -0.95]) + class_weights[1] * np.sum(
                xi_test[u[test_index] > 0.95])
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
        svm_phase1_results = sklearn_ClassWgtSVM(period_Context, K, z_vals, bestC, separable)
    except ValueError:
        svm_phase1_results = ClassWgtSVM(period_Context, K, z_vals, bestC, separable, bigM_limit_time, LogToConsole)
    w_vals = svm_phase1_results['w']
    abs_w = np.abs(w_vals)
    q_largest = np.argpartition(abs_w, (-1) * q)[-q:]
    # restrict indices
    period_Context_subset = period_Context.iloc[:, q_largest]
    try:
        svm_phase2_results = sklearn_ClassWgtSVM(period_Context_subset, K, z_vals, bestC, separable)
    except ValueError:
        svm_phase2_results = ClassWgtSVM(period_Context_subset, K, z_vals, bestC, separable, bigM_limit_time,
                                         LogToConsole)
    # sqrt(2 ObjSVM) is the soln is big_w2 and hence big w
    ObjFeasibleSVM = svm_phase2_results['obj_value']
    big_w2 = min(math.sqrt(2 * ObjFeasibleSVM), math.sqrt(2 * bestC))
    # solve BSS SVM
    w_vals = np.zeros(p)
    t_vals = np.zeros(p)
    w_vals[q_largest] = svm_phase2_results['w']
    t_vals[q_largest] = np.ones(len(q_largest))
    b_val = svm_phase2_results['b']

    warm_start = {'w_vals': w_vals, 'b_val': b_val, 't_vals': t_vals}
    BestSubsetSVM_results = BestSubsetClassWgtSVM(period_Context, K, z_vals, bestC, separable, q, big_w2,
                                                  bigM_limit_time,
                                                  LogToConsole, warm_start)
    ObjSVM = BestSubsetSVM_results['obj_value']
    # set epsilon so that the risk guarantee is satisfied
    epsilon = (kappa * ObjMVO * 1000) / (ObjSVM * 1000)  # this is the big line
    if Verbose:
        print("Largest epsilon value guaranteeing ", 1 + kappa, " risk: ", epsilon)
    warm_start = {'x_vals': x_vals, 'z_vals': z_vals,
                  'w_vals': BestSubsetSVM_results['w'], 't_vals': np.rint(BestSubsetSVM_results['t']),
                  'b_val': BestSubsetSVM_results['b'], 'xi_plus': BestSubsetSVM_results['xi_plus'],
                  'xi_neg': BestSubsetSVM_results['xi_neg']}
    return ObjMVO + epsilon * ObjSVM, feasible_solution, bestC, epsilon, warm_start


def ClassWgtConstructFeasibleSolutionandHyperParamsV2(**kwargs):
    # V2 also selects q
    period_Context, separable = kwargs['period_Context'], kwargs['separable']

    # q = kwargs['q']
    kappa = kwargs['kappa']

    n, p = period_Context.shape
    # do card MVO
    K = kwargs['K']
    class_weights = {-1: K / n, 1: (n - K) / n}
    ObjMVO, feasible_solution, z_vals, x_vals = ConstructFeasibleMVO(**kwargs)
    z_vals = np.rint(z_vals)
    u = 2 * z_vals - 1
    # SVM
    bigM_limit_time, LogToConsole, Verbose = unpack_bigM_params(**kwargs)

    # find the best C using k fold
    qs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, p]

    Cs = np.geomspace(2 ** (-12), 2 ** 12, 25)

    bestC = 2 ** (-12)
    bestq = 20
    lowest_error = n  # this is an upper bound on the error
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=20)
    error_list = []
    w_list = []
    norm_w = 0

    for (q, C) in product(qs, Cs):
        errors = np.zeros(n_splits)
        for i, (train_index, test_index) in enumerate(kf.split(period_Context)):
            # print(f"Fold {i}:")
            # print(f"  Train: index={train_index}")
            # print(f"  Test:  index={test_index}")
            # solve SVM
            try:
                svm_phase1_results = sklearn_ClassWgtSVM(period_Context.iloc[train_index], K, z_vals[train_index], C,
                                                         separable)
            except ValueError:
                if Verbose:
                    print("Value Error - sklearn . . . using Gurobi")
                svm_phase1_results = ClassWgtSVM(period_Context.iloc[train_index], K, z_vals[train_index], C, separable,
                                                 bigM_limit_time, LogToConsole)
            w_vals = svm_phase1_results['w']
            abs_w = np.abs(w_vals)
            q_largest = np.argpartition(abs_w, (-1) * q)[-q:]
            # restrict indices
            period_Context_subset = period_Context.iloc[:, q_largest]
            try:
                svm_phase2_results = sklearn_ClassWgtSVM(period_Context_subset.iloc[train_index], K,
                                                         z_vals[train_index], C,
                                                         separable)
                svc = svm_phase2_results['svc']
                pred_decision = svc.decision_function(period_Context_subset.iloc[test_index])
            except ValueError:
                svm_phase2_results = ClassWgtSVM(period_Context_subset.iloc[train_index], K, z_vals[train_index], C,
                                                 separable,
                                                 bigM_limit_time, LogToConsole)
                w, b = svm_phase2_results['w'], svm_phase2_results['b']
                pred_decision = period_Context_subset.iloc[test_index].values @ svm_phase2_results['w'] + b
            margin = u[test_index] * pred_decision
            xi_test = np.maximum(0, 1 - margin)
            errors[i] = class_weights[-1] * np.sum(xi_test[u[test_index] < -0.95]) + class_weights[1] * np.sum(
                xi_test[u[test_index] > 0.95])
        error_list.append(errors.mean())
        w_list.append(np.power(svm_phase2_results['w'], 2).sum())

        if errors.mean() < lowest_error:
            bestC = C
            bestq = q
            if Verbose:
                print("New Best C ", bestC)
                print("New Best q ", bestq)
                print("Test Error ", errors.mean())
                print("Selected Features ", period_Context_subset.columns)
            lowest_error = errors.mean()
            norm_w = np.power(svm_phase2_results['w'], 2).sum()

    if norm_w > 10 ** (-7) and Verbose:
        print("Non degenerate solution found")
    # calculate ObjSVM and construct bound on ||w||_1
    try:
        svm_phase1_results = sklearn_ClassWgtSVM(period_Context, K, z_vals, bestC, separable)
    except ValueError:
        svm_phase1_results = ClassWgtSVM(period_Context, K, z_vals, bestC, separable, bigM_limit_time, LogToConsole)
    w_vals = svm_phase1_results['w']
    abs_w = np.abs(w_vals)
    q_largest = np.argpartition(abs_w, (-1) * bestq)[-bestq:]
    # restrict indices
    period_Context_subset = period_Context.iloc[:, q_largest]
    try:
        svm_phase2_results = sklearn_ClassWgtSVM(period_Context_subset, K, z_vals, bestC, separable)
    except ValueError:
        svm_phase2_results = ClassWgtSVM(period_Context_subset, K, z_vals, bestC, separable, bigM_limit_time,
                                         LogToConsole)
    # sqrt(2 ObjSVM) is the soln is big_w2 and hence big w
    ObjFeasibleSVM = svm_phase2_results['obj_value']
    big_w2 = min(math.sqrt(2 * ObjFeasibleSVM), math.sqrt(2 * bestC))
    # solve BSS SVM
    w_vals = np.zeros(p)
    t_vals = np.zeros(p)
    w_vals[q_largest] = svm_phase2_results['w']
    t_vals[q_largest] = np.ones(len(q_largest))
    b_val = svm_phase2_results['b']

    warm_start = {'w_vals': w_vals, 'b_val': b_val, 't_vals': t_vals}
    BestSubsetSVM_results = BestSubsetClassWgtSVM(period_Context, K, z_vals, bestC, separable, bestq, big_w2,
                                                  bigM_limit_time,
                                                  LogToConsole, warm_start)
    ObjSVM = BestSubsetSVM_results['obj_value']
    # set epsilon so that the risk guarantee is satisfied
    epsilon = (kappa * ObjMVO * 1000) / (ObjSVM * 1000)  # this is the big line
    if Verbose:
        print("Largest epsilon value guaranteeing ", 1 + kappa, " risk: ", epsilon)
    warm_start = {'x_vals': x_vals, 'z_vals': z_vals,
                  'w_vals': BestSubsetSVM_results['w'], 't_vals': np.rint(BestSubsetSVM_results['t']),
                  'b_val': BestSubsetSVM_results['b'], 'xi_plus': BestSubsetSVM_results['xi_plus'],
                  'xi_neg': BestSubsetSVM_results['xi_neg']}
    return ObjMVO + epsilon * ObjSVM, feasible_solution, bestC, epsilon, warm_start, bestq


def ClassWgtHyperparameterBigMStrategy(**kwargs):
    period_Context, epsilon, C, q = kwargs['period_Context'], kwargs['epsilon'], kwargs['C'], kwargs['q']
    Verbose = kwargs['Verbose']
    theorem3_bool = False

    if Verbose:
        print("-" * 20)
        print("Calculating Big M")

    start = time.time()
    ObjSVMMVO, feasible_solution, C, epsilon, warm_start = ClassWgtConstructFeasibleSolutionandHyperParams(**kwargs)
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


def ClassWgtHyperparameterBigMStrategyV2(**kwargs):
    period_Context = kwargs['period_Context']
    Verbose = kwargs['Verbose']
    theorem3_bool = False

    if Verbose:
        print("-" * 20)
        print("Calculating Big M")

    start = time.time()
    ObjSVMMVO, feasible_solution, C, epsilon, warm_start, q = ClassWgtConstructFeasibleSolutionandHyperParamsV2(
        **kwargs)
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
            'xi lemma': xi_str, 'C': C, 'epsilon': epsilon, 'warm_start': warm_start, 'q': q}
