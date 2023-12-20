import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm
import time


def CreateMVOModel(mu, Q, targetRet):
    """
    #---------------------------------------------------------------------- Use this function to construct an example
    of a MVO portfolio. # # An example of an MVO implementation is given below. You can use this # version of MVO if
    you like, but feel free to modify this code as much # as you need to. You can also change the inputs and outputs
    to suit # your needs.

    # You may use quadprog, Gurobi, or any other optimizer you are familiar
    # with. Just be sure to include comments in your code.

    # *************** WRITE YOUR CODE HERE ***************
    #----------------------------------------------------------------------
    """

    # Find the total number of assets
    n = len(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1
    covariance_sqrt = np.real(sqrtm(Q))
    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.sum_squares(covariance_sqrt @ x)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb])
    return x, prob


# def MVO(mu, Q):
#     """
#     #---------------------------------------------------------------------- Use this function to construct an example of a MVO portfolio.
#     #
#     # An example of an MVO implementation is given below. You can use this
#     # version of MVO if you like, but feel free to modify this code as much
#     # as you need to. You can also change the inputs and outputs to suit
#     # your needs.
#
#     # You may use quadprog, Gurobi, or any other optimizer you are familiar
#     # with. Just be sure to include comments in your code.
#
#     # *************** WRITE YOUR CODE HERE ***************
#     #----------------------------------------------------------------------
#     """
#     targetRet = mu.mean()
#     # Find the total number of assets
#     x, prob = CreateMVOModel(mu, Q, targetRet)
#     prob.solve(verbose=False, solver='ECOS')
#     return x.value


def MVO(mu, Q, targetRet):
    start = time.time()
    x, prob = CreateMVOModel(mu, Q, targetRet)
    prob.solve(verbose=False, solver='ECOS')
    end = time.time()
    return {'x': x.value, 'time': end - start}


def CreateMVOTurnoverModel(mu, Q, targetRet, turnover_limit, previous_portfolio):
    # Find the total number of assets
    n = len(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1
    covariance_sqrt = np.real(sqrtm(Q))
    # Define and solve using CVXPY
    x = cp.Variable(n)
    delta = cp.Variable(n)  # holdings shift
    if previous_portfolio is None:
        constraints = [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb]
    else:
        constraints = [A @ x <= b,
                       Aeq @ x == beq,
                       delta == x - previous_portfolio,
                       cp.norm(delta, 1) <= turnover_limit,
                       x >= lb]
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.sum_squares(covariance_sqrt @ x)), constraints)
    return x, prob


def MVOTurnover(mu, Q, targetRet, turnover_limit, previous_portfolio):
    start = time.time()
    x, prob = CreateMVOTurnoverModel(mu, Q, targetRet, turnover_limit, previous_portfolio)
    prob.solve(verbose=False, solver='ECOS')
    end = time.time()
    return {'x': x.value, 'time': end - start}


def RP(mu, Q):
    """
    Uses convex optimization to solve the risk parity problem
    $$\min y^t Q y - k \sum ln(y)$$
    $$x = y / sum(y) $$

    :param mu:
    :param Q:
    :return:
    """
    # # of Assets
    n = len(mu)

    # Decision Variables
    w = cp.Variable(n)

    # Kappa
    k = 2

    constraints = [
        w >= 0  # Disallow Short Sales
    ]

    # Objective Function
    risk = cp.quad_form(w, Q)
    log_term = 0
    for i in range(n):
        log_term += cp.log(w[i])

    prob = cp.Problem(cp.Minimize(risk - (k * log_term)), constraints=constraints)

    # ECOS fails sometimes, if it does then do SCS
    try:
        prob.solve(verbose=False)
    except:
        prob.solve(solver='SCS', verbose=False)

    x = w.value
    x = np.divide(x, np.sum(x))

    # Check Risk Parity Condition is actually met
    risk_contrib = np.multiply(x, Q.dot(x))
    if not np.all(np.isclose(risk_contrib, risk_contrib[0])):
        print(risk_contrib)
        print("RP did not work")

    return x
