import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm


def MVO(mu, Q):
    """
    #---------------------------------------------------------------------- Use this function to construct an example of a MVO portfolio.
    #
    # An example of an MVO implementation is given below. You can use this
    # version of MVO if you like, but feel free to modify this code as much
    # as you need to. You can also change the inputs and outputs to suit
    # your needs.

    # You may use quadprog, Gurobi, or any other optimizer you are familiar
    # with. Just be sure to include comments in your code.

    # *************** WRITE YOUR CODE HERE ***************
    #----------------------------------------------------------------------
    """

    # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

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
    prob.solve(verbose=False, solver = 'ECOS')
    return x.value



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