import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer
import torch
from services.strategies import *
from services.estimators import *
from services.torch_data_utils import *
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


def penalty_regularized_MVO(nr_assets):
    """
    convex optimization layer for portfolio optimization

    solves regularized MVO

    min x Sigma x - lambda mu x + delta ||x||^2
    """
    x = cp.Variable(nr_assets)

    # define parameters
    vec_returns = cp.Parameter(nr_assets)  # risk aversion adjusted vector returns
    target_ret = cp.Parameter()

    covariance_sqrt = cp.Parameter((nr_assets, nr_assets))
    delta = cp.Parameter(nonneg=True)

    # define objectives
    risk = cp.sum_squares(covariance_sqrt @ x)

    regularization = delta * cp.sum_squares(x)
    # constraints
    constraints = [cp.sum(x) == 1,
                   x >= 0, vec_returns.T @ x >= target_ret]

    obj = cp.Minimize(risk + regularization)
    prob = cp.Problem(obj, constraints)
    # the parameters below will be tensors
    return CvxpyLayer(prob, parameters=[target_ret, vec_returns, covariance_sqrt, delta], variables=[x])


def gmean_tch(input_x, dim=None):
    log_x = torch.log(input_x + 1)
    if dim is not None:
        return torch.exp(torch.mean(log_x)) - 1
    else:
        return torch.exp(torch.mean(log_x, dim=dim)) - 1


class e2e_regularization_estimator(nn.Module):
    """
    this estimator uses end to end differentiable layers to
    learn the best combination of regularization for a mean-variance optimizer

    The goal is to replace the investment process inside this training loop so that we can select the parameters to
    optimize investment outcomes

    i.e.  let r_t, t = 1, ..., T be a series of returns suppose, let W be an estimation window, and investPeriod be
    the investment period.

    At the first trade we use returns up until the Window to estimate the covariance and mean
    mu_1 =  1/W \sum_{t = 1} r_t
    Sigma_1 = 1/W \sum_{t = 1}^W (r_t - mu_1) (r_t - mu_1)^T
    then we solve MVO to obtain the first trade
    x_1 = argmin x^T Sigma_1x - \lambda \mu_1^T x
    then we use the return compounded over the investment period
    R_1 = return_over_{1+W --> 1+W +investPeriod} = \prod_{j=1}^{INVEST PERIOD} (1 + r_{t+W+i}) - 1
    to evaluate our portfolio out of sample:

    i.e. R_1^T x_1

    so now assume that we are using shrinkage to estimate the covariance
    then Sigma_1(\delta)  = 1/W \sum_{t = 1}^W (r_t - mu_1) (r_t - mu_1)^T + \delta I
    where I is nxn identity

    ideally we would want to pick \delta so that the out of sample performance is maximized

    i.e. we would solve

    \max_{\delta} performance of x_1
        where x_1 = argmin x^T Sigma_1(\delta) x - \lambda \mu_1^T x
    which is the same as
    \min_{\delta} - performance of x_1
        where x_1 = argmin x^T Sigma_1(\delta) x - \lambda \mu_1^T x

    to do this we use backpropagation

    \delta <-- \delta - learning_rate* \partial (Objective function) /\partial \delta

    to get this we need to have \partial x_1 /\partial \delta
    which is what cvxpy layers provides

    Furthermore, we don't just want to do this for a single period.
    We would like to do this for every rebalancing period in the training
    data.

    A good objective is to maximize the sharpe ratio obtained by applying the strategy over the data.
    This means we have to calculate x_1 based on data up until 1 ... W calculate its out of sample
    return on the return from W+1 to W+1+Investment period and then do the same thing for x_2, ... all the
    way till we run out of out of sample performance data

    This is not meant to be the best example of how to do end to end but
    rather a guideline/a useful pattern. Different objectives may be useful
    """

    def __init__(self, nr_assets, estimation_frequency, investPeriod, NumObs, opt_layer, lr=0.01, epochs=10,
                 store=False):
        super(e2e_regularization_estimator, self).__init__()

        self.estimation_frequency = estimation_frequency  # it takes too long to re-learn delta at each rebalancing period. This controls how often we reoptimize
        self.periods_since_last_estimation = 0  # this is a tracking variable that keeps track since the last period we optimized delta

        self.investPeriod = investPeriod  # this is the investment period over which the portfolio is held
        self.NumObs = NumObs  # this number of observations used to form our portfolio (our estimation window)
        self.nr_assets = nr_assets  # number of assets

        self.delta = nn.Parameter(torch.tensor(0.001, dtype=torch.float64))  # the parameter we are trying to optimize

        self.opt_layer = eval(opt_layer)(
            self.nr_assets)  # a fancy way for us to specify a string indicating which optimization layer we would like to use

        self.lr = lr  # learning rate
        self.epochs = epochs  # number of times we wish to pass over the dataset

        self.store = store  # a boolean indicating whether to store losses and deltas in the lists immediately below
        self.deltas = []
        self.losses = []

    def estimate_and_optimize(self, returns, factRet=None):
        # the dataloader forms allows us to get the returns up until t+NumObs
        # and the out of sample returns from t+NumObs +1  to t+NumObs+1+investment period
        training_set = DataLoader(ReturnsSlidingWindow(returns, self.NumObs, self.investPeriod))

        if self.periods_since_last_estimation % self.estimation_frequency == 0:
            print("Retraining regularization parameter ... ")
            # self.delta.data = torch.tensor(0.0001, dtype=torch.float64)
            self.train(training_set)
            self.periods_since_last_estimation = 0
        self.periods_since_last_estimation += 1
        # get the subset of returns that corresponds to the ones that we
        # want to use to form our portfolio
        returns_tch = torch.tensor(returns.iloc[(-1) * self.NumObs:, :].values)
        x_out = self(returns_tch)  # this is our prescribed portfolio
        return x_out.detach().cpu().numpy()

    def forward(self, returns, factRet=None):
        """
        forward pass. From a sample of data, computes the corresponding MVO portfolio
        with regularization self.delta and risk aversion
        :param returns:
        :param factRet:
        :return: mvo optimal portfolio
        """

        # empirical estimators
        mu, Q = sample_estimator(returns.numpy(), factRet=None)
        # sqrt of Q
        root_Q_tch = torch.from_numpy(np.real(sqrtm(Q)))
        # write to torch
        mu_tch = torch.from_numpy(mu).squeeze()
        # solve the for the optimal porfolio
        # SUPER IMPORTANT: set the Solver to ECOS. The default cvxpylayers solver does not solve QP's correctly ATM.
        x_, = self.opt_layer(mu_tch.mean(), mu_tch, root_Q_tch, self.delta, solver_args={"solve_method": "ECOS"})
        return x_

    def train(self, train_set):

        lr = self.lr
        epochs = self.epochs

        # Define the optimizer and its parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Number of elements in training set
        n_train = len(train_set)

        # Train the neural network
        for epoch in range(epochs):
            # TRAINING: forward + backward pass
            train_loss = 0

            optimizer.zero_grad()

            returns_of_strategy = torch.ones(len(train_set))  # tensor to hold the returns of the strategy
            x_star = torch.ones((len(train_set), self.nr_assets), dtype=torch.float64)  # tensor to hold the portfolios
            for t, (r, r_perf) in enumerate(train_set):
                # Forward pass: get the portfolio at time t. The returns given by the Data Loader already incorporate our estimation window
                x_star[t] = self(r.squeeze())
                # Loss function

                if r_perf.shape[1] > 1:  # if we are investing for many periods then
                    # the realized return is given by the compounded returns
                    # for lazyness use the geo mean function and then (1 + r_geo)^periods = compounded return
                    r_perf = torch.pow(1 + gmean_tch(r_perf, dim=1), r_perf.shape[1]) - 1

                returns_of_strategy[t] = (x_star[t] @ r_perf.squeeze())
                # Backward pass: backpropagation

            # calculate the Sharpe ratio realized by the current strategy defined by delta
            loss = -1 * gmean_tch(returns_of_strategy) / torch.std(returns_of_strategy)

            loss.backward()
            # Accumulate loss of the fully trained model
            train_loss = loss.item()

            if self.store:
                self.deltas.append(self.delta.item())
                self.losses.append(train_loss)

            # Update parameters
            optimizer.step()

            # Ensure that delta > 0 after taking a descent step
            for name, param in self.named_parameters():
                if name == 'delta':
                    param.data.clamp_(0.000000001)

    def visualize(self):
        plt.figure()
        plt.plot(self.deltas)
        plt.title("Delta at each outer training iteration")
        plt.figure()
        plt.plot(self.losses)
        plt.title("Average training loss (negative of average return over training data)")