import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


class ReturnsSlidingWindow(Dataset):
    """Sliding window dataset constructor taken from Giorgio Costas Github
    """

    def __init__(self, R, n_obs, perf_period):
        """Construct a sliding (i.e., rolling) window dataset from a complete timeseries dataset

        Inputs
        Y: pandas dataframe with the complete asset return dataset
        n_obs: Number of observations in the window
        perf_period: Number of observations in the 'performance window' used to evaluate out-of-sample
        performance. The 'performance window' is also a sliding window

        Output
        Dataset where each element is the tuple (y, y_perf)
        y: Realizations window (dim: n_obs x n_y)
        y_perf: Window of forward-looking (i.e., future) realizations (dim: perf_period x n_y)

        Note: For each feature window 'x', the last scenario x_t is reserved for prediction and
        optimization. Therefore, no pair in 'y' is required (it is assumed the pair y_T is not yet
        observable)
        """
        self.R = Variable(torch.tensor(R.values, dtype=torch.double))
        self.n_obs = n_obs
        self.perf_period = perf_period

    def __getitem__(self, index):
        r = self.R[index:index + self.n_obs]
        r_perf = self.R[index + self.n_obs: index + self.n_obs + self.perf_period]
        return r, r_perf

    def __len__(self):
        return len(self.R) - self.n_obs - self.perf_period + 1
