import hashlib
import json
import os
import pandas as pd
import pickle as pkl
import numpy as np


def grab_scalar_backtest(backtest_results, scalar_str='optimality gap'):
    """
    grabs a scalar from the backtest results dictionary
    the backtest results dict has keys t = 0 ... T
    at each t backtest_results[t] stores the scalar_str metric

    returns a pandas series of the desired metric
    """
    gaps = []
    dates = []
    for t in backtest_results.keys():
        gap = backtest_results[t][scalar_str]
        gaps.append(gap)
        dates.append(backtest_results[t]['calEnd'])

    return pd.Series(gaps, index=dates, name=scalar_str)


def grab_eigs_backtest(backtest_results):
    """
    scans the backtest_results dictionary
    and calculates the eigenvalues of the covariance matric

    returns a n x T numpy array of the eigenvalues
    """
    eigs_over_time = []
    dates = []
    for t in backtest_results.keys():
        cov = backtest_results[t]['cov']
        eigs_ = np.linalg.eig(cov)[0]
        eigs_over_time.append(eigs_[:, None])
        dates.append(backtest_results[t]['calEnd'])
    return np.concatenate(eigs_over_time, axis=1)


def export_experimental_results(path, uid, portfValue, elapsed_time, x, turnover):
    """
    store the portfValue series, elapsed_time, holdings series x, and turnover
    in a dictionary with the uid as a prefix at the given path
    """
    experimental_results = {}

    experimental_results['portfValue'] = portfValue
    experimental_results['elapsed_time'] = elapsed_time
    experimental_results['x'] = x
    experimental_results['turnover'] = turnover

    experiment_file_path = path + "//" + uid + "_experimental_info.pkl"
    with open(experiment_file_path, 'wb') as fp:
        pkl.dump(experimental_results, fp)


def import_experimental_results(path, uid):
    """
    loads the experimental results dictionary
    unpacks its values and returns them
    """
    with open(path + "//" + uid + "_experimental_info.pkl", 'rb') as fp:
        experimental_results = pkl.load(fp)

    portfValue = experimental_results['portfValue']
    elapsed_time = experimental_results['elapsed_time']
    x = experimental_results['x']
    turnover = experimental_results['turnover']

    return portfValue, elapsed_time, x, turnover


def export_dict(path, uid, dictionary, name):
    """
    generic dictionary export function
    """
    experiment_file_path = path + "//" + uid + "_" + name + ".pkl"
    with open(experiment_file_path, 'wb') as fp:
        pkl.dump(dictionary, fp)


def import_dict(path, uid, name):
    """
    generic dictionary import function
    """
    experiment_file_path = path + "//" + uid + "_" + name + ".pkl"
    with open(experiment_file_path, 'rb') as fp:
        experimental_results = pkl.load(fp)
    return experimental_results


def str2bool(v):
    """
    v: string
    returns True if v is in ("yes", "true", "t", "1")
    """
    return v.lower() in ("yes", "true", "t", "1")


def init_dataframe(investor_preferences, path):
    """
    each experiment directory contains a dataframe called the data dictionary
    the data dictionary contains all the experimental information required to
    run the experiment

    this function initializes the dictionary and gives the user an
    option to override the existing data dictionary
    """
    investor_preferences_cols = list(investor_preferences.keys())
    investor_preferences_cols.sort()
    if not os.path.exists(path):
        os.makedirs(path)
        print("Path created: ", path)

    isFile = os.path.isfile(path + "//data_dictionary.pkl")
    if not isFile:
        columns = ['uid', 'estimator', 'optimizer', 'universe', 'imputation_method', 'ticker_str', 'hyperparam_search',
                   'NoPeriods', 'run']
        for col in investor_preferences_cols:
            columns.append(col)
        # if file does not exist create it
        df = pd.DataFrame(columns=columns)
        df.to_pickle(path + "//data_dictionary.pkl")
    else:
        old_df = pd.read_pickle(path + "//data_dictionary.pkl")
        old_df.to_pickle(path + "//data_dictionary_backup.pkl")
        decision = input("Would you like to overwrite the experiments. Input \"yes\", \"true\", \"t\", \"1\" for Yes")
        if str2bool(decision):
            columns = ['uid', 'estimator', 'optimizer', 'universe', 'imputation_method', 'ticker_str',
                       'hyperparam_search', 'NoPeriods', 'run']
            for col in investor_preferences_cols:
                columns.append(col)
            # if file does not exist create it
            df = pd.DataFrame(columns=columns)
            df.to_pickle(path + "//data_dictionary.pkl")
        else:
            df = pd.read_pickle(path + "//data_dictionary.pkl")
    return df


def add_to_data_dict(path, df, run, estimator, optimizer, universe,
                     imputation_method, ticker_str, hyperparam_search, NoPeriods,
                     investor_preferences):
    """
    adds the arguments to the data dictionary df
    """
    row_dict = {'run': run, 'estimator': estimator.__name__, 'optimizer': optimizer.__name__, 'universe': universe,
                'imputation_method': imputation_method, 'ticker_str': ticker_str,
                'hyperparam_search': hyperparam_search, 'NoPeriods': NoPeriods}
    investor_preferences_cols = list(investor_preferences.keys())
    investor_preferences_cols.sort()
    for col in investor_preferences_cols:
        if hasattr(investor_preferences[col], '__name__'):
            row_dict[col] = investor_preferences[col].__name__
        else:
            if type(investor_preferences[col]) is float:
                row_dict[col] = str(round(investor_preferences[col], 5))
            else:
                row_dict[col] = str(investor_preferences[col])
    uid = hashlib.sha1(json.dumps(row_dict, sort_keys=True).encode()).hexdigest()
    row_dict['uid'] = uid

    row_df = pd.DataFrame(row_dict, index=[row_dict['uid']])
    df = pd.concat([df, row_df], axis=0)
    df.drop_duplicates(inplace=True)
    # df.to_pickle(path+"//data_dictionary.pkl")
    return df, uid
