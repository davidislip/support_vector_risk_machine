import hashlib
import json
import os
import pandas as pd
from pathlib import Path
import pickle as pkl
import numpy as np

def grab_scalar_backtest(backtest_results, scalar_str =  'optimality gap'):
    gaps = []
    dates = []
    for t in backtest_results.keys():
        gap = backtest_results[t][scalar_str]
        gaps.append(gap)
        dates.append(backtest_results[t]['calEnd'])

    return pd.Series(gaps, index = dates, name = scalar_str)

def grab_eigs_backtest(backtest_results):
    eigs_over_time = []
    dates = []
    for t in backtest_results.keys():
        cov = backtest_results[t]['cov']
        eigs_ = np.linalg.eig(cov)[0]
        eigs_over_time.append(eigs_[:, None])
        dates.append(backtest_results[t]['calEnd'])
    return np.concatenate(eigs_over_time, axis = 1)

def export_experimental_results(path, uid, portfValue, elapsed_time, x, turnover):

    experimental_results = {}

    experimental_results['portfValue'] = portfValue
    experimental_results['elapsed_time'] = elapsed_time
    experimental_results['x'] = x
    experimental_results['turnover'] = turnover

    experiment_file_path = path + "//" + uid + "_experimental_info.pkl"
    with open(experiment_file_path, 'wb') as fp:
        pkl.dump(experimental_results, fp)

def import_experimental_results(path, uid):

    with open(path + "//" + uid + "_experimental_info.pkl",'rb') as fp:
        experimental_results = pkl.load(fp)

    portfValue = experimental_results['portfValue']
    elapsed_time = experimental_results['elapsed_time']
    x  = experimental_results['x']
    turnover =  experimental_results['turnover']

    return portfValue, elapsed_time, x, turnover


def export_dict(path, uid, dictionary, name):
    experiment_file_path = path + "//" + uid + "_"+name+".pkl"
    with open(experiment_file_path, 'wb') as fp:
        pkl.dump(dictionary, fp)

def import_dict(path, uid, name):
    experiment_file_path = path + "//" + uid + "_"+name+".pkl"
    with open(experiment_file_path,  'rb') as fp:
        experimental_results = pkl.load(fp)
    return experimental_results

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def init_dataframe(investor_preferences, path):
    investor_preferences_cols = list(investor_preferences.keys())
    investor_preferences_cols.sort()
    if not os.path.exists(path):
        os.makedirs(path)
        print("Path created: ", path)

    isFile = os.path.isfile(path+"//data_dictionary.pkl")
    if not isFile:
        columns = ['uid', 'estimator', 'optimizer', 'universe', 'imputation_method', 'ticker_str', 'hyperparam_search', 'NoPeriods', 'run']
        for col in investor_preferences_cols:
            columns.append(col)
        #if file does not exist create it
        df = pd.DataFrame(columns = columns)
        df.to_pickle(path+"//data_dictionary.pkl")
    else:
        old_df = pd.read_pickle(path+"//data_dictionary.pkl")
        old_df.to_pickle(path+"//data_dictionary_backup.pkl")
        decision = input("Would you like to overwrite the experiments. Input \"yes\", \"true\", \"t\", \"1\" for Yes")
        if str2bool(decision):
            columns = ['uid', 'estimator', 'optimizer', 'universe', 'imputation_method', 'ticker_str', 'hyperparam_search', 'NoPeriods', 'run']
            for col in investor_preferences_cols:
                columns.append(col)
            #if file does not exist create it
            df = pd.DataFrame(columns = columns)
            df.to_pickle(path+"//data_dictionary.pkl")
        else:
            df = pd.read_pickle(path+"//data_dictionary.pkl")
    return df
def add_to_data_dict(path, df, run, estimator, optimizer, universe,
                     imputation_method, ticker_str, hyperparam_search, NoPeriods,
                     investor_preferences):
    row_dict = {}
    row_dict['run'] = run
    row_dict['estimator'] = estimator.__name__
    row_dict['optimizer'] = optimizer.__name__
    row_dict['universe'] = universe
    row_dict['imputation_method'] = imputation_method
    row_dict['ticker_str'] = ticker_str
    row_dict['hyperparam_search'] = hyperparam_search
    row_dict['NoPeriods'] = NoPeriods
    investor_preferences_cols = list(investor_preferences.keys())
    investor_preferences_cols.sort()
    for col in investor_preferences_cols:
        if hasattr(investor_preferences[col] , '__name__'):
            row_dict[col] = investor_preferences[col].__name__
        else:
            if type(investor_preferences[col]) is float:
                row_dict[col] = str(round(investor_preferences[col], 5))
            else:
                row_dict[col] = str(investor_preferences[col])
    uid = hashlib.sha1(json.dumps(row_dict, sort_keys=True).encode()).hexdigest()
    row_dict['uid'] = uid

    row_df = pd.DataFrame(row_dict, index = [row_dict['uid']])
    df = pd.concat([df, row_df], axis = 0)
    df.drop_duplicates(inplace = True)
    #df.to_pickle(path+"//data_dictionary.pkl")
    return df, uid