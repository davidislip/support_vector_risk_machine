import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gmean
from IPython.display import clear_output


def plot_results_relative_ticker(CardMVO_results, MVO_results, turnovers, cardinalities, card_strings,
                                 turnover_strings, adjClose, initialVal=100000, ticker_str='SPY', figsize=(8.5, 10.5)):
    max_y = 0
    lw = 1
    min_y = 0.8

    fig, axs = plt.subplots(len(cardinalities), len(turnovers),
                            sharex=True, sharey=True, figsize=figsize)

    if len(cardinalities) == 1:
        axs = np.expand_dims(axs, axis=0)
    if len(turnovers) == 1:
        axs = np.expand_dims(axs, axis=1)

    df_cols = []
    # load in the appropriate files
    for i in range(len(cardinalities)):

        df_rows = []

        for j in range(len(turnovers)):
            turnover_lim = turnovers[j]
            card = cardinalities[i]
            card_mvo_wealth, elapsed_time, x, turnover = CardMVO_results[(str(card), str(turnover_lim))]
            if ticker_str is not None:
                value_target_index = adjClose.loc[card_mvo_wealth.index, ticker_str]
                time0 = value_target_index.index[0] - pd.offsets.MonthEnd(1)
                spy_time0 = adjClose.loc[time0, ticker_str]
                value_target_index = initialVal * value_target_index / spy_time0

            card_mvo_wealth.columns = ['Card-MVO']

            mvo_wealth, elapsed_time, x, turnover = MVO_results[str(turnover_lim)]
            mvo_wealth.columns = ['MVO']

            card_mvo_rel_wealth = pd.Series(card_mvo_wealth.values.flatten() / value_target_index.values.flatten(),
                                            index=card_mvo_wealth.index)
            mvo_wealth_rel_wealth = pd.Series(mvo_wealth.values.flatten() / value_target_index.values.flatten(),
                                              index=mvo_wealth.index)
            card_mvo_rel_wealth.plot(fontsize=8, ax=axs[i, j], linewidth=lw, legend=False, label='Card-MVO')
            mvo_wealth_rel_wealth.plot(fontsize=8, ax=axs[i, j], linewidth=lw, legend=False, label='MVO')

            # if ticker_str is not None:
            # value_target_index.plot(fontsize = 8, ax=axs[i, j], linewidth = lw, legend = False)

            card_string = card_strings[i]
            turnover_string = turnover_strings[j]

            axs[i, j].set_title(card_string + ", " + turnover_string, fontsize=8)
            # print(card_mvo_wealth.max()[0])
            max_ = card_mvo_rel_wealth.max()  # [0]
            # print(max_)
            if max_ > max_y:
                max_y = max_

    axs[len(cardinalities) - 1, 0].legend(loc='upper center',
                                          bbox_to_anchor=(0.5, -0.4),
                                          fancybox=False,
                                          shadow=False, ncol=3,
                                          fontsize=8)
    for ax in axs:
        for sub_ax in ax:
            sub_ax.yaxis.set_tick_params(which='minor', bottom=False)
            sub_ax.set_yticks([min_y + i * 0.2 for i in range(2 + int((max_y - min_y) / 0.2))])
            sub_ax.grid()
            sub_ax.set_xlabel("Date", fontsize=8)
    # fig.supxlabel('Date',fontsize = 8)
    fig.supylabel("Cumulative Relative Wealth", fontsize=8)
    # plt.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.4)

    return


def plot_weights(x, tickers):
    weights = pd.DataFrame(x[(x > 0.01).any(axis=1)], index=tickers[(x > 0.01).any(axis=1)])
    weights.columns = [col + 1 for col in weights.columns]
    n = len(weights)
    if n <= 20:
        legend_bool = True
    else:
        legend_bool = False
    weights.T.plot.area(title='Portfolio weights',
                        ylabel='Weights', xlabel='Rebalance Period',
                        figsize=(6, 3),
                        legend=legend_bool, stacked=True)
    if legend_bool:
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


def calculate_return_stats(portfValue, riskFree):
    portfRets = portfValue.pct_change(1).iloc[1:, :]
    # Calculate the portfolio excess returns
    portfExRets = portfRets.subtract(
        riskFree[(riskFree.index >= portfRets.index[0]) & (riskFree.index <= portfRets.index[-1])], axis=0)

    # Calculate the portfolio Sharpe ratio
    Ret = 12 * ((portfExRets + 1).apply(gmean, axis=0) - 1)
    Vol = (12 ** 0.5) * (portfExRets.std())
    SR = (12 ** 0.5) * (((portfExRets + 1).apply(gmean, axis=0) - 1) / portfExRets.std())
    # Sortino = (12**0.5)*(((portfExRets + 1).apply(gmean, axis=0) - 1)/portfExRets.loc[portfExRets.values < 0].std())
    # Calculate the average turnover rate

    return Ret.iloc[0], Vol.iloc[0], SR.iloc[0]  # , Sortino.iloc[0]


def make_stats_table(CardMVO_results, MVO_results, turnovers, cardinalities, card_strings,
                     turnover_strings, riskFree):
    df_cols = []
    # load in the appropriate files
    for i in range(len(cardinalities)):

        df_rows = []

        for j in range(len(turnovers)):
            turnover_lim = turnovers[j]
            card = cardinalities[i]
            card_mvo_wealth, elapsed_time, x, turnover = CardMVO_results[(str(card), str(turnover_lim))]

            card_mvo_wealth.columns = ['Card-MVO']

            mvo_wealth, elapsed_time, x, turnover = MVO_results[str(turnover_lim)]
            mvo_wealth.columns = ['MVO']

            mvo_ret_stats = calculate_return_stats(mvo_wealth, riskFree)
            card_ret_stats = calculate_return_stats(card_mvo_wealth, riskFree)

            card_string = card_strings[i]
            turnover_string = turnover_strings[j]

            row_tuples = [(turnover_string, "MVO"), (turnover_string, "Card MVO")]
            row_index = pd.MultiIndex.from_tuples(row_tuples)

            col_tuples = [(card_string, "$\mu$"), (card_string, "$\sigma$"),
                          (card_string, "S.R")]  # , (card_string,"Sortino.R")]
            col_index = pd.MultiIndex.from_tuples(col_tuples)

            ret_stats = [mvo_ret_stats, card_ret_stats]

            out = pd.DataFrame(ret_stats, index=row_index, columns=col_index)

            df_rows.append(out)

        column = pd.concat(df_rows)
        df_cols.append(column)

    return pd.concat(df_cols, axis=1)


def make_turnover_table(CardMVO_results, MVO_results, turnovers, cardinalities, card_strings,
                        turnover_strings):
    df_cols = []
    # load in the appropriate files
    for i in range(len(cardinalities)):

        df_rows = []

        for j in range(len(turnovers)):
            turnover_lim = turnovers[j]
            card = cardinalities[i]
            card_mvo_wealth, elapsed_time, x, turnover = CardMVO_results[(str(card), str(turnover_lim))]
            card_ret_stats = np.mean(turnover[1:])
            card_mvo_wealth.columns = ['Card-MVO']

            mvo_wealth, elapsed_time, x, turnover = MVO_results[str(turnover_lim)]
            mvo_wealth.columns = ['MVO']
            mvo_ret_stats = np.mean(turnover[1:])

            card_string = card_strings[i]
            turnover_string = turnover_strings[j]

            row_tuples = [(turnover_string, "MVO"), (turnover_string, "Card MVO")]
            row_index = pd.MultiIndex.from_tuples(row_tuples)

            col_tuples = [(card_string, "$C_0$")]  # , (card_string,"Sortino.R")]
            col_index = pd.MultiIndex.from_tuples(col_tuples)

            ret_stats = [mvo_ret_stats, card_ret_stats]

            out = pd.DataFrame(ret_stats, index=row_index, columns=col_index)

            df_rows.append(out)

        column = pd.concat(df_rows)
        df_cols.append(column)

    return pd.concat(df_cols, axis=1)


def efficient_frontier_premium(Strategy, max_return, min_return, NumPts, periodReturns, periodFactRet, env):
    """
    compute the efficient frontier
    using Strategy, from min_return to max_return with NumPts between
    The covariance estimation is done using periodReturns and periodFactRet and
    other data is stored in the environment (env)
    """
    max_premium = max_return / Strategy.current_estimates[0].mean() - 1
    min_premium = min_return / Strategy.current_estimates[0].mean() - 1

    premiums = np.linspace(min_premium, max_premium, NumPts)
    vols = np.zeros(NumPts)
    rets = np.zeros(NumPts)
    cardinalities = np.zeros(NumPts)
    mip_gaps = np.zeros(NumPts)
    oldVerbose = Strategy.investor_preferences['Verbose']
    oldLogToConsole = Strategy.investor_preferences['LogToConsole']
    Strategy.investor_preferences['Verbose'] = False
    Strategy.investor_preferences['LogToConsole'] = False
    results = {}
    for i, premium in enumerate(premiums):
        clear_output(wait=True)
        Strategy.investor_preferences['premium'] = premium
        Strategy.execute_strategy(periodReturns, periodFactRet, environment=env)
        vols[i] = Strategy.current_results['obj_value']
        rets[i] = (1 + premium) * Strategy.current_estimates[0].mean()
        cardinalities[i] = (Strategy.current_results['x'] >= 0.001).sum()
        mip_gaps[i] = Strategy.current_results['optimality gap']
        # asset_eligibilities[i] = Strategy.current_results['z']
        # hyperplane_coefficients[i] = Strategy.current_results['w']
        # biases[i] = Strategy.current_results['b']
        # feature_vectors[i] = Strategy.current_results['optimization_params']['period_Context']
        results[i] = Strategy.current_results
    Strategy.investor_preferences['Verbose'] = oldVerbose
    Strategy.investor_preferences['LogToConsole'] = oldLogToConsole
    return vols, rets,  premiums, cardinalities, mip_gaps, results
