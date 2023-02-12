import numpy as np
import pandas as pd
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import matplotlib.pyplot as plt


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rolling_stats(series, window_size: int):
    df_avg_roll = series.rolling(window_size).mean()
    df_std = series.rolling(window_size).std()

    plt.figure(figsize=(9, 5))
    plt.plot(series, color='#379BDB', label='Original')
    plt.plot(df_avg_roll, color='#D22A0D', label='Rolling Mean')
    plt.plot(df_std, color='#142039', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


def test_stationarity(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                                             'p-value', '#Lags Used',
                                             'Number of Observations Used'])
    for [key, value] in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
    Plot time series, its ACF and PACF, calculate Dickey–Fuller test
    y - timeseries
    lags - how many lags to include in ACF, PACF calculation
    """
    test_stationarity(y)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller:\
                         p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


def obtain_adf_kpss_results(series, max_d: int):
    """ Build dataframe with ADF statistics and p-value for time series
        after applying difference on time series
    Args:
        series (df): Dataframe of univariate time series
        max_d (int): Max value of how many times apply difference
    Return:
        Dataframe showing values of ADF statistics and p when applying
        ADF test after applying d times differencing on a time-series.
    """
    results = []

    for idx in range(max_d):
        adf_result = adfuller(series, autolag='AIC')
        kpss_result = kpss(series, regression='c', nlags="auto")
        series = series.diff().dropna()
        if adf_result[1] <= 0.05:
            adf_stationary = True
        else:
            adf_stationary = False
        if kpss_result[1] <= 0.05:
            kpss_stationary = False
        else:
            kpss_stationary = True

        stationary = adf_stationary & kpss_stationary
        results.append((idx, adf_result[1], kpss_result[1], adf_stationary,
                        kpss_stationary, stationary))

    results_df = pd.DataFrame(results, columns=['d', 'adf_stats',
                                                'p-value (kpss)',
                                                'is_adf_stationary',
                                                'is_kpss_stationary',
                                                'is_stationary'])
    return results_df
