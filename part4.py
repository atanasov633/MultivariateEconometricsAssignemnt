# Assignment Multivariate Econometrics 
# Group 23: Atanas Atanasov, Busra Turk, Ion Paraschos, Max Hugen


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arch.unitroot import cointegration, engle_granger, DFGLS
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Constant(s)
VARIABLES = ('mean_rad', 'mean_tmn', 'NY.GDP.MKTP.KD', 'SP.POP.TOTL')
TITLES = ('Annual average of monthly total rainfall (millimetres per month)', 'Annual average temperature (degrees Celsius)', 'Agricultural land (sq. km)', 'Crop production index (2014-2016 = 100)')


# Helper function(s)
def parse_data(path):
    data = pd.read_csv(path, parse_dates=['year'])
    data = data[data['cntry.name'] == 'France']
    data['log_NY.GDP.MKTP.KD'] = np.log(data['NY.GDP.MKTP.KD'])
    data['log_SP.POP.TOTL'] = np.log(data['SP.POP.TOTL'])
    return data


###Function for residual based test, Mckinnon t-stat
def mckinnon(dep, df):
    x = sm.tsa.add_trend(df, trend='ct')
    ols_res = sm.OLS(dep, x).fit()
    e = ols_res.resid
    e_diff = ols_res.resid.diff()
    T = len(e)
    ar_res = sm.tsa.AutoReg(e, 1).fit()
    u = ar_res.resid
    T_rho1 = (np.dot(e[:-1], e_diff[1:]) / T) / (np.dot(e[:-1], e[:-1]) / T ** 2)  ##slide 140
    t_stat = T_rho1 * np.sqrt(np.dot(e[:-1], e[:-1]) / T ** 2) / np.std(u)

    return t_stat


def Johansen_Max_eigen(data, c_or_t, lags):
    # perform Johansen cointegration test on all 4 variables
    # c_or_t = 1 corresponds to 'constant and trend'
    jt = coint_johansen(data, c_or_t, lags)

    print('t-stat for Johansen trace test\n', jt.trace_stat)
    print('Intervals for Johansen trace test\n', jt.trace_stat_crit_vals)
    print('t-stat for Max Eigenvalue test', jt.max_eig_stat)
    print('Intervals for Max Eigenvalue test \n', jt.max_eig_stat_crit_vals)

    return jt


def part4():
    data = parse_data('VU_MultivariateEcnmtrcs_assignment_dataset.csv')
    endog = data['log_NY.GDP.MKTP.KD']
    exog = data[['mean_rad','log_SP.POP.TOTL', 'mean_tmn',]]

    # Engle-Granger test
    eg_res = engle_granger(endog, exog, trend='ct')
    print(f'Engle-Granger test results: {eg_res}\n')

    # Philips-Ouliaris' test
    po_res = cointegration.phillips_ouliaris(endog, exog, trend='ct')
    print(f'Philips-Ouliaris test results: {po_res}\n')

    # ADF-type test (McKinnon)
    stat = mckinnon(endog, exog)
    print(stat)

    # static OLS
    exog_w_const = sm.add_constant(exog)
    static_ols = sm.OLS(endog, exog_w_const).fit()
    print(static_ols.summary())

    # dynamic OLS
    dynamic_ols = cointegration.DynamicOLS(endog, exog, trend='c').fit()
    print(dynamic_ols.summary())

    # fully modified OLS
    fully_modified_ols = cointegration.FullyModifiedOLS(endog, exog, trend='c').fit()
    print(fully_modified_ols.summary())

    # VECM model
    result_ECM = sm.tsa.VECM(data[['log_NY.GDP.MKTP.KD', 'log_SP.POP.TOTL', 'mean_tmn', 'mean_rad']], coint_rank=1, deterministic='ci').fit()
    print(result_ECM.summary())

    # Use system approach and Johansen's analysis
    # Johansen trace test
    johansen_df = data[['log_NY.GDP.MKTP.KD', 'log_SP.POP.TOTL', 'mean_tmn', 'mean_rad']]
    johansen_res = Johansen_Max_eigen(johansen_df, 1, 1)

    # check stationarity
    ts = (johansen_res.evec[0, 0] * data['log_NY.GDP.MKTP.KD']) + (johansen_res.evec[0, 1] * data['log_SP.POP.TOTL']) + (johansen_res.evec[0, 2] * data['mean_tmn']) + (
                johansen_res.evec[0, 3] * data['mean_rad'])
    plt.plot(ts)
    plt.show()
    print(DFGLS(ts))



if __name__ == '__main__':
    part4()
