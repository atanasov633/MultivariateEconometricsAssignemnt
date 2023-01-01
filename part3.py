# Assignment Multivariate Econometrics 
# Group 23: Atanas Atanasov, Busra Turk, Ion Paraschos, Max Hugen


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch.unitroot import PhillipsPerron, DFGLS


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


def seasonal_decomposition(data):
    for var in ('mean_rad', 'mean_tmn', 'log_NY.GDP.MKTP.KD', 'log_SP.POP.TOTL'):
        decomposition = sm.tsa.seasonal_decompose(data[var], model='additive', period=12)
        decomposition.plot()
        plt.show()


def unit_root_tests(data):
    for var in ('mean_rad', 'mean_tmn', 'log_NY.GDP.MKTP.KD', 'log_SP.POP.TOTL'):
        adf_res = sm.tsa.adfuller(data[var], regression='ct')
        dfgls_res = DFGLS(data[var], trend='ct')
        pp_res = PhillipsPerron(data[var], trend='ct')
        kpss_res = sm.tsa.kpss(data[var], regression='ct')

        print(f'Augmented-Dickey-Fuller test ({var}):\n{adf_res}\n')
        print(f'DFGLS test ({var}):\n{dfgls_res}')
        print(f'Philips Perron test ({var}):\n{pp_res}\n')
        print(f'KPSS test ({var}):\n{kpss_res}\n')

def chow_test(data):
    year = np.arange(1961,2017)

    # res is the Sum of Squared Residuals for the whole period
    res = sm.OLS(data['NY.GDP.MKTP.KD'].apply(np.log), year).fit()
    res_total = res.ssr
    print("SSR 1961 - 2014: ", res_total)

    # SSR for period 1 (1961- 1974)
    pop_before_74 = data['NY.GDP.MKTP.KD'][0:14].apply(np.log);
    year_before_74 = year[0:14]
    # SSR for period 2 (1974-2016)
    pop_after_74 = data['NY.GDP.MKTP.KD'][15:].apply(np.log);
    year_after_74 = year[15:]

    # calculate residuals for the sub periods 1974:
    res_before = sm.OLS(pop_before_74, year_before_74).fit()
    res_before_1 = res_before.ssr
    res_after = sm.OLS(pop_after_74, year_after_74).fit()
    res_after_1 = res_after.ssr

    # Check the validity of the test:
    numerator = (res_total - (res_before_1 + res_after_1)) / 2
    # k - degrees of freedom = number of OLS coeff. (in our case 2, because of a constant and the coeff. before yt-1)  (55 - 2 * k)
    denominator = (res_before_1 + res_after_1) / 51
    chow_test_1 = numerator / denominator
    print('result for 1974:', chow_test_1)

    # SSR for period 1 (1961- 1993)
    pop_before_93 = data['NY.GDP.MKTP.KD'][0:33].apply(np.log); year_before_93 = year[0:33]
    # SSR for period 2 (1993-2016)
    pop_after_93 = data['NY.GDP.MKTP.KD'][34:].apply(np.log); year_after_93 = year[34:]

    # calculate residuals for the sub periods 1993:
    res_before = sm.OLS(pop_before_93, year_before_93).fit()
    res_before_2 = res_before.ssr
    res_after = sm.OLS(pop_after_93, year_after_93).fit()
    res_after_2 = res_after.ssr

    # Check the validity of the test:
    numerator = (res_total - (res_before_2 + res_after_2)) / 2
    # k - degrees of freedom = number of OLS coeff. (in our case 2, because of a constant and the coeff. before yt-1)  (
    # 55 - 2 * k)
    denominator = (res_before_2 + res_after_2) / 51
    chow_test_2 = numerator / denominator
    print('result for 1993:', chow_test_2)

    # SSR for period 1 (1961- 2009)
    pop_before_09 = data['NY.GDP.MKTP.KD'][0:49].apply(np.log);
    year_before_09 = year[0:49]
    # SSR for period 2 (2009-2016)
    pop_after_09 = data['NY.GDP.MKTP.KD'][50:].apply(np.log);
    year_after_09 = year[50:]

    # calculate residuals for the sub periods 2009:
    res_before = sm.OLS(pop_before_09, year_before_09).fit()
    res_before_3 = res_before.ssr
    res_after = sm.OLS(pop_after_09, year_after_09).fit()
    res_after_3 = res_after.ssr

    # Check the validity of the test:
    numerator = (res_total - (res_before_3 + res_after_3)) / 2
    # k - degrees of freedom = number of OLS coeff. (in our case 2, because of a constant and the coeff. before yt-1)  (
    # 55 - 2 * k)
    denominator = (res_before_3 + res_after_3) / 51
    chow_test_3 = numerator / denominator
    print('result for 2009:', chow_test_3)

    plt.plot(year, data['NY.GDP.MKTP.KD'])
    plt.axvline(x=1974, c='red')
    plt.axvline(x=1993, c='red')
    plt.axvline(x=2009, c='red')
    plt.show()

def part3():
    data = parse_data('VU_MultivariateEcnmtrcs_assignment_dataset.csv')

    seasonal_decomposition(data)
    unit_root_tests(data)
    chow_test(data)


if __name__ == '__main__':
    part3()
