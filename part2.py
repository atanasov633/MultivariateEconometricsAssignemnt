# Assignment Multivariate Econometrics 
# Group 23: Atanas Atanasov, Busra Turk, Ion Paraschos, Max Hugen


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constant(s)
VARIABLES = ('mean_rad', 'mean_tmn', 'NY.GDP.MKTP.KD', 'SP.POP.TOTL')
TITLES = ('Annual average of monthly all-sky radiation (SSR) (watts per square metre)', 'Minimum of averages of monthly temperatures (degrees Celsius)', 'GDP (constant 2010 US$)', 'Population, total')


# Helper function(s)
def parse_data(path):
    data = pd.read_csv(path, parse_dates=['year'])
    data = data[data['cntry.name'] == 'France']
    return data


def plot(data):
    # plot levels
    print('Plotting levels...')
    fig, axs = plt.subplots(2,2)
    for ax, var, title in zip(axs.flat, VARIABLES, TITLES):
        ax.plot(data['year'], data[var], color='black')
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    # plot logs
    print('Plotting logs...')
    fig, axs = plt.subplots(2, 2)
    for ax, var, title in zip(axs.flat, VARIABLES, TITLES):
        ax.plot(data['year'], np.log(data[var]), color='black')
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    # plot first differences
    print('Plotting first differences...')
    fig, axs = plt.subplots(2, 2)
    for ax, var, title in zip(axs.flat, VARIABLES, TITLES):
        ax.plot(data['year'], data[var].diff(), color='black')
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def part2():
    data = parse_data('VU_MultivariateEcnmtrcs_assignment_dataset.csv')
    plot(data)


if __name__ == '__main__':
    part2()
