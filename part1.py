# Assignment Multivariate Econometrics 
# Group 23: Atanas Atanasov, Busra Turk, Ion Paraschos, Max Hugen

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


# Helper function(s)
def DGPQ1(T, delta, lamda):
    # Simulate x_t until T=200 with innovations
    x = np.ones((3, T))
    eps = np.random.normal(loc=0, scale=1, size=(3, T))
    for t in np.arange(1, T):
        x[:, t] = delta + np.dot(lamda, x[:, t - 1]) + eps[:, t]
    return x


def DGPQ6(T, lamda):
    # Simulate x_t until T=200 with innovations
    x = np.zeros((3, T))
    eps = np.random.normal(loc=0, scale=1, size=(3, T))
    for t in np.arange(1, T):
        x[:, t] = np.dot(lamda, x[:, t - 1]) + eps[:, t]
    return x


def DGPQ9(T):
    # Simulate x_t until T=200 with innovations
    x = np.zeros(T)
    eps = np.random.normal(loc=0, scale=1, size=T)
    for t in np.arange(1, T):
        x[t] = x[t - 1] + eps[t]
    return x

def get_var_mat(lamda):
    # get unconditional var.-covar. matrix
    lamda_mul = np.kron(lamda, lamda)
    par = np.identity(9) - lamda_mul
    inverse_par = np.linalg.inv(par)
    var_x = np.dot(inverse_par, np.identity(3).flatten())
    var_x = np.reshape(var_x, (3,3))
    return(var_x)


def get_mean(delta, lamda):
    # get unconditional mean using formula from the slides
    factor = np.linalg.inv(np.identity(3) - lamda)
    uncond_mean = np.dot(factor, delta)
    return uncond_mean


# Questions(s)
def q1(x):
    # generate data using DGP in assignment (part 1.1)
    print('QUESTION 1')
    print(x)


def q2(lamda, n):
    # check stability of lamda (part 1.2)
    print('QUESTION 2')
    for _ in range(n):
        lamda = np.dot(lamda, lamda)
    print(f'Lambda^{n}:\n{lamda}')


def q3(x):
    # plot time series data (part 1.3)
    print('QUESTION 3')
    for i in np.arange(3):
        plt.plot(x[i, :])
    plt.legend(('$x_1$','$x_2$','$x_3$'))
    plt.title('VAR(1) process over time')
    plt.xlabel('Time (t)')
    plt.ylabel('Value of $x_t$')
    plt.show()


def q4(x, delta, lamda):
    # compute mean and avg of time series data (part 1.4)
    print('QUESTION 4')
    print(f'average : {np.mean(x, axis=1)}')
    print(f'unconditional_mean : {get_mean(delta, lamda)}')


def q5(x, lamda):
    # compute var (with innovations) and var (without innovations) of time series data (part 1.5)
    print('QUESTION 5')
    print(f'Conditional_variance : {np.var(x, axis=1)}')
    print(f'Unconditional_variance: {get_var_mat(lamda)}')


def q6(x):
    # generate data using DGP in assignment (part 1.6)
    print('QUESTION 6')
    print(x)


def q7(x, n, lamda):
    print('QUESTION 7')

    # check stability
    for _ in range(n):
        lamda = np.dot(lamda, lamda)
    print(f'Lambda^{n}:\n{lamda}')

    # dickey-fuller
    pvals = np.empty(3)
    for i in np.arange(3):
        pvals[i] = sm.tsa.adfuller(x[i, :])[1]
    print(f'p-vals: {pvals}')
    # if pval > 0.05:  # adf[1] is p-val of ADF-test
    #     print(f'We fail to reject the null hypothesis that there is a unit root with a p-value of {pval}.')
    # else:
    #     print(f'We reject the null hypothesis that there is a unit root with a p-value of {pval}.')


def q8(x):
    # plot time series data (part 1.8)
    print('QUESTION 8')
    for i in np.arange(3):
        plt.plot(x[i, :])
    plt.legend(('$x_1$', '$x_2$', '$x_3$'))
    plt.title('VAR(1) process over time')
    plt.xlabel('Time (t)')
    plt.ylabel('Value of $x_t$')
    plt.show()


def q9(x):
    print('QUESTION 9')
    print(x)


def q10(x):
    print('QUESTION 10')
    print(x)
    plt.plot(x, c='black')
    plt.title('AR(1) process over time')
    plt.ylabel('Value of $x_t$')
    plt.xlabel('Time (t)')
    plt.show()


def q11(x):
    adf = sm.tsa.adfuller(x)
    print('QUESTION 11')
    pval = adf[1]
    if pval > 0.05: # adf[1] is p-val of ADF-test
        print(f'We fail to reject the null hypothesis that there is a unit root with a p-value of {pval}.')
    else:
        print(f'We reject the null hypothesis that there is a unit root with a p-value of {pval}.')


def part1():
    np.random.seed(123)

    # initialize parameters for DGP (Q1)
    delta_q1 = np.ones(3)
    lamda_q1 = np.array([[0.1, 0.1, 0.1],
                         [0.1, 0.2, 0.1],
                         [0.1, 0.1, 0.3]])
    lamda_q6 = np.array([[0.2, 0.2, 0.6,],
                         [0.3, 0, 0.7],
                         [0.5, 0.1, 0.4]])
    T = 200

    # compute time series data
    x_q1 = DGPQ1(T, delta_q1, lamda_q1)
    x_q6 = DGPQ6(T, lamda_q6)
    x_q9 = DGPQ9(T)

    # questions
    q1(x_q1)
    q2(lamda_q1, 10000)
    q3(x_q1)
    q4(x_q1, delta_q1, lamda_q1)
    q5(x_q1, lamda_q1)

    q6(x_q6)
    q7(x_q6, 10000, lamda_q6)
    q8(x_q6)
    #
    q9(x_q9)
    q10(x_q9)
    q11(x_q9)


if __name__ == '__main__':
    part1()
