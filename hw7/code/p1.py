from gd import gd_armijo, gd_const_ss
import utils

import numpy as np


def f(x):
    return np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)


def fp(x):
    fp1 = np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] -
                                               3*x[1] - 0.1) - np.exp(-x[0] - 0.1)
    fp2 = 3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1)
    return np.array([fp1, fp2])


def f_2d(x1, x2):
    return np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)


x0 = [1.5, 1]
t0 = 1
alpha = 0.1
beta = 0.7

x_traces, ts, n_inner_loop = gd_armijo(f, fp, x0, t0, alpha, beta)

print(f'solution: {x_traces[-1]}')
print(f'number of iterations in the outer loop: {len(x_traces)}')
print(f'number of iterations in the inner loop: {n_inner_loop}')
utils.plot_traces_2d(f_2d, x_traces, 'figures/f1.png')

f_min = 2 * np.sqrt(2) * np.exp(-0.1)
f_traces = [f(x) - f_min for x in x_traces]
utils.plot(f_traces, 'Number of Iterations', 'f(x) - f(x*)', 'figures/f2.png')
utils.plot(ts, 'Number of Iterations', 't', 'figures/f3.png', logscale=False)

step_size = 0.1
x_traces = gd_const_ss(fp, x0, step_size)

print(f'solution: {x_traces[-1]}')
print(f'number of iterations: {len(x_traces)}')
