import numpy as np
import newton
import utils


def f(x):
    return f_2d(x[0], x[1])


def fp(x):
    y1 = np.exp(x[0]+3*x[1]-0.1) + np.exp(x[0]-3*x[1]-0.1) - np.exp(-x[0]-0.1)
    y2 = 3*np.exp(x[0]+3*x[1]-0.1) - 3*np.exp(x[0]-3*x[1]-0.1)
    return np.array([y1, y2])


def fpp(x):
    y11 = np.exp(x[0]+3*x[1]-0.1) + np.exp(x[0]-3*x[1]-0.1) + np.exp(-x[0]-0.1)
    y12 = 3*np.exp(x[0]+3*x[1]-0.1) - 3*np.exp(x[0]+3*x[1]-0.1)
    y21 = 3*np.exp(x[0]+3*x[1]-0.1) - 3*np.exp(x[0]+3*x[1]-0.1)
    y22 = 9*np.exp(x[0]+3*x[1]-0.1) + 9*np.exp(x[0]-3*x[1]-0.1)
    return np.array([[y11, y12], [y21, y22]])


def f_2d(x1, x2):
    return np.exp(x1+3*x2-0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1-0.1)


# use the value you find in HW7
f_opt = 4 * np.exp(-0.1) / np.sqrt(2)


def gap(x):
    return f(x) - f_opt


def draw_plot(x0, path):
    # Newton
    x_traces = newton.newton(fp, fpp, x0)
    f_value = f(x_traces[-1])

    print(f([-np.log(2)/2, 0]))
    print()
    print("Newton's method")
    print('  number of iterations:', len(x_traces)-1)
    print('  solution:', x_traces[-1])
    print('  value:', f_value)

    utils.plot_traces_2d(f_2d, x_traces, path+'nt_traces.png')
    utils.plot(gap, x_traces, path+'nt_gap.png')


draw_plot(np.array([-1.5, 1.0]), 'figures/1a/')
draw_plot(np.array([1.5, 1.0]), 'figures/1b/')
