import newton as nt
import numpy as np


def centering_step(c, A, b, x0, t):
    """
    c, A, b: parameters in LP

            min   c^T x
            s.t.  Ax = b
                  x >= 0

    x0: feasible initial point for constrained Newton's method
    t:  penalty parameter in barrier method

    This function returns the central point x^*(t) 
    """

    def f(x):
        return np.dot(c, x) - np.sum(np.log(x)) / t

    def fp(x):
        return c - 1 / x / t

    def fpp(x):
        return np.diag(x ** -2) / t

    return nt.newton_eq_2(f, fp, fpp, x0, A, b)[-1]


def barrier(c, A, b, x0, tol=1e-8, t0=1, rho=10):
    """
    c, A, b: parameters in LP

            min   c^T x
            s.t.  Ax = b
                  x >= 0

    x0:  feasible initial point for the barrier method
    tol: tolerance parameter for the suboptimality gap. The algorithm stops when

             f(x) - f^* <= tol

    t0:  initial penalty parameter in barrier method
    rho: factor by which the penalty parameter is increased in each centering step

    This function should return a list of the iterates
    """
    t = t0
    x = np.array(x0)
    x_traces = [np.array(x0)]

    m = len(c)
    while True:
        x = centering_step(c, A, b, x, t)
        x_traces.append(x.copy())
        if m < t * tol:
            break
        t *= rho

    return x_traces
