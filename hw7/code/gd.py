import numpy as np


def gd_const_ss(fp, x0, stepsize, tol=1e-5, maxiter=100000):
    x_traces = [np.array(x0)]
    x = np.array(x0)
    g = fp(x)
    while np.linalg.norm(g) >= tol and len(x_traces) <= maxiter:
        x -= stepsize * g
        x_traces.append(x.copy())
        g = fp(x)
    return x_traces


def gd_armijo(f, fp, x0, t0, alpha, beta, tol=1e-5, maxiter=100000):
    x_traces = [np.array(x0)]
    ts = []
    x = np.array(x0)
    g = fp(x)
    n_inner_loop = 0
    while np.linalg.norm(g) >= tol and len(x_traces) <= maxiter:
        t = t0
        while f(x - t*fp(x)) > f(x) - alpha*t*np.linalg.norm(fp(x))**2:
            t = beta * t
            n_inner_loop += 1
        ts.append(t)
        x -= t * fp(x)
        x_traces.append(x.copy())
        g = fp(x)
    return x_traces, ts, n_inner_loop
