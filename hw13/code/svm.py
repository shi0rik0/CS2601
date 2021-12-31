import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import zeros_like
import proj_gd as gd


def projection(y, z):
    u = z[y > 0]
    u.sort()
    w = -z[y < 0]
    w.sort()
    p = len(u)
    m = len(w)
    if p == 0 or m == 0:
        return np.zeros([p + m])
    U = np.concatenate([[0], np.cumsum(u[::-1])])[::-1]
    W = np.concatenate([[0], np.cumsum(w)])
    u = np.concatenate([[-np.inf], u, [np.inf]])
    w = np.concatenate([[-np.inf], w, [np.inf]])
    k = 0
    l = 0
    while k <= p and l <= m:
        lam = (U[k]+W[l]) / (p-k+l)
        if u[k] <= lam and lam <= u[k+1] and w[l] <= lam and lam <= w[l+1]:
            break
        if u[k+1] < w[l+1]:
            k += 1
        else:
            l += 1
    return np.maximum(0, z-lam*y)


def svm(X, y):
    """
    X: n x m matrix, X[i,:] is the m-D feature vector of the i-th sample
    y: n-D vector, y[i] is the label of the i-th sample, with values +1 or -1

    This function returns the primal and dual optimal solutions w^*, b^*, mu^*
    """
    Xy = X * y
    Q = Xy @ Xy.T

    def fp(mu):
        # f(mu) = 0.5 * mu.T@Q@mu - np.sum(mu)
        return Q@mu - np.ones_like(mu)

    def proj(mu):
        return projection(y, mu)

    mu0 = np.zeros_like(y)
    mu_traces, _ = gd.proj_gd(fp, proj, mu0, stepsize=0.1, tol=1e-8)
    mu = mu_traces[-1]

    n = len(y)
    w = zeros_like(X[0, :])
    for i in range(n):
        w += mu[i] * y[i] * X[i, :]

    for i in range(n):
        if mu[i] > 0:
            b = y[i] - np.dot(X[i, :], w)
            break

    return w, b, mu
