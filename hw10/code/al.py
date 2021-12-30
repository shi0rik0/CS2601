import numpy as np


def solve(y, z):
    u = z[y > 0]
    u.sort()
    w = -z[y < 0]
    w.sort()
    p = len(u)
    m = len(w)
    # 这个判断好像可以去掉？
    if p == 0 or m == 0:
        return np.zeros([p + m])
    U = np.concatenate([[0], np.cumsum(u[::-1])])[::-1]
    W = np.concatenate([[0], np.cumsum(w)])
    u = np.concatenate([[-np.inf], u, [np.inf]])
    w = np.concatenate([[-np.inf], u, [np.inf]])
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


y = np.array([1, 1, -1])
z = np.array([1, 2, 1])
print(solve(y, z))
