import numpy as np
import matplotlib.pyplot as plt
import proj_gd as gd
import svm

X = np.genfromtxt('code/X.dat')
X = X.reshape(-1, 2)

y = np.genfromtxt('code/y.dat')*2-1  # convert labels to +1 and -1
y = y.reshape(-1, 1)

w, b, mu = svm.svm(X, y)

print("primal optimal:")
print(f"	w = {np.squeeze(w)}")
print(f"	b = {b}\n")
print("dual optimal:")
print(f"	mu = {np.squeeze(mu)}")

# visualization
xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
yp = - (w[0]*xp + b)/w[1]
# margin boundary for support vectors for y=1
yp1 = np.array(- (w[0]*xp + b-1)/w[1])
# margin boundary for support vectors for y=0
yp0 = np.array(- (w[0]*xp + b+1)/w[1])

idx0 = np.where(y == -1)
idx1 = np.where(y == 1)

plt.axis('equal')
plt.plot(X[idx0[0], 0], X[idx0[0], 1], 'ro')
plt.plot(X[idx1[0], 0], X[idx1[0], 1], 'go')
plt.plot(xp, yp, '-b', xp, yp1, '--g', xp, yp0, '--r')
plt.title('decision boundary for linear SVM')
plt.tight_layout()
plt.savefig('figures/svm.png')
