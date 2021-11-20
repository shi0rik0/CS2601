import numpy as np
import matplotlib.pyplot as plt


def plot_traces_2d(f_2d, x_traces, filename):
    fig = plt.figure(figsize=(5, 4))

    plt.plot(*zip(*x_traces), '-o', color='red')

    x1, x2 = zip(*x_traces)
    x1 = np.arange(min(x1)-.2, max(x1)+.2, 0.01)
    x2 = np.arange(min(x2)-.2, max(x2)+.2, 0.01)
    x1, x2 = np.meshgrid(x1, x2)
    plt.contour(x1, x2, f_2d(x1, x2), 20, colors='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.tight_layout()

    fig.savefig(filename)


def plot(data,  x_label, y_label, filename, logscale=True):
    fig = plt.figure(figsize=(5, 4))

    if logscale:
        plt.semilogy(data)
    else:
        plt.plot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    fig.savefig(filename)
