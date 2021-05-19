import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker


def f(x, y):
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2


def grad(x, y):
    return np.matrix([[2 * x - 2 + 400 * x * (x * x - y)],
                      [200 * (y - x * x)]])


def H(x, y):
    return np.matrix([[1200 * x * x - 400 * y + 2, -400 * x],
                      [-400 * x, 200]])


def gradient_descent_delta(x, y):
    return 1e-3 * grad(x, y)


def newton_delta(x, y):
    return H(x, y).I * grad(x, y)


def optimize(delta_func):
    step_count = 20000
    count = 0
    eps = 1e-8

    x = np.mat([-1, 1]).T
    x_list, y_list = [x[0, 0]], [x[1, 0]]
    while count < step_count:
        delta = delta_func(x[0, 0], x[1, 0])
        if np.linalg.norm(delta) < eps:
            break
        x = x - delta
        x_list.append(x[0, 0])
        y_list.append(x[1, 0])
        count += 1
    print(f'round = {count}')
    return x_list, y_list


if __name__ == '__main__':
    for name, func in zip(['./data/1.jpg', './data/2.jpg'], [gradient_descent_delta, newton_delta]):
        n = 1000
        x = np.linspace(-1, 1.1, n)
        y = np.linspace(-1, 1.1, n)
        X, Y = np.meshgrid(x, y)
        plt.figure()
        plt.contourf(X, Y, f(X, Y), 5, alpha=0, cmap=plt.cm.hot)
        plt.clabel(plt.contour(X, Y, f(X, Y), 8, locator=ticker.LogLocator(),
                               colors='black'), inline=True, fontsize=10)

        x_list, y_list = optimize(func)
        plt.plot(x_list, y_list)
        plt.savefig(name)
        print(f'({x_list[-1]:.3f}, {y_list[-1]:.3f})')
