import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, log


def ADMM(A, y):
    max_round = 10000
    count = 0

    _, n = A.shape
    z_hat, u = np.zeros([n, 1]), np.zeros([n, 1])

    A_t_A = A.T.dot(A)
    w, _ = np.linalg.eig(A_t_A)
    r = np.max(np.absolute(w))
    ratio = sqrt(2 * log(n, 10)) * r / 2

    A_t_y = A.T.dot(y)
    Q = np.linalg.inv(A_t_A + np.identity(n) / r)

    while count < max_round:
        u = u + Q.dot(A_t_y + (z_hat - u) / r)
        z_hat = np.sign(u) * np.maximum(0, np.absolute(u) - ratio)
        u = u - z_hat
        if count % 1000 == 0:
            print(count)
        count += 1
    return z_hat


if __name__ == '__main__':
    m, n, non_zero = 50, 200, 10
    x = np.zeros((n, 1))
    x[np.random.randint(0, n, non_zero)] = 100 * np.random.randn(non_zero, 1)

    plt.plot(x, color='r')  # original
    A = np.random.randn(m, n)
    result = ADMM(A, A.dot(x) + np.random.randn(m, 1))
    plt.plot(result, color='g')  # estimate
    plt.savefig('./admm.jpg')
