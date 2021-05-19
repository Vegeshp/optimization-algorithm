from typing import List
import numpy as np
from matplotlib import pyplot as plt
from math import sin, pi

def f(x):
    return sin(2 * pi * x)


def h(x, theta):
    n = theta.shape[0] - 1
    result = 0
    for i in range(n + 1):  # from 0 to n
        result += theta[i] * x ** i
    return result


def loss(x_list, y_list, theta, lmd = 1e-8):
    sum = 0
    for x, y in zip(x_list, y_list):
        sum += (h(x, theta) - y) ** 2
    for i in range(theta.shape[0]):
        sum += lmd * abs(theta[i])
    return sum / 2


def grad_descent(x_list, y_list, theta, l):
    alpha = 1e-2
    n = theta.shape[0] - 1
    result = np.zeros(n + 1)
    for k in range(n + 1):
        for x, y in zip(x_list, y_list):
            result[k] += x ** k * (h(x, theta) - y)
    temp = theta - alpha * result
    return temp, l(x_list, y_list, temp)


def l1_grad_descent(x_list, y_list, theta, l = loss):
    alpha = 1e-2
    n = theta.shape[0] - 1
    result = np.copy(theta)
    prev = l(x_list, y_list, result)
    for k in range(n + 1):
        while True:
            result[k] += alpha
            temp_loss = l(x_list, y_list, result)
            if prev <= temp_loss:
                result[k] -= alpha
                break
            prev = temp_loss
        while True:
            result[k] -= alpha
            temp_loss = l(x_list, y_list, result)
            if prev <= temp_loss:
                result[k] += alpha
                break
            prev = temp_loss
    return result, prev


def fit(x_list: List[int], y_list: List[int], k: int = 4, descent = grad_descent, l = loss):
    count = 0
    max_round = 10000000
    theta = np.zeros(k + 1)
    while count < max_round:
        temp, cost = descent(x_list, y_list, theta, l)
        # if np.linalg.norm(temp - theta) < np.linalg.norm(theta) * eps:
        if cost <= 1:
            break
        count += 1
        theta = temp
        if count % 10000 == 0:
            print(count, cost)
    # print(theta)
    return theta


if __name__ == '__main__':
    n = 30
    x_list = np.linspace(0, 1, n)
    y_list = np.sin(2 * np.pi * x_list)
    plt.plot(x_list, y_list)
    plt.savefig('./data/sin.jpg')

    theta = fit(x_list, y_list, 30, grad_descent)
    new_y_list = h(x_list, theta)
    plt.plot(x_list, new_y_list, color='r')
    plt.savefig('./data/gradient_descent.jpg')

    theta = fit(x_list, y_list, 30, l1_grad_descent)
    new_y_list = h(x_list, theta)
    plt.plot(x_list, new_y_list, color='g')
    plt.savefig('./data/lasso.jpg')
