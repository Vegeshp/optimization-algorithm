import cv2
import numpy as np
from task2 import fit, l1_grad_descent


origin_image = cv2.imread('./data/lena.jpg')
n, m, channel = origin_image.shape
sigma = 20
noise_image = np.uint8(np.clip(np.random.normal(0, sigma, origin_image.shape)
                               + origin_image, 0, 255))
cv2.imwrite('./data/noise_lena.jpg', noise_image)

noise_image = noise_image / 255

temp_image = noise_image.copy()


def new_loss(x_list, y_list, theta):
    result = 0
    for i in range(len(x_list)):
        sum = y_list[i] - (theta[0] + theta[1] * x_list[i]
                           [0] + theta[2] * x_list[i][1])
        result += sum ** 2
    return result


for i in range(n):
    print(i)
    for j in range(m):
        y_list = []
        x_list = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                if i + x < 0 or i + x >= n or j + y < 0 or j + y >= m:
                    continue
                x_list.append([x, y])
                y_list.append(temp_image[i + x, j + y])
        x_list = np.array(x_list)
        for r in range(channel):
            new_y_list = [y[r] for y in y_list]
            noise_image[i, j, r] = fit(
                x_list, new_y_list, 2, l1_grad_descent, new_loss)[0]
noise_image = np.uint8(255 * np.clip(noise_image, 0, 1))
cv2.imwrite('./regression_lena.jpg', noise_image)
