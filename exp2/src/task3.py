import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

IMAGE_SIZE = 60  # size of zipped image


def pca(data, dim):
    mean = np.mean(data, axis=0)
    mat = data - mean
    cov = np.cov(mat, rowvar=0)
    eigenVal, eigen_vec = np.linalg.eig(np.mat(cov))
    index = np.argsort(eigenVal)[:-(dim + 1):-1]
    eigen_vec = eigen_vec[:, index]
    low_data = mat * eigen_vec
    return low_data, low_data * eigen_vec.T + mean


def psnr(a, b):
    return np.abs(20 * np.log10(255.0 / np.sqrt(np.mean((b-a)**2))))


def read(path):
    image = Image.open(path).convert('L')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE),
                         Image.ANTIALIAS)
    return np.real(np.mat(image))


if __name__ == '__main__':
    path = ['./jaffe/' + str(i + 1) + '.tiff' for i in range(23)]
    pictures = np.array([read(i) for i in path])
    print(pictures[0].shape)
    pictures = pictures.reshape(
        [pictures.shape[0], pictures.shape[1] * pictures.shape[1]])
    ans = pca(pictures, 5)
    result = ans[1]
    print(result[0].shape)
    # for i in range(len(result)):
    #     pic = pictures[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
    #     res = result[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
    #     print(psnr(pic, res))
