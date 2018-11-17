import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('8.png')
# im = cv2.imread('6.png')
im = np.asarray(im, dtype=np.float)
patch_size = (15 - 1) // 2
t0 = 0.1
height, width, channel = np.shape(im)


j_dark = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        patch = im[max(i - patch_size, 0):min(i + patch_size, height), max(j - patch_size, 0):min(j + patch_size, width), :]
        j_dark[i, j] = np.min(patch)
dark_channel_list = np.sort(np.reshape(j_dark, (-1, 1)))
dark_channel_max = dark_channel_list[int(len(dark_channel_list) * 0.001)]
A = np.max(im[j_dark>dark_channel_max, :], axis=0)

j_map = np.zeros((height, width, 3))

t_ = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        # patch = im[max(i - patch_size, 0):min(i + patch_size, height), max(j - patch_size, 0):min(j + patch_size, width), :]
        # t_[i, j] = 1 - 0.95 * np.min(patch/A)
        t_[i, j] = 1 - 0.8 * np.min(im[i, j, :]/A)


L = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        l = 0
        center = im[i, j]
        patch = im[max(i - patch_size, 0):min(i + patch_size, height), max(j - patch_size, 0):min(j + patch_size, width), :]
        patch = np.reshape(patch, (-1, 3))
        wk = np.shape(patch)[0]
        miu = np.mean(patch, axis=0)
        Ii = np.repeat(np.mat(center - miu), wk, axis=0)
        Ij = np.mat(patch) - np.repeat(np.mat(miu), wk, axis=0)
        # delta = np.var(patch)
        delta = np.mat(np.cov(patch.T))
        epsilon = 0.0001 #  ？？？？？
        U3 = np.mat(np.eye(3))
        V = (delta + epsilon/wk*U3).I
        L[i, j] = 1 - 1 - 1 / wk * np.trace(Ii * V * Ij.T)


lambda_ = 0.0001
U = np.zeros((height, width))
m = min(width, height)
U[0:m, 0:m] = np.eye(m)
t = np.mat(L + lambda_ * U).I * lambda_ * t_
# t = (L + lambda_ * U) ** (-1) * lambda_ * t_
# t = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        j_map[i, j, :] = (im[i, j, :] - A) / max(t[i, j], t0) + A


# plt.subplot(121)
# plt.imshow(im)
# plt.subplot(122)
# plt.imshow(j_dark)
# plt.show()
# A = np.max()
cv2.imshow(u'raw image', np.asarray(im, dtype=np.uint8))
cv2.imshow(u'result', np.asarray(j_map, dtype=np.uint8))
cv2.imshow(u'dark channel', np.asarray(j_dark, dtype=np.uint8))
cv2.imshow(u'transmission', np.asarray(t*255, dtype=np.uint8))

# plt.imshow(j_dark)
# plt.show()
cv2.waitKey(0)
