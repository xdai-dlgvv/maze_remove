import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('1.jpg')
# im = cv2.imread('3.png')
im = np.asarray(im, dtype=np.float)
patch_size = (7 - 1) // 2
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
        t_[i, j] = 1 - 0.95 * np.min(im[i, j, :]/A)

L = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        print(i, ',', j)
        l = 0
        center = im[i, j]
        Ii = np.mat(center)
        patch = im[max(i - patch_size, 0):min(i + patch_size, height), max(j - patch_size, 0):min(j + patch_size, width), :]
        patch = np.reshape(patch, (-1, 3))
        wk = np.shape(patch)[0]
        miu = np.mean(patch, axis=0)
        # delta = np.var(patch)
        delta = np.cov(patch.T)
        epsilon = 1 #  ？？？？？
        U3 = np.eye(3)
        for k in range(wk):
            Ij = np.mat(patch[k])
            kronecker_delta = 1 if (Ii == Ij).all() else 0
            tmp = (delta + epsilon / wk * U3)**(-1)
            t2 = (Ii - miu)
            t3 = (Ij - miu).T
            l += kronecker_delta - (1/wk) * (Ii - miu) * (delta + (epsilon / wk) * U3)**(-1) * (Ij - miu).T
        L[i, j] = l
    if i > 20:
        break
lambda_ = 0.0001
U = np.eye(min(width, height)) + np.zeros((height, width))
t = (L + lambda_ * U) * lambda_ * t_
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
