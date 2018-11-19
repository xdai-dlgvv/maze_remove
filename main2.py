import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from PIL import ImageFilter, Image

im_name = '4.jpg'
im_path = 'sample'
save_path = 'output'
im = cv2.imread(os.path.join(im_path, im_name))
im = np.asarray(im, dtype=np.float)
filter_size = 15
patch_size = (filter_size - 1) // 2
t0 = 0.1
omega = 0.9
dark_channel_maximum = 220
height, width, channel = np.shape(im)

# j_dark = np.zeros((height, width))

# for i in range(height):
#     for j in range(width):
#         patch = im[max(i - patch_size, 0):min(i + patch_size, height),
#                 max(j - patch_size, 0):min(j + patch_size, width), :]
#         j_dark[i, j] = np.min(patch)
j_dark = np.array(Image.fromarray(np.min(im, axis=2)).filter(ImageFilter.MinFilter(filter_size)))

dark_channel_list = np.sort(np.reshape(j_dark, (-1, 1)))
dark_channel_max = dark_channel_list[int(len(dark_channel_list) * 0.001)]
A = np.max(im[j_dark > dark_channel_max, :], axis=0)
A[A > dark_channel_maximum] = dark_channel_maximum
tt = time.time()
t_ = np.zeros((height, width))
L = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        l = 0
        center = im[i, j]
        patch = im[max(i - patch_size, 0):min(i + patch_size, height),
                max(j - patch_size, 0):min(j + patch_size, width), :]
        patch = np.reshape(patch, (-1, 3))
        wk = np.shape(patch)[0]
        miu = np.mean(patch, axis=0)
        Ii = np.repeat(np.mat(center - miu), wk, axis=0)
        Ij = np.mat(patch) - np.repeat(np.mat(miu), wk, axis=0)
        # delta = np.var(patch)
        delta = np.mat(np.cov(patch.T))
        epsilon = 0.0001
        U3 = np.mat(np.eye(3))
        V = (delta + epsilon / wk * U3).I
        L[i, j] = 1 - 1 - 1 / wk * np.trace(Ii * V * Ij.T)

        # calculate the t~
        # patch = im[max(i - patch_size, 0):min(i + patch_size, height), max(j - patch_size, 0):min(j + patch_size, width), :]
        # t_[i, j] = 1 - 0.95 * np.min(patch/A)
        t_[i, j] = 1 - omega * np.min(im[i, j, :] / A)
print(time.time() - tt)
lambda_ = 0.001
U = np.ones((height, width))
m = min(width, height)
# U[0:m, 0:m] = np.eye(m)
# t = np.mat(L + lambda_ * U).I * lambda_ * t_  # something error
t = (L + lambda_ * U) ** (-1) * lambda_ * t_  # something error

# t = (L + lambda_ * U) ** (-1) * lambda_ * t_
# t = np.zeros((height, width))
# j_map = np.zeros((height, width, 3))


t = np.max((t, np.ones(np.shape(t)) * t0), axis=0)[:, :, np.newaxis]
t = np.tile(t, (1, 1, 3))
A = np.reshape(A, (1, 1, 3))
A = np.tile(A, (height, width, 1))
j_map = (im - A) / t + A
j_map[j_map > 255] = 255
j_map[j_map < 0] = 0
# for i in range(height):
#     for j in range(width):
#         j_map[i, j, :] = (im[i, j, :] - A) / max(t[i, j], t0) + A

cv2.imshow(u'raw image', np.asarray(im, dtype=np.uint8))
cv2.imshow(u'result', np.asarray(j_map, dtype=np.uint8))
cv2.imshow(u'dark channel', np.asarray(j_dark, dtype=np.uint8))
cv2.imshow(u'transmission', np.asarray(t * 255, dtype=np.uint8))

# cv2.imwrite(os.path.join(save_path, im_name), np.asarray(j_map, dtype=np.uint8))
cv2.waitKey(0)
