import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from PIL import ImageFilter, Image

im_name = '5.jpg'
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

tt = time.time()

a = ImageFilter.MinFilter(filter_size)
j_dark = np.array(Image.fromarray(np.min(im, axis=2)).filter(ImageFilter.MinFilter(filter_size)))

dark_channel_list = np.sort(np.reshape(j_dark, (-1, 1)))
dark_channel_max = dark_channel_list[int(len(dark_channel_list) * 0.001)]
A = np.max(im[j_dark > dark_channel_max, :], axis=0)
A[A > dark_channel_maximum] = dark_channel_maximum

A = np.reshape(A, (1, 1, 3))
A = np.tile(A, (height, width, 1))
t_ = 1 - omega * np.min(im / A, axis=2)
lambda_ = 0.001
U = np.ones((height, width))
m = min(width, height)

t = t_  #

t = np.max((t, np.ones(np.shape(t)) * t0), axis=0)[:, :, np.newaxis]
t = np.tile(t, (1, 1, 3))
# A = np.reshape(A, (1, 1, 3))
# A = np.tile(A, (height, width, 1))
j_map = (im - A) / t + A
j_map[j_map > 255] = 255
j_map[j_map < 0] = 0

print(time.time() - tt)

cv2.imshow(u'raw image', np.asarray(im, dtype=np.uint8))
cv2.imshow(u'result', np.asarray(j_map, dtype=np.uint8))
cv2.imshow(u'dark channel', np.asarray(j_dark, dtype=np.uint8))
cv2.imshow(u'transmission', np.asarray(t * 255, dtype=np.uint8))

# cv2.imwrite(os.path.join(save_path, im_name), np.asarray(j_map, dtype=np.uint8))
cv2.waitKey(0)
