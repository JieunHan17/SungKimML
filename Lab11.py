import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Make the image
image = np.array([[[[1], [2], [3]],
                   [[4], [5], [6]],
                   [[7], [8], [9]]]], dtype=np.float32)
print("image.shape", image.shape)
plt.imshow(image.reshape(3, 3), cmap="Greys")
plt.show()

# Initialize the weights
weight = np.array([[[[1.]], [[1.]]],
                   [[[1.]], [[1.]]]])
print("weight.shape", weight.shape)

# Convolve
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
print("conv2d.shape", conv2d.shape)

# Show the result
conv2d_img = np.swapaxes(conv2d, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2, 2))
    plt.subplot(1, 2, i + 1), plt.imshow(one_img.reshape(2, 2), cmap='gray')
    plt.show()
