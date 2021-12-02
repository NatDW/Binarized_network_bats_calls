import os
import random
import sys
import time

import keras
import larq
import larq_zoo
import larq_compute_engine
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import network_builder


def main():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images / 255
    test_images = test_images / 255

    train_images = train_images.reshape((len(train_images), 28, 28, 1))
    test_images = test_images.reshape((len(test_images), 28, 28, 1))

    kwargs = dict(use_bias=False,
                  input_quantizer="ste_sign",
                  kernel_quantizer="ste_sign",
                  kernel_constraint="weight_clip")

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())
    model.add(larq.layers.QuantDense(784, use_bias=False, activation="linear", kernel_quantizer="ste_sign",
                                     kernel_constraint="weight_clip"))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(larq.layers.QuantDense(64, activation="sigmoid", **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(larq.layers.QuantDense(784, activation="sigmoid", **kwargs))

    model.add(tf.keras.layers.Reshape((28, 28, 1)))

    model.add(larq.layers.QuantConv2D(16, (3, 3), strides=2, activation="relu", padding="same"))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(larq.layers.QuantConv2D(8, (3, 3), strides=2, activation="relu", padding="same"))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(larq.layers.QuantConv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(larq.layers.QuantConv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(larq.layers.QuantConv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'))

    model.compile(optimizer="adam", loss="mse")

    """model.fit(train_images, train_images, epochs=1, shuffle=True, validation_data=(test_images, test_images))
    model.summary()

    res = model.predict(test_images)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_images[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(res[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()"""

    model = tf.keras.models.load_model("batML-main/batML_multiclass/data/models/23_04_21_14_51_55_detect_cnn2_hnm0_model")
    model.summary()

    network_builder.quantize_network(model, "ste_sign", "ste_sign", "weight_clip").summary()


if __name__ == '__main__':
    main()
