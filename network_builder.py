import sys

import larq as lq
import larq.models
import tensorflow as tf
import numpy as np


def build_QuantDense(input_shape, nb_outputs, nb_layers, kernel_quantizer, kernel_constraint, input_quantizer):
    kwargs = dict(use_bias=False,
                  input_quantizer=input_quantizer,
                  kernel_quantizer=kernel_quantizer,
                  kernel_constraint=kernel_constraint)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=input_shape))

    model.add(lq.layers.QuantDense(32, use_bias=False, kernel_quantizer=kernel_quantizer,
                                   kernel_constraint=kernel_constraint))
    # model.add(tf.keras.layers.BatchNormalization(scale=False))
    for i in range(nb_layers):
        model.add(lq.layers.QuantDense(16, **kwargs))
        # model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(lq.layers.QuantDense(nb_outputs, activation="softmax", **kwargs))

    model.summary()
    larq.models.summary(model)

    return model


def build_Dense(input_shape, bias, nb_outputs, nb_layers, activation):
    kwargs = dict(use_bias=bias,
                  activation=activation)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=input_shape))

    model.add(tf.keras.layers.Dense(32, **kwargs))

    for i in range(nb_layers):
        model.add(tf.keras.layers.Dense(16, **kwargs))

    model.add(tf.keras.layers.Dense(nb_outputs, use_bias=bias, activation="softmax"))

    model.summary()

    return model


def build_cnn(input_shape, bias, nb_outputs, nb_layers_cnn, nb_layers_dense, kernel_quantizer, kernel_constraint,
              input_quantizer):
    kwargs = dict(use_bias=bias,
                  input_quantizer=input_quantizer,
                  kernel_quantizer=kernel_quantizer,
                  kernel_constraint=kernel_constraint)

    model = tf.keras.models.Sequential()

    for i in range(nb_layers_cnn):

        if i == 0:
            model.add(lq.layers.QuantConv2D(48, 3, input_shape=input_shape, padding="same", data_format="channels_last",
                                            use_bias=bias, kernel_quantizer=kernel_quantizer,
                                            kernel_constraint=kernel_constraint))
            model.add(
                tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=(2, 2), data_format="channels_last"))

        model.add(lq.layers.QuantConv2D(48, 3, input_shape=input_shape, padding="same", data_format="channels_last",
                                        **kwargs))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=(2, 2), data_format="channels_last"))

    for i in range(nb_layers_dense):
        model.add(tf.keras.layers.Dropout(rate=0.5))
        if i == 0:
            model.add(tf.keras.layers.Flatten())
        model.add(lq.layers.QuantDense(384, **kwargs))

    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(lq.layers.QuantDense(nb_outputs, **kwargs))

    model.summary()

    return model


def quantize_network(network, input_quantizer, kernel_quantizer, kernel_constraint):
    if input_quantizer == kernel_quantizer is None:
        return network

    model = tf.keras.models.Sequential()
    for i in range(len(network.layers)):
        config = network.layers[i].get_config()
        class_name = "Quant"+type(network.layers[i]).__name__
        layer_class = getattr(sys.modules['larq.layers'], class_name, None)
        if layer_class is not None:
            del config["kernel_constraint"]
            converted_layer = layer_class(input_quantizer=input_quantizer, kernel_quantizer=kernel_quantizer,
                                          kernel_constraint=kernel_constraint, **config)
            converted_layer.build(network.layers[i].input_shape)
            converted_layer.set_weights(network.layers[i].get_weights())
            model.add(converted_layer)
        else:
            model.add(type(network.layers[i])(**config))
    return model
