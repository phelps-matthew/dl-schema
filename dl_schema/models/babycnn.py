"""Two-layer CNN for MNIST"""
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
)

class BabyCNN(tf.keras.layers.Layer):
    def __init__(self, cfg=None):
        super(BabyCNN, self).__init__()
        self.cfg = cfg
        self.conv1 = Conv2D(filters=32, kernel_size=3)
        self.conv2 = Conv2D(filters=64, kernel_size=3)
        self.dropout1 = Dropout(self.cfg.dropout1)
        self.dropout2 = Dropout(self.cfg.dropout2)
        self.fc1 = Dense(self.cfg.fc_units)
        self.fc2 = Dense(self.cfg.out_features)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x):
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=1, padding="VALID")
        x = self.dropout1(x)
        x = Flatten()(x)
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = tf.nn.softmax(x)
        return output
