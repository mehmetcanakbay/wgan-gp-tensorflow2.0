import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = layers.Dense(7*7*128, input_dim=(100,))
        self.reshape = layers.Reshape((7,7,128))
        self.deconv1 = layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')
        self.batchnorm1 = layers.BatchNormalization(momentum=0.8)
        self.deconv2 = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')
        self.batchnorm2 = layers.BatchNormalization(momentum=0.8)
        self.outconv = layers.Conv2D(1, kernel_size=4, padding='same', activation='tanh')
    def call(self, inputs):
        x = self.fc1(inputs)
        x = tf.nn.leaky_relu(x, 0.2)
        x = self.reshape(x)
        x = self.batchnorm1(tf.nn.leaky_relu(self.deconv1(x), 0.2))
        x = self.batchnorm2(tf.nn.leaky_relu(self.deconv2(x), 0.2))
        outputs = self.outconv(x)
        return outputs

class Discriminator(tf.keras.Model):
    def __init__(self, img_size):
        super().__init__()
        self.conv1 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(img_size,img_size,1))
        self.conv2 = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')
        self.conv3 = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1, activation='linear')
        
    def call(self, inputs):
        x = tf.nn.leaky_relu(self.conv1(inputs), 0.2)
        x = tf.nn.leaky_relu(self.conv2(x), 0.2)
        x = tf.nn.leaky_relu(self.conv3(x), 0.2)
        x = self.flatten(x)
        outputs = self.fc1(x)
        return outputs