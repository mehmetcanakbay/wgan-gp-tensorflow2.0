import tensorflow as tf
import numpy as np

batch_size = 64
penaltyLambda = 10
img_size=28

discriminator_optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1 = 0.5, beta_2=0.9)
generator_optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1 = 0.5, beta_2=0.9)

def gradient_penalty(discriminator, real_img, fake_img):
    epsilon = tf.random.uniform((batch_size, 1, 1, 1))
    interpolated_img = real_img*epsilon + fake_img*(1-epsilon)
    with tf.GradientTape() as gt:
        gt.watch(interpolated_img)
        mixed_score = discriminator(interpolated_img, training=False)
    gradients = gt.gradient(mixed_score, interpolated_img)
    gradients = tf.keras.backend.reshape(gradients, shape=(gradients.shape[0], -1))
    gradients = tf.norm(gradients, ord='euclidean', axis=1)
    gradient_pen = tf.reduce_mean((gradients-1) ** 2)
    return gradient_pen

def wass_loss(y_true, y_pred):
    loss = tf.keras.backend.mean(y_true - y_pred)
    return loss


""" def wass_loss_cr(discriminator, generator, real_imgs, noise):
    gen_img = generator(noise, training=False)
    print("HELLO")
    def loss(y_true, y_pred):
        penalty = penaltyLambda*gradient_penalty(discriminator, real_imgs, gen_img)
        print("did this work at all")
        return (-tf.keras.backend.mean(y_true*y_pred)) + penalty
    return loss """


