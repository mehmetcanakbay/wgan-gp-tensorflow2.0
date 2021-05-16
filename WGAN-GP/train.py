import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow.python.ops.functional_ops import Gradient
disc_loss_list = list()
gen_loss_list = list()

@tf.function
def train_step(generator, discriminator, real_imgs, img_size, batch_size, wass_loss, gradient_penalty, discriminator_optimizer, generator_optimizer):
    for _ in range(2):
        noise = tf.random.normal((batch_size, 100))
        gen_imgs = generator(noise, training=False)
        with tf.GradientTape() as tape:
            fake = generator(noise, training=False)
            noise = tf.random.normal((batch_size, 100))
            y_hat_real = discriminator(real_imgs)
            y_hat_fake = discriminator(fake)
            gp = gradient_penalty(discriminator, real_imgs, fake)
            loss_fn = (-(tf.reduce_mean(y_hat_real) - tf.reduce_mean(y_hat_fake)) + 10 * gp)

        grads = tape.gradient(loss_fn, discriminator.trainable_weights)
        discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

    noise = tf.random.normal((batch_size, 100))
    y_true = tf.ones((batch_size,))
    with tf.GradientTape() as tape:
        generated_batch = generator(noise)
        y_hat = discriminator(generated_batch)
        gen_loss = wass_loss(y_true, y_hat) 
    gradients = tape.gradient(gen_loss, generator.trainable_weights)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_weights))
    return loss_fn, gen_loss

def train(generator, discriminator, dataset, img_size, batch_size, wass_loss, wass_loss_cr, discriminator_optimizer, generator_optimizer, epochs=5):
    count = 0
    for epoch in range(epochs):
        print("Epoch {0}/{1}".format(epoch+1, epochs))
        start_time = time.time()
        for i, batch in enumerate(dataset):
            c_loss, g_loss = train_step(generator, discriminator, batch, img_size, batch_size, wass_loss, wass_loss_cr, discriminator_optimizer, generator_optimizer)
            disc_loss_list.append(c_loss)
            gen_loss_list.append(g_loss)
            if i % 200 == 0 and i != 0:
                print(f"Step {i}/{len(dataset)}")
                time_now = time.time()
                print(f"Elapsed time: {time_now - start_time}")
                print(f"D LOSS: {c_loss}, G LOSS: {g_loss}")
                start_time = time.time()
                noise = tf.random.normal((1, 100))
                my_new_img = generator(noise, training=False).numpy()
                plt.imshow(my_new_img.reshape(28,28,1), cmap='gray')
                plt.savefig(f"generated_img{count}_{i}.png")
        count += 1
        