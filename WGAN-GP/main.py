import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from models import *
from utils import *
from train import *

generator = Generator()
discriminator = Discriminator(img_size)

(data, _), (data2,_) = tf.keras.datasets.mnist.load_data()

data = tf.concat([data, data2], axis=0) 
del data2
data = tf.cast(data, tf.float32)
data = tf.reshape(data, (-1, img_size, img_size, 1))
data = (data -127.5) / 127.5

tf_data = tf.data.Dataset.from_tensor_slices(data).shuffle(111)
tf_data = tf_data.batch(batch_size, drop_remainder=True)

train(generator, discriminator, tf_data, img_size, batch_size,
                    wass_loss, gradient_penalty, discriminator_optimizer,
                    generator_optimizer, epochs=25)

import matplotlib.pyplot as plt

noise = tf.random.normal((1, 100))
my_new_img = generator(noise, training=False).numpy()

plt.imshow(my_new_img.reshape(28,28,1))
plt.savefig("generated_img.png", cmap='gray')
plt.show()