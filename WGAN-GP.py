import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Reshape, Dense, LeakyReLU, Conv2D, Input
from tensorflow.keras.layers import Dropout, Flatten

# This line is for GPU.
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Our paramaters, penaltyLambda is from the improved GAN paper, and its used for gradient penalty.
batch_size = 64
penaltyLambda = 10
img_size=28

#Loading the data
(data, _), (_,_) = tf.keras.datasets.mnist.load_data()

#Normalizing the data
data = np.array(data).astype('float32')
data = data.reshape(-1, img_size, img_size, 1)
data = (data -127.5) / 127.5

tf_data = tf.data.Dataset.from_tensor_slices(data).shuffle(60000)
tf_data = tf_data.batch(batch_size, drop_remainder=True)

#This is a bit tricky to explain. Normally, its possible to do a custom training loop using for example, for batch in tf_data, and then train on this batch.
#But, if we do that, we cant update our batches in real time for the custom loss function. 
#To achieve that, we make a data_list, which puts all the batches inside a list. Then, we make a copy of our list and make it iterable.
#And then we will use the next() function to iterate over this list. That way, we update our batches in real time.
data_list = [x.numpy() for x in tf_data]
iterable_list = iter(data_list)

#GP from the paper.
def gradient_penalty(discriminator, real, fake):
    img_size, img_size, c = real.shape[1:]
    epsilon = np.random.uniform(size=(batch_size,1,1,1))
    interpolated_img = real*epsilon + fake*(1-epsilon)
    #Gradient Tape is used for calculating gradients when you are running on eager mode. (Which is enabled in TF2 by default)
    with tf.GradientTape() as gt:
        gt.watch(interpolated_img)
        mixed_score = discriminator(interpolated_img, training=False)
    gradients = gt.gradient(mixed_score, interpolated_img)
    gradients = tf.keras.backend.reshape(gradients, shape=(gradients.shape[0], -1))
    gradients = tf.norm(gradients, ord='euclidean', axis=1)
    gradient_penalty = tf.reduce_mean((gradients-1) ** 2)
    return gradient_penalty

#Wasserstein Loss 
def wass_loss(y_true, y_pred):
    loss = tf.keras.backend.mean(y_true*y_pred)
    return -loss

#This is the custom loss for our critic (discriminator). To use more variables in our loss function, we make another function surrounding it.
def wass_loss_cr(model, batch, noise):
    gen_img = generator(noise, training=False)
    def loss(y_true, y_pred):
        penalty = penaltyLambda*gradient_penalty(model,batch,gen_img)
        return (-tf.keras.backend.mean(y_true*y_pred)) + penalty
#         return penalty
    return loss

def make_generator():
    model = Sequential()
    
    model.add(Dense(7*7*128, input_dim=100))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))
    
    model.add(Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))

    
    model.add(Conv2D(1, kernel_size=4, padding='same',
                             activation='tanh'))
    return model

#Make the critic more powerful. Both training-wise, and structure-wise.
def make_critic():
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(img_size,img_size,1)))
    model.add(LeakyReLU(0.2))
    
    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
        
    model.add(Conv2D(64, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))

    return model

glosslist = list()
d_f_losslist = list()
d_t_losslist = list()
d_loss_list = list()
def train(gan, datas, batch_size, epochs=5):
    generator, discriminator = gan.layers
    for epoch in range(epochs):
        print("Epoch {0}/{1}".format(epoch+1, epochs))
        for batch in data_list:
            #This for loop is to train the critic even more. Critic is trained 5 times.
            for _ in range(5):
                noise = np.random.normal(0,1, size=(batch_size, 100))
                gen_img = generator(noise, training=False)
                discriminator.trainable=True
                batch = np.reshape(batch, ((batch_size, img_size,img_size,1)))
                dY_f = -np.ones(batch_size)
                dY_t = np.ones(batch_size)
                d_f_loss = discriminator.train_on_batch(gen_img, dY_f)
                d_t_loss = discriminator.train_on_batch(batch, dY_t)
            discriminator.trainable = False
            #######################

            noise = np.random.normal(0,1, size=(batch_size, 100))
            gY = np.ones(batch_size)
            gLOSS = gan.train_on_batch(noise, gY)

            #This is to plot the losses
            glosslist.append(gLOSS)
            d_f_losslist.append(d_f_loss)
            d_t_losslist.append(d_t_loss)
            d_loss_list.append(0.5*np.add(d_f_loss, d_t_loss))

#Activate the code below and the code for variable "noisey" if you want to save images for every epoch.
        # pred = generator.predict(noisey)
        # plt.imshow(pred.reshape(img_size,img_size), cmap='gray')
        # plt.savefig('Epoch{0}.jpg'.format(epoch+1))


critic = make_critic()
generator = make_generator()

#This is to test if our model is working properly
# noisey = np.random.normal(0,1,size=(1, 100))

#This is to feed our generator noise
noise = np.random.normal(0,1, size=(batch_size, 100))

#Compiling the critic model. Hyperparameters are from the Improved-GAN paper, which works perfectly for MNIST numbers dataset.
critic.compile(loss=wass_loss_cr(critic, np.array(next(iterable_list)), noise), optimizer=tf.keras.optimizers.Adam(lr=1e-5, beta_1 = 0.0, beta_2=0.9))
critic.trainable = False

gan = Sequential([generator, critic])
gan.compile(loss=wass_loss, optimizer=tf.keras.optimizers.Adam(lr=1e-5, beta_1 = 0.0, beta_2=0.9))

#Calling the train function here.
train(gan, data_list, batch_size, epochs=25)

#Activate code below if you want to see an output of the model.
# noise = np.random.normal(0,1,size=(1, 100))
# pred = generator.predict(noise)
# plt.imshow(pred.reshape(28,28), cmap='gray')
# plt.show()


plt.figure(figsize=(16,8))
plt.plot(glosslist, label='generator')
plt.plot(d_f_losslist, label='critic_fake')
plt.plot(d_t_losslist, label='critic_true')
plt.plot(d_loss_list, label='critic')
plt.legend()
plt.show()