import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class Generator(tf.keras.Model):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.gen = Sequential([
            layers.InputLayer(input_shape=(self.config.latent_dim,)),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(8 * 8 * 128),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((8, 8, 128)),
            layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation='sigmoid'),
            ])

    def call(self, z):
        x = self.gen(z)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.dis = Sequential([
            layers.InputLayer(input_shape=(32, 32, 1)),
            layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(),
            layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Flatten(),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(1),
            ])

    def call(self, x):
        z = self.dis(x)
        return z


class DCGAN(object):
    def __init__(self, config):
        self.config = config
        self.dis = Discriminator(self.config)
        self.gen = Generator(self.config)
        self.d_optim = tf.keras.optimizers.Adam(self.config.d_lr, 0.5)
        self.g_optim = tf.keras.optimizers.Adam(self.config.g_lr, 0.5)
        self.global_step = tf.Variable(0, trainable=False)
        self.global_epoch = tf.Variable(0, trainable=False)
        
    def loss(self, x_batch):
        z = tf.random.normal([self.config.batch_size, self.config.latent_dim])
        g_fake = self.gen(z, training=True)
        d_fake = self.dis(g_fake, training=True)
        d_real = self.dis(x_batch, training=True)
        
        BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        d_loss_real = BCE(y_true=tf.ones_like(d_real), y_pred=d_real)
        d_loss_fake = BCE(y_true=tf.zeros_like(d_fake), y_pred=d_fake)
        d_loss = d_loss_real + d_loss_fake
        g_loss = BCE(y_true=tf.ones_like(d_fake), y_pred=d_fake)
        return d_loss, g_loss