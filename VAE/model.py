import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

EPS = 1e-12


class Encoder(tf.keras.Model):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.latent_dim = self.config.latent_dim
        self.enc = Sequential([
            layers.InputLayer(input_shape=(32, 32, 1)),
            layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same'),
            layers.ReLU(),
            layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same'),
            layers.ReLU(),
            layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same'),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(1024),
            layers.ReLU(),
            layers.Dense(2 * self.latent_dim),
            ])

    def call(self, x):
        mean = self.enc(x)[:, :self.latent_dim]
        log_var = self.enc(x)[:, self.latent_dim:]
        return mean, log_var


class Decoder(tf.keras.Model):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.dec = Sequential([
            layers.InputLayer(input_shape=(self.config.latent_dim,)),
            layers.Dense(1024),
            layers.ReLU(),
            layers.Dense(4 * 4 * 64),
            layers.ReLU(),
            layers.Reshape((4, 4, 64)),
            layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same'),
            layers.ReLU(),
            layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same'),
            layers.ReLU(),
            layers.Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same'),
            ])

    def call(self, z):
        x_logit = self.dec(z)
        x = tf.nn.sigmoid(x_logit)
        return x_logit, x


class VAE(object):
    def __init__(self, config):
        self.config = config
        self.enc = Encoder(self.config)
        self.dec = Decoder(self.config)
        self.optim = tf.keras.optimizers.Adam(self.config.learning_rate, 0.5)
        self.global_step = tf.Variable(0, trainable=False)
        self.global_epoch = tf.Variable(0, trainable=False)

    def reparameterize_normal(self, mean, logvar):
        std = tf.math.exp(0.5 * logvar)
        eps = tf.random.normal(std.shape)
        return mean + std * eps

    def loss(self, x_batch):
        mean, logvar = self.enc(x_batch)
        z = self.reparameterize_normal(mean, logvar)
        x_logit, x_rec = self.dec(z)

        # Reconstruction loss term
        #rec_loss = -tf.reduce_mean(tf.reduce_sum(x_batch * tf.math.log(EPS + x_rec) + (1 - x_batch) * tf.math.log(EPS + 1 - x_rec), axis=[1, 2, 3]))
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x_batch)
        rec_loss = tf.reduce_mean(tf.reduce_sum(cross_ent, axis=[1, 2, 3]))

        # KL divergence loss term
        kl_normal = tf.reduce_mean(0.5 * tf.reduce_sum(tf.math.square(mean) + tf.math.exp(logvar) - logvar - 1, axis=[1]))

        tot_loss = rec_loss + kl_normal
        return tot_loss