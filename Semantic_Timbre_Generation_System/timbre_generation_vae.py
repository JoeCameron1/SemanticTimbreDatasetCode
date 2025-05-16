# ---------------------------------------------------
# PYTHON SCRIPT FOR CONSTRUCTING THE TIMBRE GENERATION VAE MODEL
# SCRIPT NAME = timbre_generation_vae.py
# USAGE = timbre_generation_vae.py

# AUTHOR = Joseph M. Cameron

# ---------------------------------------------------
# IMPORT STATEMENTS

import os
import pickle
import numpy as np

# Keras Imports
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import MeanSquaredError

# Eager Execution is Turned Off For Better Performace and Future Portability Outside Python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# ---------------------------------------------------
# HELPER FUNCTIONS

# Function for Calculating the Reconstruction Loss of a VAE
def calculate_reconstruction_loss(y_target, y_predicted):
    error = y_target - y_predicted
    reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
    return reconstruction_loss

# Function for Calculating the KL Loss of a VAE
# Returns another function for compatibility with Keras' metrics parameter in the 'compile' method
def calculate_kl_loss(model):
    def calculate_kullback_leibler(*args):
        kl_loss = -0.5 * K.sum(1 + model.log_variance - K.square(model.mu) - K.exp(model.log_variance), axis=1)
        return kl_loss
    return calculate_kullback_leibler

# ---------------------------------------------------

# VAE CLASS
# Class for Holding the Timbre Generation Variational Autoencoder Model Architecture
# and Accompanying Training/Saving/Loading Methods.
class VAE:

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dimension):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dimension = latent_space_dimension
        self.encoder = None
        self.decoder = None
        self.model = None
        self.model_input_layer = None
        self.pre_bottleneck_shape = None
        self.construct_timbre_generation_vae()

    def construct_timbre_generation_vae(self):
        self.construct_vae_encoder()
        self.construct_vae_decoder()
        model_input = self.model_input_layer
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="timbre_generation_vae")

    def construct_vae_encoder(self):
        encoder_input = Input(shape=self.input_shape, name="encoder_input")
        x = encoder_input
        for i in range(len(self.conv_filters)):
            x = Conv2D(filters=self.conv_filters[i], kernel_size=self.conv_kernels[i], strides=self.conv_strides[i], padding="same")(x)
            x = ReLU()(x)
            x = BatchNormalization()(x)
        bottleneck = self.construct_vae_bottleneck(x)
        self.model_input_layer = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def construct_vae_bottleneck(self, x):
        self.pre_bottleneck_shape = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.mu = Dense(self.latent_space_dimension, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dimension, name="log_variance")(x)
        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0., stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point
        x = Lambda(sample_point_from_normal_distribution, name="encoder_output")([self.mu, self.log_variance])
        return x
    
    def construct_vae_decoder(self):
        decoder_input = Input(shape=self.latent_space_dimension, name="decoder_input")
        x = Dense(np.prod(self.pre_bottleneck_shape))(decoder_input)
        x = Reshape(self.pre_bottleneck_shape)(x)
        for i in reversed(range(1, len(self.conv_filters))):
            x = Conv2DTranspose(filters=self.conv_filters[i], kernel_size=self.conv_kernels[i], strides=self.conv_strides[i], padding="same")(x)
            x = ReLU()(x)
            x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=1, kernel_size=self.conv_kernels[0], strides=self.conv_strides[0], padding="same")(x)
        decoder_output = Activation("sigmoid", name="decoder_output")(x)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def calculate_total_loss(self, y_target, y_predicted):
        reconstruction_loss = calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = calculate_kl_loss(self)()
        total_loss = 1000000 * reconstruction_loss + kl_loss
        return total_loss

    def compile(self, learning_rate=0.0001):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=self.calculate_total_loss, metrics=[calculate_reconstruction_loss, calculate_kl_loss(self)])

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, x_train, batch_size=batch_size, epochs=num_epochs, shuffle=True)

    def reconstruct(self, spectrograms):
        latent_representations = self.encoder.predict(spectrograms)
        reconstructed_spectrograms = self.decoder.predict(latent_representations)
        return reconstructed_spectrograms, latent_representations

    def save(self, save_folder="."):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # Save the Model's Parameters
        with open(os.path.join(save_folder, "parameters.pkl"), "wb") as f:
            pickle.dump([self.input_shape, self.conv_filters, self.conv_kernels, self.conv_strides, self.latent_space_dimension], f)
        # Save the Model's Weights
        self.model.save_weights(os.path.join(save_folder, "weights.h5"))

    @classmethod
    def load(cls, save_folder="."):
        # Load Model Parameters
        with open(os.path.join(save_folder, "parameters.pkl"), "rb") as f:
            parameters = pickle.load(f)
        timbre_generation_vae = VAE(*parameters)
        # Load Model Weights
        timbre_generation_vae.model.load_weights(os.path.join(save_folder, "weights.h5"))
        return timbre_generation_vae

# ---------------------------------------------------
# MAIN SCRIPT

# Run Script to See VAE Model Architecture Summaries
if __name__ == "__main__":
    timbre_generation_vae = VAE(
        input_shape=(512, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dimension=128
    )
    timbre_generation_vae.summary()

# ---------------------------------------------------
