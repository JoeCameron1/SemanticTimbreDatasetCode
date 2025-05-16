# ---------------------------------------------------
# PYTHON SCRIPT FOR TRAINING THE TIMBRE GENERATION VAE MODEL
# SCRIPT NAME = train_timbre_generation_vae.py
# USAGE = train_timbre_generation_vae.py

# AUTHOR = Joseph M. Cameron

# ---------------------------------------------------
# IMPORT STATEMENTS

import os
import numpy as np
from timbre_generation_vae import VAE

# ---------------------------------------------------

# GLOBAL TRAINING VARIABLES
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 300

# Function for loading all spectrograms
def load_spectrograms(spectrograms_directory):
    spectrogram_data = []
    for root, _, filenames in os.walk(spectrograms_directory):
        for filename in filenames:
            spectrogram_path = os.path.join(root, filename)
            spectrogram = np.load(spectrogram_path, allow_pickle=True)
            spectrogram_data.append(spectrogram)
    spectrogram_data = np.array(spectrogram_data)
    spectrogram_data = spectrogram_data[..., np.newaxis] # Ensure Correct Shape
    return spectrogram_data

# ---------------------------------------------------
# MAIN SCRIPT

if __name__ == "__main__":

    # Load All Training Spectrograms
    x_train = load_spectrograms("spectrograms/")

    # Initialise Timbre Generation VAE
    timbre_generation_vae = VAE(
        input_shape=(512, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dimension=128
    )

    # Print A Summary of the Timbre Generation VAE Model Architecture
    timbre_generation_vae.summary()

    # Compile the Timbre Generation VAE Model
    timbre_generation_vae.compile(LEARNING_RATE)

    # Train the Timbre Generation VAE Model
    timbre_generation_vae.train(x_train, BATCH_SIZE, EPOCHS)

    # Save the Timbre Generation VAE Model
    timbre_generation_vae.save("model")

# ---------------------------------------------------
