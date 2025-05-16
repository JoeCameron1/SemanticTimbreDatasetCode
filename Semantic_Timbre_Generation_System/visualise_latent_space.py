# ---------------------------------------------------
# PYTHON SCRIPT VISUALISING THE LATENT SPACE OF THE TRAINED TIMBRE GENERATION VAE MODEL
# SCRIPT NAME = visualise_latent_space.py
# USAGE = visualise_latent_space.py [LABEL]

# AUTHOR = Joseph M. Cameron

# ----------------------------------------------------------------
# IMPORT STATEMENTS

import sys
import os
import numpy as np
from timbre_generation_vae import VAE
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# ----------------------------------------------------------------
# HANDLE ARGUMENTS

parser = argparse.ArgumentParser(description='Visualise the latent space of the timbre generation variational autoencoder.')
group = parser.add_mutually_exclusive_group()
group.add_argument('-timbre', '--timbre', action='store_true', help='Visualise the latent space and label samples based on their timbre descriptor.')
group.add_argument('-magnitude', '--magnitude', action='store_true', help='Visualise the latent space and label samples based on their timbre descriptor magnitude.')
group.add_argument('-pitch', '--pitch', action='store_true', help='Visualise the latent space and label samples based on their pitch.')
args = parser.parse_args()

# ----------------------------------------------------------------
# HELPER FUNCTIONS

# Function for loading all spectrograms
def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train, file_paths

# Function for extracting labels from a spectrogram's file name
def extract_labels(file_paths):
    labels = {'descriptor': [], 'magnitude': [], 'pitch': []}
    pattern = r'(\w+)X(\d+)X(\d+-\d+)\.wav\.npy'
    for path in file_paths:
        filename = os.path.basename(path)
        print("Filename:", filename)
        match = re.match(pattern, filename)
        print("Match:", match)
        if match:
            labels['descriptor'].append(match.group(1))
            labels['magnitude'].append(int(match.group(2)))
            labels['pitch'].append(match.group(3))
    return labels

# ----------------------------------------------------------------
# MAIN LATENT SPACE VISUALISATION SCRIPT

if __name__ == "__main__":
    # Load the trained VAE model
    vae = VAE.load("model")

    # Load all spectrograms and their file paths
    specs, file_paths = load_fsdd("spectrograms/")

    # Get the labels for the spectrograms
    labels = extract_labels(file_paths)

    # Get all the latent representations from the Timbre Generation VAE's Encoder
    latent_representations = vae.encoder.predict(specs)

    # Use t-SNE for dimensionality reduction to 2D
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(latent_representations)

    # Now put together the latent space plot depending on user-specified arguments
    plt.figure(figsize=(10, 10))

    if args.timbre:
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels['descriptor'], palette=sns.color_palette("hsv", len(set(labels['descriptor']))))
    elif args.magnitude:
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels['magnitude'], palette=sns.color_palette("hsv", len(set(labels['magnitude']))))
    elif args.pitch:
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels['pitch'], palette=sns.color_palette("hsv", len(set(labels['pitch']))))
    else:
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels['descriptor'], style=labels['pitch'], size=labels['magnitude'], sizes=(20, 200), palette=sns.color_palette("hsv", len(set(labels['descriptor']))))

    plt.title("t-SNE Visualisation of the VAE's Latent Space")
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    if args.timbre:
        plt.legend(title='Timbre Descriptor', bbox_to_anchor=(1.05, 1), loc='upper left')
    elif args.magnitude:
        plt.legend(title='Timbre Descriptor Magnitude', bbox_to_anchor=(1.05, 1), loc='upper left')
    elif args.pitch:
        plt.legend(title='Pitch', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(title='Descriptors & Pitches', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True)

    if args.timbre:
        plt.savefig('Visualise_VAE_Latent_Space_tSNE_TimbreLabel.png', bbox_inches='tight')
    elif args.magnitude:
        plt.savefig('Visualise_VAE_Latent_Space_tSNE_MagnitudeLabel.png', bbox_inches='tight')
    elif args.pitch:
        plt.savefig('Visualise_VAE_Latent_Space_tSNE_PitchLabel.png', bbox_inches='tight')
    else:
        plt.savefig('Visualise_VAE_Latent_Space_tSNE.png', bbox_inches='tight')

    plt.show()
    plt.close()

# ----------------------------------------------------------------
