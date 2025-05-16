# ---------------------------------------------------
# PYTHON SCRIPT FOR GENERATING A SOUND FROM THE TIMBRE GENERATION VAE MODEL
# SCRIPT NAME = generate_single_sample.py
# USAGE = generate_single_sample.py [TIMBRE-DESCRIPTOR] [TIMBRE-MAGNITUDE] [PITCH]

# AUTHOR = Joseph M. Cameron

# ---------------------------------------------------
# IMPORT STATEMENTS

import sys
import os
import pickle
import numpy as np
import librosa
import soundfile as sf
from timbre_generation_vae import VAE

# ---------------------------------------------------
# HANDLE ARGUMENTS

# Define the allowed values for arguments 1, 2, and 3
timbre_descriptor_allowed_values = ["Bright", "Crunch", "Crush", "Dark", "Dirt", "Fat", "Fluttery", "Fuzz", "Jittery", "Punch", "Resonant", "Sharp", "Shimmery", "Smooth", "Soft", "Stuttering", "Thin", "Tight", "WahWah"]
magnitude_allowed_values = [str(x) for x in range(0, 101, 25)]
pitch_allowed_values = ['E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6']

# Check if enough arguments are passed and validate arguments 2 and 3
if (len(sys.argv) < 4) or (sys.argv[2] not in magnitude_allowed_values) or (sys.argv[3] not in pitch_allowed_values):
    print("Usage: python script.py [Timbre_Descriptor] [Timbre_Descriptor_Magnitude] [Pitch]")
    print(f"       The [Timbre_Descriptor] argument must be one of: {timbre_descriptor_allowed_values}")
    print(f"       The [Timbre_Descriptor_Magnitude] argument must be one of: {magnitude_allowed_values}")
    print(f"       The [Pitch] argument must be one of: {pitch_allowed_values}")
    sys.exit(1)  # Exit the script if not enough arguments or incorrect values are provided

# Assigning arguments to variables
timbre_descriptor = sys.argv[1]
timbre_magnitude = sys.argv[2]
pitch_map = {
    'E4': '1-0',
    'F4': '1-1',
    'F#4': '1-2',
    'G4': '1-3',
    'G#4': '1-4',
    'A4': '1-5',
    'A#4': '1-6',
    'B4': '1-7',
    'C5': '1-8',
    'C#5': '1-9',
    'D5': '1-10',
    'D#5': '1-11',
    'E5': '1-12',
    'F5': '1-13',
    'F#5': '1-14',
    'G5': '1-15',
    'G#5': '1-16',
    'A5': '1-17',
    'A#5': '1-18',
    'B5': '1-19',
    'C6': '1-20',
    'C#6': '1-21',
    'D6': '1-22'
}
pitch = pitch_map.get(sys.argv[3])

# If the user supplies a timbre magnitude of 0, then assign 'Fluttery' as the timbre descriptor which is where the 'Clean' notes reside
# This is because all notes with a timbre magnitude of 0 are 'Clean' guitar notes and all sound identical
if timbre_magnitude == "0":
    timbre_descriptor = "Fluttery"

# ---------------------------------------------------
# HELPER FUNCTIONS

# Function for loading all spectrograms
def load_spectrograms(spectrograms_directory):
    spectrogram_data = []
    spectrogram_paths = []
    for root, _, filenames in os.walk(spectrograms_directory):
        for filename in filenames:
            spectrogram_path = os.path.join(root, filename)
            spectrogram = np.load(spectrogram_path)
            spectrogram_data.append(spectrogram)
            spectrogram_paths.append(spectrogram_path)
    spectrogram_data = np.array(spectrogram_data)
    spectrogram_data = spectrogram_data[..., np.newaxis] # Ensure Correct Shape
    return spectrogram_data, spectrogram_paths

# Function for selecting the specified spectrogram
def find_relevant_spectrogram(spectrograms, min_max_values, spectrogram_paths, relevant_spectrogram_path):
    index = spectrogram_paths.index(relevant_spectrogram_path)
    relevant_spectrogram = spectrograms[index]
    relevant_min_max_value = min_max_values[relevant_spectrogram_path]
    relevant_spectrograms = np.array([relevant_spectrogram])
    relevant_min_max_values = np.array([relevant_min_max_value])
    print("Relevant Spectrogram Path: ", relevant_spectrogram_path)
    print("Relevant Spectrogram's Min-Max Values: ", relevant_min_max_values)
    return relevant_spectrograms, relevant_min_max_values

# Function for denormalising a min-max normalised spectrogram
def denormalise(norm_array, original_min, original_max):
        array = (norm_array - 0) / (1 - 0)
        array = array * (original_max - original_min) + original_min
        return array

# Function for converting spectrograms into audio signals
def convert_spectrograms_to_audio(spectrograms, min_max_values):
    signals = []
    for spectrogram, min_max_value in zip(spectrograms, min_max_values):
        log_spectrogram = spectrogram[:, :, 0]
        # Denormalise
        denorm_log_spec = denormalise(log_spectrogram, min_max_value["min"], min_max_value["max"])
        spec = librosa.db_to_amplitude(denorm_log_spec)
        # Use Griffin-Lim algorithm for phase reconstruction
        signal = librosa.griffinlim(spec, hop_length=256)
        signals.append(signal)
    return signals

# Function for saving audio signals as 24-bit .wav files
def save_audio_signals(audio_signals, save_directory):
    for i, audio_signal in enumerate(audio_signals):
        sf.write(os.path.join(save_directory, str(i) + ".wav"), audio_signal, 22050, subtype='PCM_24')

# ---------------------------------------------------
# MAIN SCRIPT

if __name__ == "__main__":
    
    # Load the Timbre Generation VAE Model
    timbre_generation_vae = VAE.load("model")

    # Load All Spectrograms
    dataset_spectrograms, dataset_spectrograms_paths = load_spectrograms("spectrograms/")

    # Load All Min-Max Values
    with open("minmax/min_max_values.pkl", "rb") as f:
        min_max_values = pickle.load(f)

    # Get the Specified Spectrogram and Its Min-Max Values for Reconstruction
    path_from_specification = "spectrograms/" + timbre_descriptor + "X" + timbre_magnitude + "X" + pitch + ".wav.npy"
    specified_spectrograms, specified_min_max_values = find_relevant_spectrogram(dataset_spectrograms, min_max_values, dataset_spectrograms_paths, path_from_specification)

    # Generate Audio Signals From the Timbre Generation VAE
    generated_spectrograms, latent_representations = timbre_generation_vae.reconstruct(specified_spectrograms)
    generated_audio_signals = convert_spectrograms_to_audio(generated_spectrograms, specified_min_max_values)

    # Make Audio Signals From Original Spectrograms to Provide Comparison to the Generated Audio Signals
    original_audio_signals = convert_spectrograms_to_audio(specified_spectrograms, specified_min_max_values)

    # Save the VAE-Generated Audio Signals
    save_audio_signals(generated_audio_signals, "singlesample/generated/")

    # Save the Original Audio Signals (For Experimental Comparison)
    save_audio_signals(original_audio_signals, "singlesample/original/")

# ---------------------------------------------------
