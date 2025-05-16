# ---------------------------------------------------
# PYTHON SCRIPT FOR GENERATING INTERPOLATED SOUNDS FROM THE TIMBRE GENERATION VAE MODEL
# SCRIPT NAME = generate_timbre_interpolation.py
# USAGE = generate_timbre_interpolation.py [START-TIMBRE-DESCRIPTOR] [START-TIMBRE-MAGNITUDE] [START-PITCH] [END-TIMBRE-DESCRIPTOR] [END-TIMBRE-MAGNITUDE] [END-PITCH] [INTERPOLATION-STEPS]

# AUTHOR = Joseph M. Cameron

# ----------------------------------------------------------------
# IMPORT STATEMENTS

import sys
import os
import pickle
import numpy as np
import librosa
import soundfile as sf
from timbre_generation_vae import VAE

# ----------------------------------------------------------------
# HANDLE ARGUMENTS

# Define the allowed values for arguments 1, 2, 3, 4, 5, 6, and 7
timbre_descriptor_allowed_values = ["Bright", "Crunch", "Crush", "Dark", "Dirt", "Fat", "Fluttery", "Fuzz", "Jittery", "Punch", "Resonant", "Sharp", "Shimmery", "Smooth", "Soft", "Stuttering", "Thin", "Tight", "WahWah"]
magnitude_allowed_values = [str(x) for x in range(0, 101, 25)]
pitch_allowed_values = ['E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6']
interpolation_allowed_values = [str(y) for y in range(3, 11)]

# Check if enough arguments are passed
if (len(sys.argv) < 8):
    print("Usage: python script.py [1st_Timbre_Descriptor] [1st_Timbre_Descriptor_Magnitude] [1st_Pitch] [2nd_Timbre_Descriptor] [2nd_Timbre_Descriptor_Magnitude] [2nd_Pitch] [No._Of_Interpolation_Steps]")
    print("       Please provide the correct number of arguments (7 arguments)")
    sys.exit(1)  # Exit the script if an invalid number of arguments are provided

# Check argument 1 (1st Timbre Descriptor)
if (sys.argv[1] not in timbre_descriptor_allowed_values):
    print("Usage: python script.py [1st_Timbre_Descriptor] [1st_Timbre_Descriptor_Magnitude] [1st_Pitch] [2nd_Timbre_Descriptor] [2nd_Timbre_Descriptor_Magnitude] [2nd_Pitch] [No._Of_Interpolation_Steps]")
    print(f"       The [1st_Timbre_Descriptor] argument must be one of: {timbre_descriptor_allowed_values}")
    sys.exit(1)  # Exit the script if invalid timbre descriptor value is provided

# Check argument 2 (1st Timbre Descriptor Magnitude)
if (sys.argv[2] not in magnitude_allowed_values):
    print("Usage: python script.py [1st_Timbre_Descriptor] [1st_Timbre_Descriptor_Magnitude] [1st_Pitch] [2nd_Timbre_Descriptor] [2nd_Timbre_Descriptor_Magnitude] [2nd_Pitch] [No._Of_Interpolation_Steps]")
    print(f"       The [1st_Timbre_Descriptor_Magnitude] argument must be one of: {magnitude_allowed_values}")
    sys.exit(1)  # Exit the script if invalid timbre magnitude value is provided

# Check argument 3 (1st Pitch)
if (sys.argv[3] not in pitch_allowed_values):
    print("Usage: python script.py [1st_Timbre_Descriptor] [1st_Timbre_Descriptor_Magnitude] [1st_Pitch] [2nd_Timbre_Descriptor] [2nd_Timbre_Descriptor_Magnitude] [2nd_Pitch] [No._Of_Interpolation_Steps]")
    print(f"       The [1st_Pitch] argument must be one of: {pitch_allowed_values}")
    sys.exit(1)  # Exit the script if invalid pitch is provided

# Check argument 4 (2nd Timbre Descriptor)
if (sys.argv[4] not in timbre_descriptor_allowed_values):
    print("Usage: python script.py [1st_Timbre_Descriptor] [1st_Timbre_Descriptor_Magnitude] [1st_Pitch] [2nd_Timbre_Descriptor] [2nd_Timbre_Descriptor_Magnitude] [2nd_Pitch] [No._Of_Interpolation_Steps]")
    print(f"       The [2nd_Timbre_Descriptor] argument must be one of: {timbre_descriptor_allowed_values}")
    sys.exit(1)  # Exit the script if invalid timbre descriptor value is provided

# Check argument 5 (2nd Timbre Descriptor Magnitude)
if (sys.argv[5] not in magnitude_allowed_values):
    print("Usage: python script.py [1st_Timbre_Descriptor] [1st_Timbre_Descriptor_Magnitude] [1st_Pitch] [2nd_Timbre_Descriptor] [2nd_Timbre_Descriptor_Magnitude] [2nd_Pitch] [No._Of_Interpolation_Steps]")
    print(f"       The [2nd_Timbre_Descriptor_Magnitude] argument must be one of: {magnitude_allowed_values}")
    sys.exit(1)  # Exit the script if invalid timbre magnitude value is provided

# Check argument 6 (2nd Pitch)
if (sys.argv[6] not in pitch_allowed_values):
    print("Usage: python script.py [1st_Timbre_Descriptor] [1st_Timbre_Descriptor_Magnitude] [1st_Pitch] [2nd_Timbre_Descriptor] [2nd_Timbre_Descriptor_Magnitude] [2nd_Pitch] [No._Of_Interpolation_Steps]")
    print(f"       The [2nd_Pitch] argument must be one of: {pitch_allowed_values}")
    sys.exit(1)  # Exit the script if invalid pitch is provided

# Check argument 7 (Number of Interpolation Steps)
if (sys.argv[7] not in interpolation_allowed_values):
    print("Usage: python script.py [1st_Timbre_Descriptor] [1st_Timbre_Descriptor_Magnitude] [1st_Pitch] [2nd_Timbre_Descriptor] [2nd_Timbre_Descriptor_Magnitude] [2nd_Pitch] [No._Of_Interpolation_Steps]")
    print(f"       The [Interpolation_Steps] argument must be one of: {interpolation_allowed_values}")
    sys.exit(1)  # Exit the script if invalid number of interpolation steps is provided

# Align Pitches and Filenames
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

# Assigning arguments to variables
first_timbre_descriptor = sys.argv[1]
first_timbre_magnitude = sys.argv[2]
first_pitch = pitch_map.get(sys.argv[3])
second_timbre_descriptor = sys.argv[4]
second_timbre_magnitude = sys.argv[5]
second_pitch = pitch_map.get(sys.argv[6])
interpolation_steps = int(sys.argv[7])

# If the user supplies a timbre magnitude of 0, then assign 'Fluttery' as the timbre descriptor which is where the 'Clean' notes reside
# This is because all notes with a timbre magnitude of 0 are 'Clean' guitar notes and all sound identical
if first_timbre_magnitude == "0":
    first_timbre_descriptor = "Fluttery"

if second_timbre_magnitude == "0":
    second_timbre_descriptor = "Fluttery"

# ----------------------------------------------------------------
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

# Function for selecting the two specified spectrograms for performing interpolation
def find_relevant_spectrograms(spectrograms, min_max_values, spectrogram_paths, filepath1, filepath2):
    first_index = spectrogram_paths.index(filepath1)
    second_index = spectrogram_paths.index(filepath2)
    relevant_spectrograms = spectrograms[[first_index, second_index]]
    first_relevant_min_max_value = min_max_values[filepath1]
    second_relevant_min_max_value = min_max_values[filepath2]
    relevant_min_max_values = np.array([first_relevant_min_max_value, second_relevant_min_max_value])
    print(filepath1, filepath2)
    print(relevant_min_max_values)
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

# Function for performing linear interpolation between two points in the latent space of a VAE
def perform_linear_interpolation(latent_representation1, latent_representation2, n_steps):
    ratios = np.linspace(0, 1, num=n_steps)
    interpolated_latent_representations = list()
    for ratio in ratios:
        v = (1.0 - ratio) * latent_representation1 + ratio * latent_representation2
        interpolated_latent_representations.append(v)
    return np.asarray(interpolated_latent_representations)

# Function for performing interpolation between the min-max values of two spectrograms
# Note:
# Performing interpolation between the two known min-max values from both specified spectrograms
# provides a sensible solution for assigning sensible min-max values to new spectrograms
def perform_min_max_values_interpolation(specified_min_max_value1, specified_min_max_value2, n_steps):
    min1 = specified_min_max_value1['min']
    min2 = specified_min_max_value2['min']
    interpolated_min_values = np.linspace(min1, min2, n_steps).tolist()
    max1 = specified_min_max_value1['max']
    max2 = specified_min_max_value2['max']
    interpolated_max_values = np.linspace(max1, max2, n_steps).tolist()
    interpolated_min_max_values = [{'min': min_val, 'max': max_val} for min_val, max_val in zip(interpolated_min_values, interpolated_max_values)]
    return interpolated_min_max_values

# Function for saving audio signals as 24-bit .wav files
def save_audio_signals(audio_signals, save_directory):
    for i, audio_signal in enumerate(audio_signals):
        sf.write(os.path.join(save_directory, str(i) + ".wav"), audio_signal, 22050, subtype='PCM_24')

# ----------------------------------------------------------------
# MAIN TIMBRE INTERPOLATION SCRIPT

if __name__ == "__main__":

    # Load the Timbre Generation VAE Model
    timbre_generation_vae = VAE.load("model")

    # Load All Spectrograms
    dataset_spectrograms, dataset_spectrograms_paths = load_spectrograms("spectrograms/")

    # Load All Min-Max Values
    with open("minmax/min_max_values.pkl", "rb") as f:
        min_max_values = pickle.load(f)

    # Obtain the file paths of the two spectrograms that represent the two user-specified groups of timbres, magnitudes, and pitches
    filepath1 = "spectrograms/" + first_timbre_descriptor + "X" + first_timbre_magnitude + "X" + first_pitch + ".wav.npy"
    filepath2 = "spectrograms/" + second_timbre_descriptor + "X" + second_timbre_magnitude + "X" + second_pitch + ".wav.npy"
    # Get the Specified Spectrograms and Their Min-Max Values for Interpolation
    specified_spectrograms, specified_min_max_values = find_relevant_spectrograms(dataset_spectrograms, min_max_values, dataset_spectrograms_paths, filepath1, filepath2)

    # Obtain the two latent representations that represent both spectrograms in the VAE's latent space
    generated_spectrograms, latent_representations = timbre_generation_vae.reconstruct(specified_spectrograms)

    # Perform linear interpolation between these two latent points
    timbre_interpolations = perform_linear_interpolation(latent_representations[0], latent_representations[-1], interpolation_steps)
    # Also perform interpolation for the specified spectrograms' min-max values
    interpolated_min_max_values = perform_min_max_values_interpolation(specified_min_max_values[0], specified_min_max_values[1], interpolation_steps)

    # Provide the interpolated latent vectors to the VAE's decoder to generate interpolated spectrograms
    generated_interpolation_spectrograms = timbre_generation_vae.decoder.predict(timbre_interpolations)

    # Generate new interpolated audio files from the interpolated spectrograms
    interpolated_audio_signals = convert_spectrograms_to_audio(generated_interpolation_spectrograms, interpolated_min_max_values)

    # Generate audio files from the original spectrograms to provide a ground truth comparison to the generated sounds
    original_audio_signals = convert_spectrograms_to_audio(specified_spectrograms, specified_min_max_values)

    # Save the VAE-Generated Interpolated Audio Signals
    save_audio_signals(interpolated_audio_signals, "interpolated_samples/generated/")
    
    # Save the Original Audio Signals (For Experimental Comparison)
    save_audio_signals(original_audio_signals, "interpolated_samples/original/")

# ----------------------------------------------------------------
