# ---------------------------------------------------
# PYTHON SCRIPT FOR PROCESSING AUDIO FILES INTO SPECTROGRAMS
# FOR TRAINING THE TIMBRE GENERATION VAE MODEL
# SCRIPT NAME = process_audio_into_spectrograms.py
# USAGE = process_audio_into_spectrograms.py

# AUTHOR = Joseph M. Cameron

# ---------------------------------------------------
# IMPORT STATEMENTS

import os
import librosa
import numpy as np
import pickle

# ---------------------------------------------------
# AUDIO TO SPECTROGRAM PROCESSING CLASS

class AudioToSpectrogramProcessor:

    def __init__(self):
        self.min_max_values = {}

    def process_audio_file(self, audio_file_path):
        # Load the Audio Signal
        signal = librosa.load(audio_file_path, sr=22050, duration=0.74, mono=True)[0]
        # Pad the Audio Signal If It's Shorter Than Expected
        if len(signal) < int(22050 * 0.74):
            signal = np.pad(signal, (0, (int(22050 * 0.74) - len(signal))), mode="constant")
        # Get the Log-Spectrogram of the Audio Signal
        log_spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=1024, hop_length=256)[:-1]))
        # Normalise the Log-Spectrogram With Min-Max Normalisation
        normalised_log_spectrogram = (log_spectrogram - log_spectrogram.min()) / (log_spectrogram.max() - log_spectrogram.min())
        # Save the Pre-Processed Spectrogram With A Meaningful Filename
        note_file_name = os.path.split(audio_file_path)[1]
        struct_dirs = os.path.normpath(os.path.split(audio_file_path)[0])
        mag_dir = os.path.basename(struct_dirs)
        descrip_dir = os.path.basename(os.path.normpath(os.path.split(struct_dirs)[0]))
        file_name = descrip_dir + 'X' + mag_dir + 'X' + note_file_name
        save_path = os.path.join("spectrograms/", file_name + ".npy")
        np.save(save_path, normalised_log_spectrogram)
        # Save the Spectrogram's Original Min-Max Values Before Min-Max Normalisation For Retrieval in the Generation Phase
        self.min_max_values[save_path] = {"min": log_spectrogram.min(), "max": log_spectrogram.max()}

    def process_all_audio_files(self, audio_directory):
        for root, _, files in os.walk(audio_directory):
            for file in files:
                audio_file_path = os.path.join(root, file)
                self.process_audio_file(audio_file_path)
                print(f"Pre-Processed Audio in File: {audio_file_path}")
        # Save All the MinMax Values For All Preprocessed Audio Files
        with open(os.path.join("minmax/", "min_max_values.pkl"), "wb") as f:
            pickle.dump(self.min_max_values, f)

# ---------------------------------------------------
# MAIN SCRIPT

if __name__ == "__main__":
    audio_to_spectrogram_processor = AudioToSpectrogramProcessor()
    audio_to_spectrogram_processor.process_all_audio_files("audio/")

# ---------------------------------------------------
