# SemanticTimbreDatasetCode README

This folder contains the code for the Semantic Timbre Generation System authored by [**Joseph M. Cameron**](https://josephcameron.info).

***

## [process_audio_into_spectrograms.py](Semantic_Timbre_Generation_System/process_audio_into_spectrograms.py)

* *DESCRIPTION*: This script generates the necessary spectrograms needed to train the timbre generation VAE model.

* *PREREQUISITES*:
  - Firstly, it expects to find a directory called 'audio' in its current directory, and this folder must contain the audio files from the SemanticTimbreDataset in the following structure:
    - `audio/[TIMBRE-GROUP]/[TIMBRE-DESCRIPTOR]/[TIMBRE-MAGNITUDE]/[AUDIO-FILE]`
    - `(e.g. audio/DistortionFX/Crunch/100/1-0.wav)`
  - The structure must match this for the script to correctly locate every audio file.
  - The full Semantic Timbre Dataset audio folder (all 275,310 audio files) can be downloaded and viewed [here on Hugging Face](https://huggingface.co/datasets/JoeCameron1/SemanticTimbreDataset), so this can be dragged into the [Semantic_Timbre_Generation_System](Semantic_Timbre_Generation_System) folder.
  - The selection of audio files I used during this project for training the timbre generation VAE and generating sounds can be downloaded online at: [https://zenodo.org/records/11398253](https://zenodo.org/records/11398253).
  - To use this data with the timbre generation system, just drag the downloaded folder to the '[Semantic_Timbre_Generation_System](Semantic_Timbre_Generation_System)' directory, and then rename this downloaded folder to 'audio'.

  - **There must also be existing empty directories in script's folder called 'spectrograms' and 'minmax'.**
  - **These folders are provided, with the ready-processed data I used during the project for convenience.**

* *USAGE*: `python3 process_audio_into_spectrograms.py`

  - No arguments are required.
  - The result of running the script will fill the 'spectrograms' folder with the necessary spectrograms needed for training the VAE, and generating sounds from a trained VAE.
  - The script also fills the 'minmax' folder with the min-max values for all audio files.

***

## [timbre_generation_vae.py](Semantic_Timbre_Generation_System/timbre_generation_vae.py)

* *DESCRIPTION*: This script is responsible for constructing a timbre generation VAE model.

* *PREREQUISITES*: None.

* *USAGE*: `python3 timbre_generation_vae.py`

  - Run the script to compile the timbre generation VAE model and see a summary of the model's architecture.

***

## [train_timbre_generation_vae.py](Semantic_Timbre_Generation_System/train_timbre_generation_vae.py)

* *DESCRIPTION*: This script is responsible for training the timbre generation VAE model.

* *PREREQUISITES*: The script expects to have spectrograms saved as NumPy arrays in the 'spectrograms' folder as training data.

* *USAGE*: `python3 train_timbre_generation_vae.py`

  - When training completes, the script produces a folder called 'model' which contains the fully trained semantic timbre generation VAE model.
  - The training process takes a long time (multiple days), so for convenience I have included the fully trained semantic timbre generation VAE model I used during this project in the '[model](Semantic_Timbre_Generation_System/model)' folder.

***

## [visualise_latent_space.py](Semantic_Timbre_Generation_System/visualise_latent_space.py)

* *DESCRIPTION*: This script is responsible for visualising the latent space of the fully trained semantic timbre generation VAE model, and how it organises latent representations.

* *PREREQUISITES*:
  - The script expects to have spectrograms saved as NumPy arrays in the 'spectrograms' folder as training data.
  - The script also expects to have a fully trained timbre generation VAE model in the 'model' folder.

* *USAGE*: `python3 visualise_latent_space.py [LABEL]`

  - The `[LABEL]` specifies what feature you would like to highlight in the latent representations with colour the latent representations in the visualisation.
  - It can equal `[-timbre/-magnitude/-pitch]`.
  - It's also an optional argument, so if left blank the default behaviour is to label based on all three features.
  - The script produces a plot of the latent space in the same directory as the script.

***

## [generate_single_sample.py](Semantic_Timbre_Generation_System/generate_single_sample.py)

* *DESCRIPTION*: This script is responsible for generating a single monophonic guitar note sound from the fully trained semantic timbre generation VAE model.

* *PREREQUISITES*:
  - The script expects to have spectrograms saved as NumPy arrays in the 'spectrograms' folder as training data.
  - The script expects to have the min-max values for all spectrograms in the 'minmax' folder.
  - The script also expects to have a fully trained timbre generation VAE model in the 'model' folder.
  - The script expects to have the 'singlesample/generated' folder for saving the generated sound.
  - The script expects to have the 'singlesample/original' folder for saving the orginal corresponding sound from the Semantic Timbre Dataset.

* *USAGE*: `python3 generate_single_sample.py [TIMBRE-DESCRIPTOR] [TIMBRE-MAGNITUDE] [PITCH]`

  - The `[TIMBRE-DESCRIPTOR]` argument specifies what timbre descriptor you would like to describe the generated sound. It can be one of: `["Bright", "Crunch", "Crush", "Dark", "Dirt", "Fat", "Fluttery", "Fuzz", "Jittery", "Punch", "Resonant", "Sharp", "Shimmery", "Smooth", "Soft", "Stuttering", "Thin", "Tight", "WahWah"]`

  - The `[TIMBRE-MAGNITUDE]` argument specifies what timbre magnitude you like the generated sound to have. It can be one of: `[0, 25, 50, 75, 100]`

  - The `[PITCH]` argument specifies what pitch the generated note should be in. It can be one of: `['E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6']`

* The script produces the VAE-generated sound in an existing 'singlesample/generated' folder. It also produces the original sound in an existing 'singlesample/original' folder for experimental comparison to the VAE-generated sound.

* *NOTE*: Every sound that was generated from the semantic timbre generation VAE during its evaluation can be downloaded online at: [https://zenodo.org/records/11398170](https://zenodo.org/records/11398170).

***

## [generate_timbre_interpolation.py](Semantic_Timbre_Generation_System/generate_timbre_interpolation.py)

* *DESCRIPTION*: This script is responsible for generating interpolated single monophonic guitar notes between specified start and target sounds from the fully trained semantic timbre generation VAE model.

* *PREREQUISITES*:
  - The script expects to have spectrograms saved as NumPy arrays in the 'spectrograms' folder as training data.
  - The script expects to have the min-max values for all spectrograms in the 'minmax' folder.
  - The script also expects to have a fully trained timbre generation VAE model in the 'model' folder.
  - The script expects to have the 'interpolated_samples/generated' folder for saving the generated interpolated sounds.
  - The script expects to have the 'interpolated_samples/original' folder for saving the orginal corresponding original two end point sounds from the Semantic Timbre Dataset.

* *USAGE*: `python3 generate_timbre_interpolation.py [START-TIMBRE-DESCRIPTOR] [START-TIMBRE-MAGNITUDE] [START-PITCH] [END-TIMBRE-DESCRIPTOR] [END-TIMBRE-MAGNITUDE] [END-PITCH] [INTERPOLATION-STEPS]`

   - The `[START-TIMBRE-DESCRIPTOR]` argument specifies what timbre descriptor you would like to describe the start sound of the interpolation.
   - The `[END-TIMBRE-DESCRIPTOR]` argument specifies what timbre descriptor you would like to describe the end sound of the interpolation.
   - Both can be one of: `["Bright", "Crunch", "Crush", "Dark", "Dirt", "Fat", "Fluttery", "Fuzz", "Jittery", "Punch", "Resonant", "Sharp", "Shimmery", "Smooth", "Soft", "Stuttering", "Thin", "Tight", "WahWah"]`

   - The `[START-TIMBRE-MAGNITUDE]` argument specifies what timbre magnitude you would like the interpolation's start sound to have.
   - The `[END-TIMBRE-MAGNITUDE]` argument specifies what timbre magnitude you would like the interpolation's end sound to have.
   - Both can be one of: `[0, 25, 50, 75, 100]`

   - The `[START-PITCH]` argument specifies what pitch the start sound of the interpolation should be in.
   - The `[END-PITCH]` argument specifies what pitch the end sound of the interpolation should be in.
   - Both can be one of: `['E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6']`

   - The `[INTERPOLATION-STEPS]` argument specified how many interpolated audio files should be produced (including the start and end sounds).
   - It can be one of: `[3, 4, 5, 6, 7, 8, 9, 10]`

   - The script produces the VAE-generated sounds from the timbre interpolation in an existing 'interpolated_samples/generated' folder.
   - It also produces the original start and end sounds that specified the timbre interpolation in an existing 'interpolated_samples/original' folder for experimental comparison to the VAE-generated sounds.

* *NOTE*: A collection of interpolated sounds from selected 5-point timbre interpolations generated from the timbre generation VAE can be downloaded online at: [https://zenodo.org/records/11398170](https://zenodo.org/records/11398170)

***

