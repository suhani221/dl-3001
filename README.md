# dl-3001

# EEG Spectrogram Generation and Denoising

This repository contains Python code for generating spectrograms from EEG (Electroencephalogram) data and implementing denoising techniques. Spectrograms provide visual representations of the frequency content of EEG signals over time, aiding in the analysis of brain activity patterns.

## Files

- **`generate_spectrograms.py`**: Python script containing the code for generating spectrograms.
- **`README.md`**: This file providing an overview of the repository and instructions for usage.

## Overview

The `generate_spectrograms.py` script processes EEG data stored in Parquet format, denoises the signals using wavelet transformation, and generates spectrograms using the Librosa library. Spectrograms are computed for different combinations of EEG channels and aggregated to create comprehensive visual representations of brain activity.

## How it Works

1. **Data Preparation**:
   - EEG data is loaded from Parquet files stored in a specified directory.
   - The data is preprocessed to ensure consistency and reliability for further analysis.

2. **Denoising**:
   - EEG signals undergo denoising using wavelet transformation.
   - This step enhances signal quality by reducing noise and artifacts, improving the accuracy of subsequent analyses.

3. **Spectrogram Generation**:
   - The denoised EEG signals are used to compute spectrograms using Librosa.
   - Mel-scaled spectrograms are calculated for each EEG channel combination, providing a frequency representation of brain activity over time.

4. **Visualization and Storage**:
   - Optionally, a subset of generated spectrograms along with their corresponding EEG signals can be displayed for visual inspection.
   - Generated spectrograms are saved as numpy arrays (.npy files) for future reference or further analysis.
