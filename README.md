# dl-3001

# EEG Spectrogram Conversion

 **`parquet-npy.ipynb`**:This repository contains Python code for converting spectrograms from Parquet format to NumPy arrays (.npy). Converting spectrograms to NumPy arrays enables faster loading and processing, significantly reducing reading time.


## Overview

The `parquet-npy.ipynb` script converts spectrograms stored in Parquet format to NumPy arrays. This conversion facilitates faster loading and processing of spectrogram data, enhancing the efficiency of subsequent analyses.

## How it Works

1. **Spectrogram Retrieval**:
   - Spectrograms are loaded from Parquet files stored in the specified directory.

2. **Conversion**:
   - The loaded spectrograms are converted to NumPy arrays using the `np.load()` function.
   - Optionally, the script can read Parquet files directly and convert them to NumPy arrays if `READ_SPEC_FILES` is set to `True`.

3. **Output**:
   - Converted spectrograms are saved as NumPy arrays (.npy files) for future use.
   - The saved arrays can be loaded much faster compared to Parquet files, reducing reading time.


# EEG Spectrogram Generation and Denoising
 **`eeg-spectogram.ipynb`**:This file contains Python code for generating spectrograms from EEG (Electroencephalogram) data and implementing denoising techniques. Spectrograms provide visual representations of the frequency content of EEG signals over time, aiding in the analysis of brain activity patterns.


## Overview

The `eeg-spectogram.ipynb` script processes EEG data stored in Parquet format, denoises the signals using wavelet transformation, and generates spectrograms using the Librosa library. Spectrograms are computed for different combinations of EEG channels and aggregated to create comprehensive visual representations of brain activity.

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
