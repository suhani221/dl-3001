# dl-3001

# EEG Spectrogram Conversion

 **`parquet-npy.ipynb`**:This file contains Python code for converting spectrograms from Parquet format to NumPy arrays (.npy). Converting spectrograms to NumPy arrays enables faster loading and processing, significantly reducing reading time.


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
     
# EEG Spectrogram Analysis

 **`models.ipynb`**:This file contains Python code for analyzing EEG spectrograms using Convolutional Neural Networks (CNNs) and Grad-CAM (Gradient-weighted Class Activation Mapping) visualization technique. The analysis involves training a CNN model on EEG spectrogram data and using Grad-CAM to interpret the model's predictions.


## Overview

The `models.ipynb` notebook includes code for training a CNN model on EEG spectrogram data, predicting EEG activity classes, and visualizing model predictions using Grad-CAM. Grad-CAM generates heatmap visualizations that highlight the regions of EEG spectrograms crucial for predicting specific brain activity classes.

## How it Works

1. **Data Preparation**:
   - EEG spectrogram data is loaded from Parquet files.
   - Spectrogram data is preprocessed and formatted for input to the CNN model.

2. **CNN Model Building**:
   - A CNN model architecture, based on the EfficientNetB0 model, is defined for predicting EEG activity classes.
   - The model is trained using EEG spectrogram data with corresponding activity class labels.

3. **Grad-CAM Visualization**:
   - Grad-CAM is applied to the trained CNN model to generate heatmap visualizations highlighting regions of interest in EEG spectrograms.
   - Heatmaps indicate areas crucial for predicting specific brain activity classes, providing insights into the model's decision-making process.

4. **Analysis and Interpretation**:
   - Model predictions and Grad-CAM visualizations are analyzed to understand how the model identifies EEG activity patterns.
   - Spectrogram images with overlaid Grad-CAM contours are displayed, facilitating interpretation of model predictions.

