# EEG Sleep Stage Classification with KNN

This project implements K-Nearest Neighbors (KNN) algorithms for classifying sleep stages using EEG (electroencephalogram) data. It includes basic KNN, an enhanced KNN with progress reporting, and an ensemble KNN approach.

## Project Components

- `KNN.py` - Basic K-Nearest Neighbors implementation
- `ProgressKNN.py` - Enhanced KNN with progress tracking
- `ensemble_knn.py` - Ensemble learning approach using KNN
- `ProcessEEG.py` - Functions for EEG signal processing and feature extraction
- `ReadFile.py` - Utilities for reading EEG and hypnogram files
- `Main.py` - Entry point for single case analysis
- `multi_case_knn_script.py` - Script for analyzing multiple EEG recordings

## Features

- Frequency-domain feature extraction from EEG signals
- Sleep stage classification according to AASM guidelines
- Progress tracking for long-running computations
- Ensemble learning using bootstrap sampling
- Data caching for improved performance

## Important Note

**The EEG dataset is not included in this repository due to its large size.**

To use this code, you need to download the Sleep-EDF Database Expanded from PhysioNet:
https://physionet.org/content/sleep-edfx/1.0.0/

After downloading, you'll need to update the file paths in the code to point to your local dataset.

## TODO

- Implement additional classification algorithms
- Create a combined ensemble model using multiple algorithm types
- Add cross-validation for hyperparameter tuning
- Optimize code for better performance with large datasets


