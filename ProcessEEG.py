import ReadFile
import numpy as np
from scipy.fft import fft
from scipy import signal
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional


def load_data(EEG_name: str, hypn_name: str, file_path: str = None) -> Tuple[Any, pd.DataFrame]:
    """
    Load EEG data and annotations from files.

    Args:
        EEG_name: Base name of the EEG file
        hypn_name: Base name of the hypnogram file
        file_path: Directory containing the files (optional)

    Returns:
        Tuple[Any, pd.DataFrame]: EEG data and annotations DataFrame
    """
    EEG = ReadFile.readEEGFile(EEG_name, file_path)
    annotations_df = ReadFile.readHypnFile(hypn_name, file_path)
    return EEG, annotations_df


def preprocess(EEG: Any, epoch_duration: int = 30) -> np.ndarray:
    """
    Preprocess EEG data by dividing into epochs.

    Args:
        EEG: MNE Raw object containing EEG data
        epoch_duration: Duration of each epoch in seconds

    Returns:
        np.ndarray: Array of epochs
    """
    sfreq = EEG.info['sfreq']
    samples_per_epoch = int(epoch_duration * sfreq)
    EEG_signal = EEG.get_data()
    n_epochs = len(EEG_signal[0]) // samples_per_epoch

    # Reshape directly instead of using array_split for better performance
    epochs = np.reshape(EEG_signal[0][:n_epochs * samples_per_epoch],
                        (n_epochs, samples_per_epoch))
    return epochs


def extract_frequency_features(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Extract frequency band features from an EEG epoch.

    Args:
        epoch: Single EEG epoch
        sfreq: Sampling frequency

    Returns:
        np.ndarray: Array of features (delta, theta, alpha, beta powers)
    """
    # Apply Hann window to reduce spectral leakage
    window_signal = epoch * signal.windows.hann(len(epoch))

    # Compute FFT
    fft_values = fft(window_signal)
    power = np.abs(fft_values) ** 2 / len(epoch)
    freqs = np.fft.fftfreq(len(epoch), 1 / sfreq)

    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (5, 8),
        'alpha': (9, 12),
        'beta': (13, 30)
    }

    # Calculate power in each band using vectorized operations
    features = []
    for band_name, (low, high) in bands.items():
        # Create mask for the frequency band and sum power
        band_mask = (freqs >= low) & (freqs < high)
        band_power = np.sum(power[band_mask])
        features.append(band_power)

    return np.array(features)


def extract_features(epochs: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Extract features from all epochs.

    Args:
        epochs: Array of EEG epochs
        sfreq: Sampling frequency

    Returns:
        np.ndarray: Feature matrix of shape (n_epochs, n_features)
    """
    X = np.array([extract_frequency_features(epoch, sfreq) for epoch in epochs])
    return X


def process_annotations(annotations_df: pd.DataFrame, n_epochs: int,
                        epoch_duration: int) -> np.ndarray:
    """
    Process annotations to create labels for each epoch.

    Args:
        annotations_df: DataFrame containing sleep stage annotations
        n_epochs: Number of epochs
        epoch_duration: Duration of each epoch in seconds

    Returns:
        np.ndarray: Array of sleep stage labels
    """
    # Initialize labels with -1 (unknown)
    y = np.full(n_epochs, -1, dtype=int)

    # Define sleep stage mapping according to AASM manual
    stage_mapping = {
        'Sleep stage W': 0,  # Wake
        'Sleep stage 1': 1,  # N1
        'Sleep stage 2': 2,  # N2
        'Sleep stage 3': 3,  # N3
        'Sleep stage 4': 3,  # N3 (merged with stage 3 as per AASM)
        'Sleep stage R': 4,  # REM
    }

    # Map annotations to epochs
    for _, row in annotations_df.iterrows():
        if row['Annotation'] in stage_mapping:
            start_epoch = int(row['Onset'] / epoch_duration)
            end_epoch = int((row['Onset'] + row['Duration']) / epoch_duration)

            # Ensure indices are within bounds
            start_epoch = max(0, start_epoch)
            end_epoch = min(n_epochs, end_epoch)

            if start_epoch < end_epoch:  # Ensure at least one epoch
                stage = stage_mapping[row['Annotation']]
                y[start_epoch:end_epoch] = stage

    # Keep only known stages (not -1)
    valid_indices = y != -1
    return y[valid_indices]


def prepare_data_for_knn(eeg_file: str, anno_file: str,
                         epoch_duration: int = 30,
                         file_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare EEG data for KNN classification.

    Args:
        eeg_file: Base name of the EEG file
        anno_file: Base name of the hypnogram file
        epoch_duration: Duration of each epoch in seconds
        file_path: Directory containing the files (optional)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix and labels
    """
    # Load and preprocess data
    EEG, annotations_df = load_data(eeg_file, anno_file, file_path)
    sfreq = EEG.info['sfreq']

    # Extract epochs
    epochs = preprocess(EEG, epoch_duration)

    # Extract features and labels
    X = extract_features(epochs, sfreq)
    y = process_annotations(annotations_df, len(epochs), epoch_duration)

    # Ensure X and y have the same number of samples
    if len(X) > len(y):
        X = X[:len(y)]
    elif len(y) > len(X):
        y = y[:len(X)]

    return X, y


