import ReadFile
import numpy as np
from scipy.fft import fft
from scipy import signal

#use function in ReadFile to load data
def load_data(EEG_name,hypn_name):
    EEG = ReadFile.readEEGFile(EEG_name)
    annotations_df = ReadFile.readHypnFile(hypn_name)
    return EEG, annotations_df

# preprocess data, seperate the eeg signal into different epoch, 30 sec per epoch
def preprocess(EEG, epoch_duration = 30):
    sfreq = EEG.info['sfreq']
    samples_per_epoch = int(epoch_duration * sfreq)
    EEG_signal = EEG.get_data()
    n_epochs = len(EEG_signal[0]) // samples_per_epoch
    epochs = np.array_split(EEG_signal[0], n_epochs)
    return epochs

#dimenstion reduction use the energy for each single epoch
def extract_frequency_features(epoch, sfreq):
    window_signal = epoch * signal.windows.hann(len(epoch))
    fft_values = fft(window_signal)
    power = np.abs(fft_values) ** 2 / len(epoch)
    freqs = np.fft.fftfreq(len(epoch), 1 / sfreq)

    bands = {
        'delta': (1, 4),
        'theta': (5, 8),
        'alpha': (9, 12),
        'beta': (13, 30)
    }

    features = []
    for band, (low, high) in bands.items():
        band_power = np.sum(power[(freqs >= low) & (freqs < high)])
        features.append(band_power)

    return np.array(features)


def extract_features(epochs, sfreq):
    X = np.array([extract_frequency_features(epoch, sfreq) for epoch in epochs])
    return X


def process_annotations(annotations_df, n_epochs, epoch_duration):
    y = np.zeros(n_epochs, dtype=int)
    stage_mapping = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,  # follow AASM manual
        'Sleep stage R': 4,  # REM
    }

    for _, row in annotations_df.iterrows():
        if row['Annotation'] in stage_mapping:  # Only process known stages
            start_epoch = int(row['Onset'] / epoch_duration)
            end_epoch = int((row['Onset'] + row['Duration']) / epoch_duration)
            stage = stage_mapping[row['Annotation']]
            y[start_epoch:end_epoch] = stage

    # Remove epochs with label 0 that weren't explicitly set (i.e., unknown stages)
    valid_epochs = y != -1
    return y[valid_epochs]


def prepare_data_for_knn(eeg_file, anno_file, epoch_duration=30):

    #prepare and adjust the data
    EEG, annotations_df = load_data(eeg_file, anno_file)
    sfreq = EEG.info['sfreq']
    eeg_signal = EEG.get_data()

    epochs = preprocess(EEG, epoch_duration)

    #X is the signal, y is the transformed annotation label
    X = extract_features(epochs, sfreq)
    y = process_annotations(annotations_df, len(epochs), epoch_duration)

    return X, y


