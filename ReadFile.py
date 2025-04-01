import mne
import numpy as np
import pandas as pd
import os

# Directory containing the sleep cassette files
filePath = #filepath

def readEEGFile(file_name):
    try:
        # Read the EDF file using MNE
        file_path = filePath + file_name + "-PSG.edf"
        data = mne.io.read_raw_edf(file_path)
        raw_data = data.get_data()
        info = data.info
        channels = data.ch_names
        EEG_Fpz_Cz = data.pick_channels(['EEG Fpz-Cz'])
        # signal = EEG_Fpz_Cz.get_data()
        # print(EEG_Fpz_Cz)
        # print(signal)
        # EEG_Fpz_Cz.plot(duration=30, n_channels=3, scalings='auto')
        return  EEG_Fpz_Cz
    except Exception as e:
        print(f"Error processing file: {file_path}")
        print(f"Error message: {str(e)}")

def readHypnFile(file_name):
    file_path = filePath + file_name + "-Hypnogram.edf"
    annotations = mne.read_annotations(file_path)
    annotations_df = pd.DataFrame({
        'Annotation': annotations.description,
        'Onset': annotations.onset,
        'Duration': annotations.duration
    })
    return annotations_df


