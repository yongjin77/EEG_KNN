import mne
import numpy as np
import pandas as pd
import os
from typing import Tuple, Optional


def readEEGFile(file_name: str, file_path: str = None) -> Optional[mne.io.Raw]:
    """
    Read an EEG file and extract the EEG Fpz-Cz channel.

    Args:
        file_name: Base name of the EEG file without extension
        file_path: Directory containing the EEG files (optional)

    Returns:
        mne.io.Raw: MNE Raw object containing the EEG Fpz-Cz channel, or None if error
    """
    try:
        # Construct file path
        full_path = os.path.join(file_path, file_name + "-PSG.edf") if file_path else file_name + "-PSG.edf"

        # Read the EDF file using MNE
        data = mne.io.read_raw_edf(full_path, preload=True)

        # Extract EEG channel
        EEG_Fpz_Cz = data.pick_channels(['EEG Fpz-Cz'])
        return EEG_Fpz_Cz
    except FileNotFoundError:
        print(f"File not found: {full_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {full_path}")
        print(f"Error message: {str(e)}")
        return None


def readHypnFile(file_name: str, file_path: str = None) -> Optional[pd.DataFrame]:
    """
    Read a hypnogram file and convert annotations to a DataFrame.

    Args:
        file_name: Base name of the hypnogram file without extension
        file_path: Directory containing the hypnogram files (optional)

    Returns:
        pd.DataFrame: DataFrame containing annotations, or None if error
    """
    try:
        # Construct file path
        full_path = os.path.join(file_path, file_name + "-Hypnogram.edf") if file_path else file_name + "-Hypnogram.edf"

        # Read annotations
        annotations = mne.read_annotations(full_path)

        # Convert to DataFrame
        annotations_df = pd.DataFrame({
            'Annotation': annotations.description,
            'Onset': annotations.onset,
            'Duration': annotations.duration
        })
        return annotations_df
    except FileNotFoundError:
        print(f"File not found: {full_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {full_path}")
        print(f"Error message: {str(e)}")
        return None
