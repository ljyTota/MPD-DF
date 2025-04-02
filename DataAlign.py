# -*- coding: utf-8 -*-

# This function preprocesses raw EEG and PSG data along with their annotation files.
# It extracts ECG, EOG, and respiratory effort (RES) signals from the PSG data and aligns EEG, ECG, EOG, and RES signals
# with their corresponding labels on a per-second basis. The processed data is saved in .mat format,
# with signal data stored as time length (in seconds) × number of channels × sampling rate and annotation data stored as 1 × time length (in seconds).

# After downloading the publicly available raw data, the basic path is assigned to the variable path,
# and the output directory is assigned to prepath. The processed data are subsequently organized and saved into 50 individual folders,
# each labeled with a volunteer ID (e.g., 01, 02, …, 50). Each folder contains the corresponding data for EEG, ECG, EOG, and RES.

# Recommend Python 3.8 or later.

# Authors: Jiayi Li, Jinbu Tang, Ligang Zhou
# Date: 2025.2.24

import logging
import numpy as np
import os
import mne
from mne.io import read_raw_edf
import re
from scipy.io import savemat

def load_annotation_one_subj(file1):  # file1 refers to annotation_path
    # Read all lines from the file
    with open(file1, 'r') as file:
        raw_list = file.readlines()

    # Initialize the lists to store timestamps and annotations
    timestamp = []
    annotation_list = []

    # Define label mapping
    label_mapping = {
        "Signal Abnormality": 8,
        "Severe Artifacts": 9
    }

    # Iterate through each line in the raw list
    for line in raw_list:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:  # Skip empty lines
            # Split the line by commas
            parts = line.split(',')

            # Extract the time portion (HH:MM:SS) and split it by ':'
            h_m_s = parts[0].split(':')  # Example: '13:53:18' -> ['13', '53', '18']
            # print(h_m_s)

            # Calculate the relative time with respect to 00:00:00 for easier computation.
            time_stamp = int(h_m_s[0]) * 3600 + int(h_m_s[1]) * 60 + int(h_m_s[2])
            timestamp.append(time_stamp)

            # Extract the annotation (last part of the line)
            label = parts[-1].strip()  # Remove any extra whitespace

            # Convert label based on mapping
            if label in label_mapping:
                annotation_list.append(label_mapping[label])
            else:
                try:
                    annotation_list.append(int(label))  # Convert to integer if possible
                except ValueError:
                    # Handle unexpected labels (optional)
                    annotation_list.append(-1)  # Use -1 or another placeholder for unknown labels

    # Return the timestamp and annotation list
    return timestamp, annotation_list

def load_eeg_psg_one_subj(eeg_files, psg_files, timestamp, annotation_list):

    # Read the raw EEG data from the specified file, preload the data into memory.
    eeg_file = read_raw_edf(eeg_files, preload=True)

    # Read the raw PSG data, excluding specific channels to isolate different signal types
    eog_file = read_raw_edf(psg_files, exclude=['ECG', 'Thor'], preload=True)
    ecg_file = read_raw_edf(psg_files, exclude=['E1-M2', 'E2-M2', 'Thor'], preload=True)
    thor_file = read_raw_edf(psg_files, exclude=['E1-M2', 'E2-M2', 'ECG'], preload=True)

    # print('ch_names',eeg_file.ch_names)

    # Extract start time for each signal from the metadata
    eeg_start_time = eeg_file.info['meas_date']
    ecg_start_time = ecg_file.info['meas_date']
    eog_start_time = eog_file.info['meas_date']
    thor_start_time = thor_file.info['meas_date']

    # Convert start times to seconds from the start of the day
    eeg_start_time = eeg_start_time.hour * 3600 + eeg_start_time.minute * 60 + eeg_start_time.second
    ecg_start_time = ecg_start_time.hour * 3600 + ecg_start_time.minute * 60 + ecg_start_time.second
    eog_start_time = eog_start_time.hour * 3600 + eog_start_time.minute * 60 + eog_start_time.second
    thor_start_time = thor_start_time.hour * 3600 + thor_start_time.minute * 60 + thor_start_time.second

    # Extract sample rate for each signal
    eeg_sample_rate = eeg_file.info['sfreq']  # 500
    ecg_sample_rate = ecg_file.info['sfreq']  # 1024
    eog_sample_rate = eog_file.info['sfreq']  # 256
    thor_sample_rate = thor_file.info['sfreq']  # 32

    # Extract signal data
    eeg_data = eeg_file.get_data()
    ecg_data = ecg_file.get_data()
    eog_data = eog_file.get_data()
    thor_data = thor_file.get_data()

    # Get signal lengths (number of samples)
    eeg_len = eeg_data.shape[1]
    ecg_len = ecg_data.shape[1]

    # print('eeg_len',eeg_len)
    # print('ecg_len',ecg_len)
    # print('eog_len',eog_len)
    # print('thor_len',thor_len)

    # Calculate the absolute time duration of the EEG and PSG signals
    eeg_time_len = eeg_len / eeg_sample_rate
    ecg_time_len = ecg_len / ecg_sample_rate

    # print('eeg_time_len',eeg_time_len)
    # print('ecg_time_len',ecg_time_len)

    # Obtain the end time for both the EEG and PSG devices separately
    eeg_end_time = eeg_start_time + eeg_time_len
    ecg_end_time = ecg_start_time + ecg_time_len

    # Check if the first timestamp from the annotation file matches the start time of the EEG signal
    if timestamp[0] == eeg_start_time:
        print('timestamp[0] = eeg_start_time')
    elif timestamp[0] - eeg_start_time == 1:
        print('timestamp[0] - eeg_start_time = 1')
    else:
        print('timestamp[0] - eeg_start_time = ',timestamp[0] - eeg_start_time)

    # Determine the latest start time among EEG, PSG, and annotation as the start time of the overlapping effective segment
    # and the earliest end time between EEG and PSG as the end time of the overlapping effective segment
    start_time = max(eeg_start_time, ecg_start_time, timestamp[0])  # Select the latest start time
    end_time = min(eeg_end_time, ecg_end_time)  # Select the earliest end time
    timestamp[0] = start_time   # Adjust the first timestamp to the start time

    # Ensure that the end time of the annotation file is before the end time of the overlapping segment of EEG and PSG
    assert end_time >= timestamp[-1]
    # Calculate the effective signal acquisition time
    effective_time = end_time - start_time

    # Annotation segmentation, associating annotations with time periods to form an array of annotations and time in seconds
    # Compute the time differences between each consecutive timestamp from the annotation file
    timestamp_diff = np.diff(timestamp)
    # Add the last segment duration
    timestamp_diff = np.append(timestamp_diff, end_time - timestamp[-1])
    segment_num = len(timestamp_diff)  # Add the last segment duration

    annotations = []
    for i in range(segment_num):
        annotations.append(
            np.full(int(timestamp_diff[i]), annotation_list[i]))    # Create an annotation for each second
    annotations = np.concatenate(annotations)  # Concatenate all segments into one array
    annotations = annotations[:int(effective_time) - 1]  # Round up the effective duration to the nearest integer minus one.

    # Crop EEG and PSG signals to the effective time range
    if start_time == timestamp[0]:
        annotation_eeg_gap_time = timestamp[0] - eeg_start_time
        annotation_psg_gap_time = timestamp[0] - ecg_start_time
        eeg_file.crop(tmin=annotation_eeg_gap_time, tmax=annotation_eeg_gap_time + int(effective_time) - 1)
        ecg_file.crop(tmin=annotation_psg_gap_time, tmax=annotation_psg_gap_time + int(effective_time) - 1)
        eog_file.crop(tmin=annotation_psg_gap_time, tmax=annotation_psg_gap_time + int(effective_time) - 1)
        thor_file.crop(tmin=annotation_psg_gap_time, tmax=annotation_psg_gap_time + int(effective_time) - 1)
    else:
        # Handle case where EEG and PSG start times are different
        gap_time = abs(ecg_start_time - eeg_start_time)
        # print('gap_time', gap_time)

        if start_time == eeg_start_time:
            eeg_file.crop(tmin=gap_time, tmax=gap_time + int(effective_time)  - 1)
            # print('tmax=gap_time + effective_time - 1', effective_time - 1)
            ecg_file.crop(tmin=0, tmax=int(effective_time)  - 1)
            eog_file.crop(tmin=0, tmax=int(effective_time)  - 1)
            thor_file.crop(tmin=0, tmax=int(effective_time)  - 1)
        elif start_time == ecg_start_time:
            eeg_file.crop(tmin=0, tmax=int(effective_time)  - 1)
            ecg_file.crop(tmin=gap_time, tmax=gap_time + int(effective_time)  - 1)
            eog_file.crop(tmin=gap_time, tmax=gap_time + int(effective_time)  - 1)
            thor_file.crop(tmin=gap_time, tmax=gap_time + int(effective_time)  - 1)

    # Read the cropped data
    eeg_data = eeg_file.get_data()
    ecg_data = ecg_file.get_data()

    # Calculate the time length of the cropped EEG and ECG signals
    eeg_crop_time_len = eeg_data.shape[1] / eeg_sample_rate
    psg_crop_time_len = ecg_data.shape[1] / ecg_sample_rate

    print('eeg_crop_time_len:',eeg_crop_time_len)
    print('psg_crop_time_len:',psg_crop_time_len)

    # Adjust the last timestamp difference
    timestamp_diff[-1] = timestamp_diff[-1] - 1
    # print('timestamp_diff[-1]',timestamp_diff[-1])

    # Convert timestamps to relative times based on start_time
    event_onset = timestamp - np.full(len(timestamp), start_time)

    # Set annotations for each signal (onset time, duration, label)
    eeg_file.set_annotations(mne.Annotations(onset=event_onset, duration=timestamp_diff,
                                             description=annotation_list))
    ecg_file.set_annotations(mne.Annotations(onset=event_onset, duration=timestamp_diff,
                                             description=annotation_list))
    eog_file.set_annotations(mne.Annotations(onset=event_onset, duration=timestamp_diff,
                                             description=annotation_list))
    thor_file.set_annotations(mne.Annotations(onset=event_onset, duration=timestamp_diff,
                                              description=annotation_list))

    # Define the event IDs and their corresponding annotations
    event_id = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '8': 8, '9': 9}

    # Create events for EEG, ECG, EOG, and Thor files based on annotations
    eeg_epoch_events, _ = mne.events_from_annotations(eeg_file, event_id=event_id, chunk_duration=1.)
    ecg_epoch_events, _ = mne.events_from_annotations(ecg_file, event_id=event_id, chunk_duration=1.)
    eog_epoch_events, _ = mne.events_from_annotations(eog_file, event_id=event_id, chunk_duration=1.)
    thor_epoch_events, _ = mne.events_from_annotations(thor_file, event_id=event_id, chunk_duration=1.)

    # Calculate the maximum time for each epoch, based on sample rate
    eeg_tmax = 1. - 1. / eeg_sample_rate  # The first point corresponds to time 0, and the corresponding time for the last point in each second is tmax.
    ecg_tmax = 1. - 1. / ecg_sample_rate
    eog_tmax = 1. - 1. / eog_sample_rate
    thor_tmax = 1. - 1. / thor_sample_rate

    # Create epochs for each signal type (EEG, ECG, EOG, Thor) with specified time range
    eeg_epochs = mne.Epochs(eeg_file, eeg_epoch_events, tmin=0, tmax=eeg_tmax, baseline=None,
                            preload=True)
    ecg_epochs = mne.Epochs(ecg_file, ecg_epoch_events, tmin=0, tmax=ecg_tmax, baseline=None, preload=True)
    eog_epochs = mne.Epochs(eog_file, eog_epoch_events, tmin=0, tmax=eog_tmax, baseline=None, preload=True)
    thor_epochs = mne.Epochs(thor_file, thor_epoch_events, tmin=0, tmax=thor_tmax, baseline=None, preload=True)

    # Extract the data for each signal type from the epochs
    eeg_epochs_data = eeg_epochs.get_data()
    ecg_epochs_data = ecg_epochs.get_data()
    eog_epochs_data = eog_epochs.get_data()
    thor_epochs_data = thor_epochs.get_data()

    # Extract the event IDs for the EEG data (annotations)
    one_subj_annotations = eeg_epochs.events[:, -1]
    # print('eeg_epochs.events',eeg_epochs.events)

    return eeg_epochs_data, ecg_epochs_data, eog_epochs_data, thor_epochs_data, one_subj_annotations

def init_path(save_path, id, type):
    # Construct the filename, which includes the preprocessed data's ID and type
    fname = f"MPDDF_preprocessed_{id}_{type}.mat"
    # Construct the filename, which includes the preprocessed data's ID and type
    data_path = os.path.join(save_path, fname)
    # Return the complete data path
    return data_path

def get_logger(path, name):
    # The function get_logger creates a logger that logs messages to both the console and a file.
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s:%(levelname)s: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(filename=path)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


if __name__ == '__main__':
    # Path Structure
    # The base directory (path) contains all raw data, organized into three subfolders:
    # EEG: Stores electroencephalogram (EEG) data files.
    # PSG: Stores polysomnography (PSG) data files.
    # Annotation: Stores expert-annotated fatigue level labels.
    path = '/data/MPDDF/rawdata'
    annotation_path = os.path.join(path, 'Annotation')
    eeg_path = os.path.join(path, 'EEG')
    psg_path = os.path.join(path, 'PSG')

    # List all the files in each subfolder
    annotation_files_name = os.listdir(annotation_path)
    eeg_files_name = os.listdir(eeg_path)
    psg_files_name = os.listdir(psg_path)

    # Print the names of the files in each folder
    print('annotation_files', annotation_files_name)
    print('eeg_files', eeg_files_name)
    print('psg_files', psg_files_name)

    # Due to potential inconsistencies in file ordering, the data files are sorted based on numerical values embedded in their filenames.
    sorted_annotation_files_name = sorted(annotation_files_name, key=lambda x: int(re.search(r'raw_(\d{2})', x).group(1)))
    sorted_eeg_files_name = sorted(eeg_files_name, key=lambda x: int(re.search(r'raw_(\d{2})', x).group(1)))
    sorted_psg_files_name = sorted(psg_files_name, key=lambda x: int(re.search(r'raw_(\d{2})', x).group(1)))

    # Print the sorted file names
    print('sorted_annotation_files_name', sorted_annotation_files_name)
    print('sorted_eeg_files_name', sorted_eeg_files_name)
    print('sorted_psg_files_name', sorted_psg_files_name)

    # Get the number of files in each folder
    annotation_len = len(sorted_annotation_files_name)
    eeg_len = len(sorted_eeg_files_name)
    psg_len = len(sorted_psg_files_name)

    # Ensure all folders have the same number of files
    assert annotation_len == eeg_len == psg_len

    # Ensure all folders have the same number of files
    for i in range(0, annotation_len):

        # print('annotation_files_name(i)', sorted_annotation_files_name[i])
        # print('eeg_files_name(i)', sorted_eeg_files_name[i])
        # print('psg_files_name(i)', sorted_psg_files_name[i])

        # Build the full file path for each file
        annotation_file = os.path.join(annotation_path, sorted_annotation_files_name[i])
        eeg_file = os.path.join(eeg_path, sorted_eeg_files_name[i])
        psg_file = os.path.join(psg_path, sorted_psg_files_name[i])

        # Load the annotations and timestamps from the annotation file
        timestamp, annotation_list = load_annotation_one_subj(annotation_file)
        # print(timestamp[0])

        # The expert-annotated annotations were integrated into the EEG, ECG, EOG, and respiratory effort(Thor) data at a per-second resolution.
        eeg_epochs_data, ecg_epochs_data, eog_epochs_data, thor_epochs_data, one_subj_annotations = load_eeg_psg_one_subj(
            eeg_file,
            psg_file,
            timestamp,
            annotation_list)

        # path to save the preprocessed data
        prepath = '/data/MPDDF/PreprocessedDatasetwithIntegratedAnnotations'

        # Extract ID of the participant from the PSG file path
        id = re.findall(r'\d+', eeg_file)[0]

        # Create and build the path to save data
        save_path = os.path.join(prepath, str(id))
        # Create the directory if it does not exist
        os.makedirs(save_path, exist_ok=True)

        # Set up the logger and output the log file
        logger = get_logger('{}/log.txt'.format(save_path),
                            f'{prepath}-{id}')
        logger.info('Using args :{}'.format(save_path))

        # Get the file paths for each type of signal (EEG, ECG, EOG, RES)
        fname_eeg = init_path(save_path, id, 'EEG')
        fname_ecg = init_path(save_path, id, 'ECG')
        fname_eog = init_path(save_path, id, 'EOG')
        fname_thor = init_path(save_path, id, 'RES')

        # Save the data for each signal type along with annotations into .mat files
        savemat(fname_eeg, mdict={'eeg_data': eeg_epochs_data, 'annotation': one_subj_annotations})
        savemat(fname_ecg, mdict={'ecg_data': ecg_epochs_data, 'annotation': one_subj_annotations})
        savemat(fname_eog, mdict={'eog_data': eog_epochs_data, 'annotation': one_subj_annotations})
        savemat(fname_thor, mdict={'thor_data': thor_epochs_data, 'annotation': one_subj_annotations})