import scipy.io
import numpy as np
import os
import torch


def mat_to_np(file_path):
    # type: (str) -> (list, np.ndarray)
    """
    Load mat data in numpy

    :param file_path: file .mat
    :return: data np array
    """

    mat = scipy.io.loadmat(file_path)
    tracks = []

    for track in mat['tracks']:
        tracks.append(track[0])

    return tracks, mat['truth']


def Human_dataset(pathfile, images, verbose=False, subject=11):
    # type: (str, str, bool, int) -> (np.ndarray, np.ndarray)
    """
    Load Human3.6 dataset

    :param pathfile: File contains pose coordinates
    :param images:  Folder contains subject images
    :param verbose:  Verbose mode
    :param subject:  Filter data by subject
    :return: Pose joints (14), Images
    """

    USEFUL_JOINTS = [15, 24, 25, 26, 28, 17, 18, 20, 1, 2, 3, 6, 7, 8]
    data_dict = torch.load(pathfile)

    data, frames = [], []

    for key in data_dict:

        if subject is not None:
            if key[0] != subject:
                continue

        print(f'Sequence ID: {key}') if verbose else None
        n_frames = data_dict[key].shape[0]
        print(f'\t> number of frames: {n_frames}') if verbose else None

        frame_numbers = []
        frame_list = []

        for i in range(n_frames):
            filename = images + 'S' + str(key[0]) + '/' + str(key[2][:-3]) + "_" + str(i) + ".jpg"
            if os.path.isfile(filename):
                frame_numbers.append(i)
                frame_list.append(filename)

        joints = data_dict[key][frame_numbers].reshape(-1, 32, 3)
        joints = joints[:, USEFUL_JOINTS, :]
        data.append(joints)
        frames.append(frame_list)
        print(joints.shape) if verbose else None

    data = np.concatenate(data, 0)
    frames = np.concatenate(frames, 0)
    p = np.random.permutation(data.shape[0])

    return data[p], frames[p]