import scipy.io
import numpy as np
import os
import torch
import json


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


def Human_dataset(pathfile, images, max_pic_per_key=float('Inf'), verbose=False):
    # type: (str, str, int, bool) -> (np.ndarray, np.ndarray)
    """
    Load S11 folder from Human3.6 dataset

    :param pathfile: File contains pose coordinates
    :param images:  Folder contains subject images
    :param verbose:  Verbose mode
    :param subject:  Filter data by subject
    :return: Pose joints (14), Images
    """

    USEFUL_JOINTS = [15, 24, 25, 26, 28, 17, 18, 20, 1, 2, 3, 6, 7, 8]
    data_dict = torch.load(pathfile)

    with open("./ds/s11_frame_list.json") as json_file:
        s11_list = set(json.load(json_file))

    data, frames = [], []
    iterator = ((k,v) for (k, v) in data_dict.items() if k[0] == 11)

    for (key, val) in iterator:

        print(f'Sequence ID: {key}') if verbose else None
        n_frames = val.shape[0]
        print(f'\t> number of frames: {n_frames}') if verbose else None

        frame_numbers = []
        frame_list = []

        for i in range(n_frames):
            filename = str(key[2][:-3]) + "_" + str(i) + ".jpg"
            filepath = images + '/' + filename
            if filename in s11_list:
                frame_numbers.append(i)
                frame_list.append(filepath)
            if len(frame_list) > max_pic_per_key:
                break

        joints = val[frame_numbers].reshape(-1, 32, 3)
        joints = joints[:, USEFUL_JOINTS, :]
        data.append(joints)
        frames.append(frame_list)
        print(joints.shape) if verbose else None

    data = np.concatenate(data, 0)
    frames = np.concatenate(frames, 0)
    p = np.random.permutation(data.shape[0])

    return data[p], frames[p]