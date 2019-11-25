import scipy.io
from matplotlib import pyplot as plt
from random import randint
from metrics import *
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import os


def show_tracks(file_path, n=0, save_pics=True, background=False):
    # type: (str, float, bool, bool) -> None
    """
    Read data from `file_path` and visualize data ( `n` samples)

    :param file_path: File containing data
    :param n: Num of samples
    :return: None
    """

    if n == 0:
        n = float('Inf')

    mat = scipy.io.loadmat(file_path)

    fig = plt.figure()
    for i, track in enumerate(mat['tracks']):
        track = track[0]
        xs, ys = track[0], track[1]
        plt.plot(xs, ys, ',-', linewidth=1)
        if i > n:
            break

    if save_pics:
        fig.savefig("./pics/Figure_0.jpg")

    if not background:
        fig.show()


def show_tracks_labels(file_path, labels, n=17, save_pics=True, background=False):
    # type: (str, np.ndarray, int, bool, bool) -> None
    """
    Visualize clusters from labelled data

    :param file_path: File containing data
    :param labels: Cluster-id
    :param n: Id colored cluster
    :return: None
    """

    color = (randint(64, 255) /255, randint(64, 255)/255, randint(64, 255)/255)
    mat = scipy.io.loadmat(file_path)
    labels = np.squeeze(labels)

    it_gray_paths = ((track[0], label) for track, label in zip(mat['tracks'], labels) if label != n)
    it_color_paths = ((track[0], label) for track, label in zip(mat['tracks'], labels) if label == n)

    fig = plt.figure()
    plt.title('Cluster n°' + str(n))

    for track, label in it_gray_paths:
        xs, ys = track[0], track[1]
        plt.plot(xs, ys, ',-', color='lightgray', linewidth=1)

    for track, label in it_color_paths:
        xs, ys = track[0], track[1]
        plt.plot(xs, ys, ',-', color=color, linewidth=1)

    if save_pics:
        fig.savefig("./pics/cluster_n%d.jpg" % n)

    if not background:
        fig.show()


def show_skeleton(joints, frames=None, labels=None, background=False, remove_pics=False):
    # type: (np.ndarray, np.ndarray, np.ndarray, bool, bool) -> None
    """
    Show skeletons amd save pics separated by cluster

    :param joints: Pose data
    :param frames: Pics
    :param labels: Cluster ids
    :param background: Background mode
    :return:
    """

    if remove_pics:
        os.system("find ./clusters/ -name '*.jpg' -delete")

    bone_list = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
                     [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13]]

    pose_center = np.zeros((np.max(labels)+1 , 14, 3))

    if labels is not None:
        fig = plt.figure()
        fig.suptitle("Pose Clusters")

        for lab in range(np.max(labels) +1):
            ax = fig.add_subplot(3, 5, lab+1, projection='3d')
            ax.set_xlim(-0.7, 0.7)
            ax.set_ylim(-0.7, 0.7)
            ax.set_zlim(-0.7, 0.7)
            ax.view_init(azim=-90, elev=100)

            idx = np.where(labels== lab)[0]
            pose_center[lab] = np.mean(joints[idx], 0)

            cx, cy, cz = pose_center[lab].T[0], pose_center[lab].T[1], pose_center[lab].T[2]
            ax.scatter3D(cx, cy, cz, color='red')

            for bone in bone_list:
                ax.plot3D([cx[bone[0]], cx[bone[1]]], [cy[bone[0]], cy[bone[1]]], [cz[bone[0]], cz[bone[1]]],color='red')

        plt.show()
        plt.close()

    with tqdm(desc='Processing pictures: ', unit='it', total=joints.shape[0]) as pbar:
        for i, joint in enumerate(joints):
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.set_xlim(-0.7, 0.7)
            ax2.set_ylim(-0.7, 0.7)
            ax2.set_zlim(-0.7, 0.7)
            ax2.view_init(azim=-90, elev=100)

            color = 'blue'
            x, y, z = joint.T[0], joint.T[1], joint.T[2]
            cx, cy, cz = pose_center[labels[i]].T[0],  pose_center[labels[i]].T[1],  pose_center[labels[i]].T[2]
            distance = hu_distance(np.array([x, y, z ]).T, np.array([cx, cy, cz ]).T)
            ax2.set_title("Distance: " + str(distance)[:6])

            ax2.scatter3D(x, y, z, color=color)
            ax2.scatter3D(cx, cy, cz, color='red')

            for bone in bone_list:
                ax2.plot3D([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], [z[bone[0]], z[bone[1]]], color=color)
                ax2.plot3D([cx[bone[0]], cx[bone[1]]], [cy[bone[0]], cy[bone[1]]], [cz[bone[0]], cz[bone[1]]], color='red')

            if frames is not None:
                img = mpimg.imread(frames[i])
                ax1.imshow(img)

            if labels is not None:
                plt.savefig("./clusters/" + str(labels[i]) + "/" + str(i) + ".jpg")

            if not background:
                plt.show()
            pbar.update(1)
            plt.close()