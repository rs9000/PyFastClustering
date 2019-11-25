import argparse
from tqdm import tqdm
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
from metrics import *
from visualizzation import show_skeleton
from utils import Human_dataset


def preprocess_data(data):
    # type: (np.ndarray) -> np.ndarray
    """
    Preprocess pose
    1) Coordinate relative to the root joint
    2) Swap y axis
    3) Normalize values

    :param data: Data
    :return: Processed data
    """

    root = (data[:, 8] + data[:, 11]) / 2
    data = data - np.expand_dims(root, 1)
    data[:, :, 1] = -data[:, :, 1]
    data = data / np.max(np.abs(data))
    data = rotate_pose(data)

    return data


@numba.jit(nopython=True, fastmath=True)
def rotate_pose(data, rot_joint=8):
    # type: (np.ndarray, int) -> np.ndarray
    """
    Rotate pose respect to `rot_joint` to align with x-axis

    :param data: Pose
    :param rot_joint: Joint of rotation
    :return: Rotated pose
    """

    for i in range(data.shape[0]):
      # X, Z coordinates of ankle joint
      m = np.arctan2(data[i, rot_joint, 0], data[i, rot_joint, 2])

      # Rotation Matrix
      R = np.array(([np.cos(m), 0, np.sin(m)],
                    [0, 1, 0],
                    [-np.sin(m), 0, np.cos(m)]))

      data[i] = data[i] @ R

    return data


@numba.jit(nopython=True, fastmath=True)
def coord_to_thetas(data):
    # type: (np.ndarray) -> np.ndarray
    """
    Convert 3d point coordinate to 3 angles (radians) respect axes
    alpha = angle with X-axis
    beta = angle with Y-axis
    gamma = angle with Z-axis

    :param data: Data
    :return:  Data (angles in radians)
    """

    thetas = np.zeros((data.shape[0], 14, 3))

    for i in range(len(data)):
        x, y, z = data[i][:, 0], data[i][:, 1], data[i][:, 2]

        for j in range(data.shape[1]):
            gamma = np.arctan(np.sqrt(np.power(x[j], 2) + np.power(y[j], 2)) / z[j])
            alpha = np.arctan(np.sqrt(np.power(z[j], 2) + np.power(y[j], 2)) / x[j])
            beta = np.arctan(np.sqrt(np.power(x[j], 2) + np.power(z[j], 2)) / y[j])
            thetas[i, j] = np.array((alpha, beta, gamma))

    return thetas


def distance_matrix(joints, distance='hu', w=None):
    # type: (np.ndarray, str, np.ndarray) -> np.ndarray
    """
    Compute distance matrix

    :param joints: Data
    :param distance:  Distance method
    :param w:  Weights joints (L=n_joints)
    :return: distance matrix
    """

    len_t = joints.shape[0]
    distance_matrix = np.full((len_t,len_t), 0.)

    with tqdm(desc='Compute similarity matrix: ', unit='it', total=len_t) as pbar:
        for i in range(0, len_t):
            for j in range(0, len_t):

                if j < i:
                    continue

                t1, t2 = joints[i].copy(), joints[j].copy()

                if distance == 'hu':
                    distance_matrix[i, j] = hu_distance(t1, t2, w)

            pbar.update(1)
        distance_matrix = distance_matrix + distance_matrix.T - np.diag(distance_matrix.diagonal())
    return distance_matrix


def cluster_pose(dist_matrix, method='kmeans', sigma=0.03):
    # type: (np.ndarray, str, float) -> np.ndarray
    """
    Clustering data

    :param dist_matrix: Distance matrix
    :param method:  Clustering method
    :param sigma:  Sigma value
    :return:  Labels
    """

    sim_matrix = dist_to_sim_mx(dist_matrix, sigma=sigma)

    if method == 'kmeans':
        clustering = KMeans(n_clusters=args.n_clusters)
    elif method == 'spectral':
        clustering = SpectralClustering(n_clusters=args.n_clusters)
    else:
        raise KeyError

    labels = clustering.fit(sim_matrix).labels_
    return labels


def main():

    n = args.num_samples
    data, frames = Human_dataset(args.data, args.images)
    data = preprocess_data(data)

    if args.use_angles:
        data2 = coord_to_thetas(data)
        data = np.concatenate((data, data2), axis=2)

    dist_matrix = distance_matrix(data[:n])
    labels = cluster_pose(dist_matrix, method=args.method, sigma=args.sigma)

    show_skeleton(data[:n], frames[:n], labels[:n], background=True, remove_pics=True)


    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Clustering trajectories')
    parser.add_argument('--data', type=str,
                        help='File containing Human3.6 data', default='/nas/majinbu/capra/Datasets/Human3.6m/data/human_3d_test.data')
    parser.add_argument('--images', type=str,
                        help='Folder contains Human3.6 subject images', default='/nas/majinbu/capra/Datasets/Human3.6m/frames_all/S11/')
    parser.add_argument('--method', type=str,
                        help='Clustering method ( kmeans | spectral )', default='spectral')
    parser.add_argument('--n_clusters', type=int,
                        help='Number of clusters', default=10)
    parser.add_argument('--sigma', type=float,
                        help='Sigma value', default=0.04)
    parser.add_argument('--num_samples', type=int,
                        help='Use n_samples from data', default=500)
    parser.add_argument('--use_angles', type=bool,
                        help='Compute pose angles', default=False)

    args = parser.parse_args()
    print(args)
    main()