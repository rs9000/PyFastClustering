import scipy.io
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
from sklearn import metrics
from random import randint
from tqdm import tqdm
import argparse
import numba
import numpy
from math import sqrt


@numba.jit(nopython=True, fastmath=True)
def euclidean_distance(a, b):
    # type: (np.ndarray, np.ndarray) -> float
    """
    Euclidean distance between point `a` and point `b`
    :param a: Point a
    :param b: Point b
    :return: Euclidean distance
    """

    assert a.shape[0] == b.shape[0]

    val = 0.
    for i in range(a.shape[0]):
        val += (a[i] - b[i]) ** 2
    return sqrt(val)


@numba.jit(nopython=True, fastmath=True)
def hu_distance(t1, t2):
    # type: (np.ndarray, np.ndarray) -> float
    """
    Compute the HU distance as the average Euclidean distance between points
    on two trajectories `t1` and `t2`

    :param t1: Trajectory a
    :param t2: Trajectory b
    :return: Hu distance
    """

    d = float(0)
    assert t1.shape[0] == t2.shape[0]

    for i in range(t1.shape[0]):
        d += euclidean_distance(t1[i], t2[i])

    return (d / t1.size)


@numba.jit(nopython=True, fastmath=True)
def hausdorff_distance(t1, t2):
    # type: (np.ndarray, np.ndarray) -> float
    """
    Compute the Hausdorff distance between two trajectories `t1` and `t2`

    :param t1: Trajectory a
    :param t2: Trajectory b
    :return: hausdorff distance
    """

    nA = t1.shape[0]
    nB = t2.shape[0]
    cmax = 0.
    for i in range(nA):
        cmin = np.inf
        for j in range(nB):
            d = euclidean_distance(t1[i, :], t2[j, :])
            if d<cmin:
                cmin = d
            if cmin<cmax:
                break
        if cmin>cmax and np.inf>cmin:
            cmax = cmin
    for j in range(nB):
        cmin = np.inf
        for i in range(nA):
            d = euclidean_distance(t1[i, :], t2[j, :])
            if d<cmin:
                cmin = d
            if cmin<cmax:
                break
        if cmin>cmax and np.inf>cmin:
            cmax = cmin
    return cmax


@numba.jit(nopython=True, fastmath=True)
def lcss_distance(t1, t2, th=100, delta=10):
    # type: (np.ndarray, np.ndarray, int, int) -> float
    """
    Compute LCSS distance between `t1` and `t2` trajectories

    :param t1: Trajectory a
    :param t2: Trajectory b
    :param th: Threshold
    :param delta: Delta
    :return: LCSS distance
    """


    m = t1.shape[0]
    n = t2.shape[0]

    L = np.full((n+1, m+1), 0.)

    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i, j] = 0
            elif euclidean_distance(t1[i - 1], t2[j - 1]) < th and abs(i-j) < delta:
                L[i, j] = L[i-1, j-1] + 1
            else:
                L[i, j] = max(L[i-1, j], L[i, j - 1])

    return 1 - (L[-1, -1] / min(m, n))


def filter_flow(flow, th=0.1):
    # type: (np.ndarray, float) -> np.ndarray
    """
    Filter input flow keeping only the significant points.
    :param flow: unfiltered flow
    :param th: significance threshold; np.ndarray with shape (N_POINTS, 4)
    :return: filtered flow; ; np.ndarray with shape (N_FILTERED_POINTS, 4)
    """
    cf = []
    for i in range(flow.shape[0]):
        if i != flow.shape[0] - 1:
            d = np.linalg.norm(flow[i][2:] - flow[i + 1][2:])
            if d > th:
                cf.append(flow[i])
    return np.array(cf).T


@numba.jit(nopython=True, fastmath=True)
def dist_to_sim_mx(dist_matrix, sigma=15):
    # type: (np.ndarray, int) -> np.ndarray
    """
    Distance matrix to similarity matrix using Gaussian Kernel

    :param dist_matrix:  Distance matrix
    :param sigma: Sigma
    :return: Similarity matrix
    """
    return np.exp(- dist_matrix ** 2 / (2. * sigma ** 2))


def similarity_matrix(trajectories, distance='hausdorff'):
    # type: (list, str) -> np.ndarray
    """
    Compute similarity matrix

    :param trajectories: Trajectories
    :param distance: Distance function
    :return: Similarity matrix
    """

    len_t = len(trajectories)
    sim_matrix = np.full((len_t,len_t), 0.)

    with tqdm(desc='Compute similarity matrix: ', unit='it', total=len_t) as pbar:
        for i in range(0, len_t):
            for j in range(0, len_t):

                t1, t2 = trajectories[i].T, trajectories[j].T

                if distance == 'hu':
                    sim_matrix[i, j] = hu_distance(t1, t2)
                elif distance == 'hausdorff':
                    sim_matrix[i, j] = hausdorff_distance(t1, t2)
                elif distance == 'lcss':
                    sim_matrix[i, j] = lcss_distance(t1, t2)
            pbar.update(1)
    return sim_matrix


def resample_traj(t, len_seq, kind='linear'):
    # type: (np.ndarray, int, str) -> np.ndarray
    """
    Resample trajectory `t` to fixed length `len_seq`

    :param t: trajectory
    :param len_seq:  New length
    :param kind: Interpolation type
    :return: Resampled trajectory
    """

    y = np.full((t.shape[0], len_seq), 0)

    f = interp1d(np.linspace(0, 1, t[0].size), t[0], kind)
    y[0] =  f(np.linspace(0, 1, len_seq))

    f = interp1d(np.linspace(0, 1, t[1].size), t[1], kind)
    y[1] = f(np.linspace(0, 1, len_seq))

    return y


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


def track_to_flow(tracks, flow_th=0.1):
    # type: (list, float) -> list
    """
    Convert input track to corresponding flow.

    :param track: (x, y) sequence; np.ndarray with shape (2, N_POINTS)
    :return: flow: (x, y, dx, dy) sequence; np.ndarray with shape (N_POINTS, 4)
    """

    with tqdm(desc='Trajectories to flows: ', unit='it', total=len(tracks)) as pbar:
        for i in range(len(tracks)):
            x, y = tracks[i][0], tracks[i][1]
            d = tracks[i][:, 1:] - tracks[i][:, :-1]
            d = np.pad(d, ((0, 0), (1, 0)), 'constant', constant_values=0)
            d = d / (numpy.linalg.norm(d + 0.0000001, axis=0))
            dx, dy = d[0], d[1]
            tracks[i] = np.array([x, y, dx, dy])

            if flow_th != 0:
                tracks[i] = filter_flow(tracks[i].T, th=flow_th)
            pbar.update(1)

    return tracks


def show_tracks(file_path, n=0, save_pics=True):
    # type: (str, float, bool) -> None
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
    fig.show()
    if save_pics:
        fig.savefig("./pics/Figure_0.jpg")
    plt.close()


def mean_trajectories(tracks, labels):
    # type: (np.ndarray, np.ndarray) -> list
    """
    Compute mean path from trajectories of each cluster
    >> NOTE: To compute mean trajectories must be of the same length

    :param tracks:  Trajectories
    :param labels:  Cluster-ids
    :return:  Mean paths
    """

    tracks_mean = []

    # Compute mean trajectories
    for i in range(max(labels)):
        idxs = [np.where(labels == i)]
        traj = np.mean(np.array(tracks)[np.squeeze(idxs)], axis=0)
        tracks_mean.append(traj)

    # Plot
    fig = plt.figure()
    plt.title('Most frequent trajectories')
    for t in tracks_mean:
        xs, ys = t[0], t[1]
        color = (randint(64, 255) / 255, randint(64, 255) / 255, randint(64, 255) / 255)
        plt.plot(xs, ys, ',-', color=color, linewidth=2)

    fig.show()
    fig.savefig("./pics/frequent_paths.jpg")
    plt.close()

    return tracks_mean


def show_tracks_labels(file_path, labels, n=17, save_pics=True):
    # type: (str, np.ndarray, int, bool) -> None
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

    fig.show()
    if save_pics:
        fig.savefig("./pics/cluster_n%d.jpg" % n)
    plt.close()


def clustering_traj(tracks, gt, method='kmeans', fixed_length=100, num_samples=0, distance="hu",
                    sigma=15, flow=True, flow_th=0., n_cluster=8):
    # type: (list, np.ndarray, str, int, int, str, int, bool, float, int) -> (np.ndarray, np.ndarray)
    """
    High level function to cluster trajectories

    :param tracks: Trajectories
    :param gt: Ground-truth labels (clusters-id)
    :param method: Clustering method
    :param fixed_length: Interpolate trajectories to fixed length
    :param num_samples: Num of samples to cluster
    :param distance: Distance measure to compute similarity matrix
    :param sigma: Value used to compute gaussian kernel
    :param flow: Convert trajectories to flows
    :param flow_th: Filter flow threeshold
    :param n_cluster: Number of clusters
    :return: Labels predicted, Ground-truth labels
    """

    clustering, dist_matrix = None, None

    if fixed_length != 0:
        for i in range(0, len(tracks)):
            tracks[i] = resample_traj(tracks[i], len_seq=100)

    if flow:
        assert (distance == 'hu' and flow_th > 0) != True
        tracks = track_to_flow(tracks, flow_th=flow_th)

    if num_samples != 0:
        tracks = tracks[:num_samples]
        gt = gt[:num_samples]

    dist_matrix = similarity_matrix(tracks, distance=distance)
    sim_matrix = dist_to_sim_mx(dist_matrix, sigma=sigma)

    if method == 'kmeans':
        clustering = KMeans(n_clusters=n_cluster)
    elif method == 'spectral':
        clustering = SpectralClustering(n_clusters=n_cluster, affinity='precomputed')

    labels = clustering.fit_predict(sim_matrix)
    return labels, gt


def main(args):

    show_tracks(file_path=args.data, n=args.num_samples, save_pics=args.save_pics)

    tracks, gt = mat_to_np(file_path=args.data)

    labels, gt = clustering_traj(tracks, gt, num_samples=args.num_samples, method=args.method,
                                 n_cluster=args.n_cluster, fixed_length=args.fixed_length,
                                 flow=args.flow, flow_th=args.flow_th, distance=args.distance_function,
                                 sigma=args.sigma)

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(np.squeeze(gt), labels))
    print("Completeness: %0.3f" % metrics.completeness_score(np.squeeze(gt), labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(np.squeeze(gt), labels))

    if args.fixed_length != 0:
        mean_trajectories(np.array(tracks), labels)

    for i in range(0, labels.max() +1):
        show_tracks_labels(args.data, labels, i, save_pics=args.save_pics)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Clustering trajectories')
    parser.add_argument('--data', type=str,
                        help='File containing data', default='ds/cross.mat')
    parser.add_argument('--num_samples', type=int,
                        help='Use n_samples from data (0: use all data)', default=0)
    parser.add_argument('--method', type=str,
                        help='Clustering method ( kmeans | spectral )', default='kmeans')
    parser.add_argument('--n_cluster', type=int,
                        help='Number of clusters', default=15)
    parser.add_argument('--fixed_length', type=int,
                        help='Resample trajectories to fixed length', default=100)
    parser.add_argument('--flow', type=bool,
                        help='Convert trajectories to flow before clustering', default=True)
    parser.add_argument('--flow_th', type=float,
                        help='Filter flow using a threeshold', default=0.)
    parser.add_argument('--distance_function', type=str,
                        help='Distance function used to compute similarity (hu | hausdorff | lcss )', default='hu')
    parser.add_argument('--sigma', type=int,
                        help='Sigma value in gaussian kernel', default=15)
    parser.add_argument('--save_pics', type=bool,
                        help='Save figures', default=True)

    args = parser.parse_args()
    print(args)
    main(args)