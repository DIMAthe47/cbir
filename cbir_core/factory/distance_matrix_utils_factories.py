from collections import Iterable

import numpy as np
from core.quantization.pq_quantizer import restore_from_clusters
from sklearn.metrics import pairwise_distances

from cbir_core.computer import computer_utils
from cbir_core.util import ds_utils
from cbir_core.util.distance_matrix_utils import l2_distance_matrix, asymmetric_distance_matrix, \
    symmetric_distance_matrix
from cbir_core.computer.computer import Computer


def l2_distance_matrix_computer_factory(computer_func_params):
    # X_ds = computer_func_params["base_model"]["output_model"]
    # X = ds_utils.read_array(X_ds)
    X = computer_utils.compute_model(computer_func_params["base_model"])

    # TODO consider all cases
    if isinstance(X, np.ndarray):
        pass
    elif isinstance(X, Iterable):
        X = list(X)

    def computer_(Q):
        Q = np.array(Q, copy=False)
        Q = Q.reshape((len(Q), -1))
        distances_matrix = l2_distance_matrix(Q, X)
        return distances_matrix

    return Computer(computer_)


def asymmetric_distance_matrix_computer_factory(computer_func_params):
    quantization_model = computer_func_params["base_model"]["computer_func_params"]["quantization_model"]
    # print(quantization_model)
    X_ds = computer_func_params["base_model"]["output_model"]
    # print("X_DS", X_ds)
    X = ds_utils.read_array(X_ds)
    cluster_centers_ds = quantization_model["output_model"]
    cluster_centers = ds_utils.read_array(cluster_centers_ds)

    def computer_(Q):
        distances_matrix = asymmetric_distance_matrix(cluster_centers, Q, X)
        return distances_matrix

    return Computer(computer_)


def symmetric_distance_matrix_computer_factory(computer_func_params):
    quantization_model = computer_func_params["base_model"]["computer_func_params"]["quantization_model"]
    # print(quantization_model)
    X_ds = computer_func_params["base_model"]["output_model"]
    # print("X_DS", X_ds)
    X = ds_utils.read_array(X_ds)
    cluster_centers_ds = quantization_model["output_model"]
    cluster_centers = ds_utils.read_array(cluster_centers_ds)
    pq_quantizer = restore_from_clusters(cluster_centers)

    n_quantizers, n_clusters, subvector_length = cluster_centers.shape
    print("n_quantizers, n_clusters, subvector_length", n_quantizers, n_clusters, subvector_length)
    cluster_centers_distance_matrix = np.empty((n_quantizers, n_clusters, n_clusters), dtype=float)
    for i in range(n_quantizers):
        cluster_centers_distance_matrix[i] = pairwise_distances(cluster_centers[i], cluster_centers[i])

    def computer_(Q):
        distances_matrix = symmetric_distance_matrix(cluster_centers_distance_matrix, Q, X, pq_quantizer)
        return distances_matrix

    return Computer(computer_)


type__distance_matrix_computer_factory = {
    "l2": l2_distance_matrix_computer_factory,
    "adc": asymmetric_distance_matrix_computer_factory,
    "sdc": symmetric_distance_matrix_computer_factory
}
