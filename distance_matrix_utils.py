import numpy as np
import sys
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters

sys.path.append('/home/dimathe47/PycharmProjects/pq')
import ds_utils
from computer import Computer
import model_utils as mu


def l2_distance_matrix(queries, descriptors):
    distances_matrix = pairwise_distances(queries, descriptors, metric="l2")
    return distances_matrix


def asymmetric_distance_matrix(cluster_centers, query_descriptors, db_pqcodes):
    distance_matrix = np.empty((len(query_descriptors), len(db_pqcodes)))
    for i in range(len(query_descriptors)):
        distance_matrix[i] = asymmetric_distance(cluster_centers, query_descriptors[i], db_pqcodes)
    return distance_matrix


def asymmetric_distance(cluster_centers, query_descriptor, db_pqcodes):
    n_quantizers, n_clusters, subvector_length = cluster_centers.shape
    assert np.issubdtype(db_pqcodes.dtype, int)
    pqcodes_T = db_pqcodes.T
    db_distances = np.zeros((len(db_pqcodes),), dtype=float)
    # print(query_descriptor)
    query_descriptor = query_descriptor.reshape((n_quantizers, subvector_length))
    for i in range(n_quantizers):
        subvector = query_descriptor[i, :].reshape((1, -1))
        #         print(subvector.shape,  cluster_centers[i].shape)
        distances_to_clusters = pairwise_distances(subvector, cluster_centers[i]).ravel()
        db_distances[...] += np.take(distances_to_clusters, pqcodes_T[i])

    return db_distances


def symmetric_distance_matrix(cluster_center_distance_matrix, query_descriptors, db_pqcodes, pq_quantizer):
    distance_matrix = np.empty((len(query_descriptors), len(db_pqcodes)))
    for i in range(len(query_descriptors)):
        distance_matrix[i] = symmetric_distance(cluster_center_distance_matrix, query_descriptors[i], db_pqcodes,
                                                pq_quantizer)
    return distance_matrix


def symmetric_distance(cluster_center_distance_matrix, query_descriptor, db_pqcodes, pq_quantizer):
    n_quantizers = cluster_center_distance_matrix.shape[0]
    n_clusters = cluster_center_distance_matrix.shape[1]
    assert np.issubdtype(db_pqcodes.dtype, int)
    db_distances = np.zeros((len(db_pqcodes),), dtype=float)
    query_descriptor = query_descriptor.reshape((1, -1))
    query_pqcodes = pq_quantizer.predict_subspace_indices(query_descriptor).ravel()
    db_pqcodes_T = db_pqcodes.T
    for i in range(n_quantizers):
        matrix_indices = np.ravel_multi_index([query_pqcodes[i], db_pqcodes_T[i]], (n_clusters, n_clusters))
        # print(matrix_indices.shape)
        db_cluster_query_cluster_distance_arr = np.take(cluster_center_distance_matrix[i], matrix_indices)
        db_distances[...] += db_cluster_query_cluster_distance_arr.ravel()

    return db_distances


def unoptimzed_asymmetric_distance(query_descriptor, db_pqcode, **kwargs):
    # print(kwargs)
    cluster_centers = kwargs["cluster_centers"]
    n_quantizers, n_clusters, subvector_length = cluster_centers.shape
    # print(db_pqcode, query_descriptor)
    db_pqcode = np.array(db_pqcode, dtype=int)
    assert np.issubdtype(db_pqcode.dtype, int)
    db_distance = 0
    # print(query_descriptor)
    query_descriptor = query_descriptor.reshape((n_quantizers, subvector_length))
    for i in range(n_quantizers):
        subvector = query_descriptor[i, :].reshape((1, -1))
        #         print(subvector.shape,  cluster_centers[i].shape)
        distances_to_clusters = pairwise_distances(subvector, cluster_centers[i]).ravel()
        db_distance += np.take(distances_to_clusters, db_pqcode[i])

    return db_distance


def l2_distance_matrix_computer_factory(computer_func_params):
    X_ds = computer_func_params["base_model"]["output_model"]
    X = ds_utils.read_array(X_ds)

    def computer_(Q):
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


def compute_distance_matrix(distance_matrix_model):
    queries = ds_utils.read_array(distance_matrix_model["input_ds"])
    queries_len = len(queries)
    descriptors = ds_utils.read_array(distance_matrix_model["kwargs"]["descriptors_model"]["output_ds"])
    #     descriptors=normalize(descriptors, norm="max") #normalize here if need
    computer_factory = type__distance_matrix_computer_factory[distance_matrix_model["type"]]
    computer_ = computer_factory(distance_matrix_model["kwargs"])
    distance_matrix = computer_.compute(queries, descriptors)
    ds_utils.save_array(distance_matrix, distance_matrix_model["output_ds"], attrs={"model": distance_matrix_model})
