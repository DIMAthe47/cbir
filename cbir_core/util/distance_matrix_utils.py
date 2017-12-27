import numpy as np
from sklearn.metrics import pairwise_distances

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
