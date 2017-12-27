import numpy as np


def knn_compute_classes(nearest_indices, classes):
    n_queries = len(nearest_indices)
    n_classes = classes.max() + 1
    n_nearest = nearest_indices.shape[1]
    queries_classes = np.empty((n_queries, n_nearest), dtype=int)
    for n_query in range(n_queries):
        # self_index_in_nearest_indices=
        occurences = np.zeros((n_classes,), dtype=int)
        nearest_classes = np.take(classes, nearest_indices[n_query])
        # print("nearest_classes", nearest_classes)
        for n_nearest_ in range(n_nearest):
            occurences[nearest_classes[n_nearest_]] += 1
            # print("occurences", occurences)
            queries_classes[n_query][n_nearest_] = occurences.argmax()
    return queries_classes
