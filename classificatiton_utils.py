import numpy as np
import sys

import ds_utils
from datetime import datetime
from computer import Computer


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


def knn_class_computer_factory(computer_func_params):
    # print(computer_func_params)
    true_classes_model = computer_func_params["true_classes_model"]
    # compute_model(true_classes_model), а сейчас предполагаем, что это обязательно ds
    true_classes = ds_utils.read_array(true_classes_model)

    def computer_(nearest_indices):
        return knn_compute_classes(nearest_indices, true_classes)

    return Computer(computer_)


classification_type__computer_factory = {
    "knn": knn_class_computer_factory
}
