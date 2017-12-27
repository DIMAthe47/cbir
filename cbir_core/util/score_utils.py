import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from cbir_core.util import ds_utils
from sklearn.metrics import accuracy_score
from cbir_core.computer.computer import Computer


def accuracy_score_computer_factory(computer_func_params):
    def computer_(queries_classes):
        true_classes_model = computer_func_params["true_classes_model"]
        # compute_model(true_classes_model), а сейчас предполагаем, что это обязательно ds
        true_classes = ds_utils.read_array(true_classes_model)
        assert len(queries_classes.shape) == 2
        n_nearest = queries_classes.shape[1]
        scores_arr = np.empty((n_nearest,))
        for n_nearest_ in range(n_nearest):
            scores_arr[n_nearest_] = accuracy_score(true_classes, queries_classes[:, n_nearest_])
        return scores_arr

    return Computer(computer_)


def accuracy_KNeighbors_computer_factory(computer_func_params):
    # if computer_func_params["base_model"]["computer_func_name"] == "pqcode":
    #     X_ds = computer_func_params["base_model"]["output_model"]
    #     # print("X_DS", X_ds)
    #     X = ds_utils.read_array(X_ds)
    #     quantization_model = computer_func_params["base_model"]["computer_func_params"]["quantization_model"]
    #     # print(quantization_model)
    #     cluster_centers_ds = quantization_model["output_model"]
    #     cluster_centers = ds_utils.read_array(cluster_centers_ds)
    #     metric = unoptimzed_asymmetric_distance
    #     metric_params = {"cluster_centers": cluster_centers}
    # else:
    X_ds = computer_func_params["base_model"]["output_model"]
    X = ds_utils.read_array(X_ds)
    metric = "l2"
    metric_params = None

    n_neighbors = computer_func_params["n_nearest"]

    true_classes_model = computer_func_params["true_classes_model"]
    # compute_model(true_classes_model), а сейчас предполагаем, что это обязательно ds
    true_classes = ds_utils.read_array(true_classes_model)

    def computer_(Q):
        Q = Q.reshape((len(Q), -1))
        k_arr = [k for k in range(1, n_neighbors + 1)]
        accuracy_arr = np.empty(len(k_arr))
        predicted_classes = np.empty((len(k_arr), len(true_classes)), dtype=int)
        for i, k in enumerate(k_arr):
            neigh = KNeighborsClassifier(n_neighbors=k, metric=metric, metric_params=metric_params, algorithm="brute")
            neigh.fit(X, true_classes)
            pred = neigh.predict(Q)
            #         print(pred)
            predicted_classes[i] = pred
            accuracy_arr[i] = accuracy_score(true_classes, pred)
        return accuracy_arr

    return Computer(computer_)


score_type__computer_factory = {
    "accuracy": accuracy_score_computer_factory,
    "accuracy_KNeighbors": accuracy_KNeighbors_computer_factory
}
