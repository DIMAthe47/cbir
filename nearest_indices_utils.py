import numpy as np
import sys

sys.path.append('/home/dimathe47/PycharmProjects/pq')
import ds_utils
from datetime import datetime
from computer import Computer


def argsort_nearest_indices_computer_factory(computer_func_params):
    def computer_(distance_matrix):
        assert len(distance_matrix.shape) == 2
        n_nearest = computer_func_params["n_nearest"]
        if n_nearest == -1:
            n_nearest = distance_matrix.shape[1]
        nearest_indices = np.argpartition(distance_matrix, axis=1, kth=n_nearest - 1)[:, :n_nearest]
        rows_range = np.arange(len(distance_matrix)).reshape((-1, 1))
        nearest_indices = nearest_indices[rows_range, np.argsort(distance_matrix[rows_range, nearest_indices])]
        return nearest_indices

    return Computer(computer_)


nearest_indices_type__computer_factory = {
    "argsort": argsort_nearest_indices_computer_factory,
}


def compute_nearest_indices(nearest_indices_model, verbose=1):
    start_datetime = datetime.now()
    distance_matrix = ds_utils.read_array(nearest_indices_model["input_ds"])
    computer_factory = nearest_indices_type__computer_factory[nearest_indices_model["type"]]
    computer_ = computer_factory(nearest_indices_model["kwargs"])
    nearest_indices = computer_.compute(distance_matrix)
    ds_utils.save_array(nearest_indices, nearest_indices_model["output_ds"], attrs={"model": nearest_indices_model})
    datetime_delta = datetime.now() - start_datetime

    if verbose >= 1:
        print("model computed: {} in {} seconds".format(nearest_indices_model["name"], datetime_delta.total_seconds()))
