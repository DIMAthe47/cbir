import numpy as np

from cbir_core.computer.computer import Computer


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