from cbir_core.util import ds_utils
from cbir_core.util.classificatiton_utils import knn_compute_classes
from cbir_core.computer.computer import Computer


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