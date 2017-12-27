from cbir_core.util.keras_utils import get_activations


def activations_factory(model, layer_name=None, print_shape_only=True):
    def get_activations_(X):
        return get_activations(model, X, layer_name, print_shape_only)
    return get_activations_