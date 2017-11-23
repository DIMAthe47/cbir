import sys, json
import numpy as np
from ds_utils import add_output_model


def generate_knn_model(input_model, true_classes_model, db_path=None, ds_name=None, dtype="float",
                                   **kwargs):
    name_ = "knn__{}".format(input_model["name"])
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "knn",
        "computer_func_params": {
            "true_classes_model": true_classes_model,
            "library_func_kwargs": {
                **kwargs
            }
        },
        "dtype": dtype,
        "chunk_size": -1,
        "input_model": input_model,  # nearest_indices_model
    }
    add_output_model(model, db_path, ds_name)
    return model


if __name__ == '__main__':
    pass
