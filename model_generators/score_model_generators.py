import sys, json
import numpy as np
from ds_utils import add_output_model


def generate_accuracy_model(input_model, true_classes_model, db_path=None, ds_name=None, dtype="float",
                            **kwargs):
    name_ = "accuracy__{}".format(input_model["name"])
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "accuracy",
        "computer_func_params": {
            "true_classes_model": true_classes_model,
            "library_func_kwargs": {
                **kwargs
            }
        },
        "dtype": dtype,
        "chunk_size": -1,
        "input_model": input_model,  # classification_model
    }
    add_output_model(model, db_path, ds_name)
    return model


def generate_KNeighbors_accuracy_model(input_model, base_descriptors_model, true_classes_model, n_nearest, db_path=None,
                                       ds_name=None, dtype="float",
                                       **kwargs):
    """
        compute accuracy directly from descriptors through KNeighbors
        metric like ADC will be unoptimized
    """
    name_ = "accuracy_KNeighbors_{}_{}".format(n_nearest, base_descriptors_model["name"])
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "accuracy_KNeighbors",
        "computer_func_params": {
            "base_model": base_descriptors_model,
            "true_classes_model": true_classes_model,
            "n_nearest": n_nearest,
            "library_func_kwargs": {
                **kwargs
            }
        },
        "dtype": dtype,
        "chunk_size": -1,
        "input_model": input_model,  # descriptors_model
    }
    add_output_model(model, db_path, ds_name)
    return model


if __name__ == '__main__':
    pass
