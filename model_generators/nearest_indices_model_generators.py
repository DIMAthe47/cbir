import sys, json
import numpy as np
from ds_utils import add_output_model


def generate_nearest_indices_model(input_model, n_nearest=-1, db_path=None, ds_name=None, dtype="float", **kwargs):
    name_ = "argsort__{}".format(input_model["name"])
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "argsort",
        "computer_func_params": {
            "n_nearest": n_nearest,
            "library_func_kwargs": {
                **kwargs
            }
        },
        "dtype": dtype,
        "chunk_size": -1,
        "input_model": input_model,
    }
    add_output_model(model, db_path, ds_name)
    return model


if __name__ == '__main__':
    pass
