import sys, json
import numpy as np
from ds_utils import add_output_model
import ds_utils

def generate_tiling_model(input_model, base_model, zoom, shape, db_path=None, ds_name=None, dtype="int",
                                   **kwargs):
    name_ = "tiling_{}__{}".format(metric, base_model["name"])
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": metric,
        "computer_func_params": {
            "zoom": zoom,
            "shape": shape,
            "library_func_kwargs": {
                **kwargs
            }
        },
        "dtype": dtype,
        "chunk_size": -1,
        # computer assumes input as
        "input_model": input_model,
    }
    add_output_model(model, db_path, ds_name)
    return model


if __name__ == '__main__':
    pass
