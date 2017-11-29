import sys, json
import numpy as np
from ds_utils import add_output_model
import ds_utils


def generate_tiling_model(input_model, zoom_factor, shape, step_shape, db_path=None, ds_name=None, dtype="int",
                          **kwargs):
    name_ = "tiling_{}_{}_{}_{}_{}__{}".format(zoom_factor, *shape, *step_shape, input_model["name"])
    model = {
        "type": "computer", 
        "name": name_,
        "computer_func_name": "openslide_tiler",
        "computer_func_params": {
            "zoom_factor": zoom_factor,
            "shape": shape,
            "step_shape": step_shape,
            "library_func_kwargs": {
                **kwargs
            }
        },
        "dtype": dtype,
        "chunk_size": -1,
        # image_model
        "input_model": input_model,
    }
    add_output_model(model, db_path, ds_name)
    return model


if __name__ == '__main__':
    pass
