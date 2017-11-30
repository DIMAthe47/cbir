import sys, json
import numpy as np
from ds_utils import add_output_model
import ds_utils


def generate_tiles_rects_model(input_model, downsample, tile_shape, tile_step, db_path=None, ds_name=None, dtype="int",
                               **kwargs):
    name_ = "tiles_rects_{}_{}_{}_{}_{}__{}".format(downsample, *tile_shape, *tile_step, input_model["name"])
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "tiles_rects",
        "computer_func_params": {
            "downsample": downsample,
            "tile_shape": tile_shape,
            "tile_step": tile_step,
            "library_func_kwargs": {
                **kwargs
            }
        },
        "dtype": dtype,
        # image_model
        "input_model": input_model,
    }
    add_output_model(model, db_path, ds_name)
    return model


def generate_tiling_model2(input_model, image_path, downsample, db_path=None, ds_name=None, dtype="int",
                           **kwargs):
    name_ = "tiling__{}".format(input_model["name"])
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "openslide_tiler2",
        "computer_func_params": {
            "image_path": image_path,
            "downsample": downsample,
            # "tiles_rects_model": tiles_rects_model,
            "library_func_kwargs": {
                **kwargs
            }
        },
        "dtype": dtype,
        # tiles_rects как output у tiles_rects_model либо просто inmemory
        "input_model": input_model
    }
    add_output_model(model, db_path, ds_name)
    return model


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
