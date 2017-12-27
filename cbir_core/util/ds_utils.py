import numpy as np
import contextlib
import h5py
import json
import file_utils

from functools import singledispatch

from PIL import Image


@singledispatch
def to_json(val):
    """Used by default."""
    return json.dumps(val, indent=4)


@to_json.register(Image.Image)
def pilimg_to_json(val):
    """Used if *val* is an instance of PIL.Image."""
    return "TODO"


class DSNotFoundError(Exception):
    def __init__(self, db_path, ds_name):
        self.db_path = db_path
        self.ds_name = ds_name

    def __str__(self):
        return "No data source for ds_name={}".format(self.ds_name)


def save_array(arr, db_path_or_source, ds_name=None, attrs=None):
    if not isinstance(db_path_or_source, str):
        db_path = db_path_or_source["db_path"]
        ds_name = db_path_or_source["ds_name"]
    else:
        db_path = db_path_or_source

    file_utils.make_if_not_exists(db_path)
    with open_h5py(db_path) as f:
        if ds_name in f:
            del f[ds_name]
        ds = f.create_dataset(ds_name, data=arr)
        #        print(len(attrs))
        for attr in ds.attrs:
            del ds.attrs[attr]

        for attr in attrs or []:
            # print(attr, attrs[attr])
            if not isinstance(attrs[attr], np.ndarray):
                ds.attrs[attr] = json.dumps(attrs[attr], indent=4, default=to_json)
            else:
                ds.attrs[attr] = attrs[attr]
        ds.attrs["shape"] = [int(x) for x in arr.shape]
        ds.attrs["dtype"] = arr.dtype.name
        f.flush()


def read_array(db_path_or_source, ds_name=None):
    if not isinstance(db_path_or_source, str):
        db_path = db_path_or_source["db_path"]
        ds_name = db_path_or_source["ds_name"]
    else:
        db_path = db_path_or_source

    with open_h5py(db_path) as f:
        if ds_name not in f:
            raise DSNotFoundError(db_path, ds_name)
        ds = f[ds_name]
        arr = np.empty(ds.shape, ds.dtype)
        ds.read_direct(arr)
        return arr


def read_attrs(db_path_or_source, ds_name=None):
    if not isinstance(db_path_or_source, str):
        db_path = db_path_or_source["db_path"]
        ds_name = db_path_or_source["ds_name"]
    else:
        db_path = db_path_or_source

    with open_h5py(db_path) as f:
        if ds_name not in f:
            raise DSNotFoundError(db_path, ds_name)
        ds = f[ds_name]
        #        attr=ds.attrs[attr_name]

        attrs = {}
        for attr in ds.attrs:
            #            print(attr, ds.attrs[attr], type(ds.attrs[attr]))
            if attr in ["dtype", "shape"]:
                attrs[attr] = ds.attrs[attr]
            elif not isinstance(ds.attrs[attr], np.ndarray):
                attrs[attr] = json.loads(ds.attrs[attr])
            else:
                attrs[attr] = ds.attrs[attr]

        return attrs


@contextlib.contextmanager
def open_h5py(db_path):
    f = h5py.File(db_path, 'a', libver='latest')
    yield f
    f.close()
