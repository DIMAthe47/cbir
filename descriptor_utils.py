import numpy as np
import image_utils as iu
from keras.applications import VGG16
import keras_utils
from skimage.feature import greycomatrix, local_binary_pattern
import image_utils as iu
import sys

from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters
import ds_utils
from computer import Computer
from sklearn.preprocessing import normalize


def glcm_computer_factory(computer_func_params):
    kwargs = computer_func_params["library_func_kwargs"]

    def computer_(img_matrix):
        img_gray_matrix = iu.img_matrix_to_gray_img_matrix(img_matrix)
        img_gray_matrix = img_gray_matrix.squeeze()
        glcm = greycomatrix(img_gray_matrix, **kwargs)
        return glcm

    shape = [kwargs["levels"], kwargs["levels"], len(kwargs["distances"]), len(kwargs["angles"])]
    return Computer(computer_, shape)


def lbp_computer_factory(computer_func_params):
    def computer_(img_matrix):
        img_gray_matrix = iu.img_matrix_to_gray_img_matrix(img_matrix)
        img_gray_matrix = img_gray_matrix.squeeze()
        kwargs = computer_func_params["library_func_kwargs"]
        lbp = local_binary_pattern(img_gray_matrix, **kwargs)
        return lbp

    return Computer(computer_, None)


from sklearn.metrics import pairwise_distances

first_hist = None
first_img = None


def gray_histogram_computer_factory(computer_func_params):
    n_bins = computer_func_params["n_bins"]
    density = computer_func_params["density"]
    n_values = 256
    bin_size = n_values // n_bins
    edges = [i * bin_size + min((bin_size, n_values - 1 - i * bin_size)) for i in range(n_bins)]
    edges = [0] + edges

    def computer_(img_matrix):
        img_gray_matrix = iu.img_matrix_to_gray_img_matrix(img_matrix)
        # iu.img_matrix_to_pil_image(img_gray_matrix).show()
        kwargs = computer_func_params["library_func_kwargs"]

        histogram, edges_ = np.histogram(img_gray_matrix, bins=edges, density=density, **kwargs)

        # if first_hist is None:
        #     first_img = iu.img_matrix_to_pil_image(img_gray_matrix)
            # first_img.show()
        # else:
        #     dist = pairwise_distances(first_hist.reshape(1, -1), histogram.reshape(1, -1))
            # iu.img_matrix_to_pil_image(img_gray_matrix).show()
            # print(dist)
        return histogram

    shape = [n_bins]
    return Computer(computer_, shape)


def vgg16_computer_factory(computer_func_params):
    vgg16_model = VGG16()
    layer_name = computer_func_params["layer_name"]

    def computer_(img_matrix_chunks):
        # return None
        kwargs = computer_func_params["library_func_kwargs"]
        if img_matrix_chunks.shape[1] != 224 or img_matrix_chunks.shape[2] != 224:
            return np.zeros((len(img_matrix_chunks), 4096), dtype=float)
        activations = keras_utils.get_activations(vgg16_model, img_matrix_chunks, layer_name=layer_name, **kwargs)
        return activations[0]

    if layer_name == "fc1":
        shape = [4096]
    elif layer_name == "fc2":
        shape = [4096]
    return Computer(computer_, shape)


def pqcode_computer_factory(computer_func_params):
    cluster_centers_ds = computer_func_params["quantization_model"]["output_model"]
    cluster_centers = ds_utils.read_array(cluster_centers_ds)
    assert len(cluster_centers.shape) == 3
    pqquantizer = restore_from_clusters(cluster_centers)

    def computer_(descriptors_chunks):
        pqcodes = pqquantizer.predict_subspace_indices(descriptors_chunks)
        return pqcodes

    n_quantizers = cluster_centers.shape[0]
    shape = [n_quantizers]
    return Computer(computer_, shape)


def normalize_factory(computer_func_params):
    norm = computer_func_params["norm"]

    def computer_(X):
        return normalize(X, norm=norm)

    return Computer(computer_, None)


descriptor_type__computer_factory = {
    "pq": pqcode_computer_factory,
    "glcm": glcm_computer_factory,
    "histogram": gray_histogram_computer_factory,
    "lbp": lbp_computer_factory,
    "vgg16": vgg16_computer_factory,
    "pqcode": pqcode_computer_factory,
    "normalize": normalize_factory
}


# will be better not to pass here model
# sources can be read from model["input_ds"]
def compute_descriptors(model, sources_):
    computer_factory = descriptor_type__computer_factory[model["type"]]
    computer = computer_factory(model["kwargs"])
    n_sources_ = len(sources_)
    shape = computer.get_shape()
    if shape:
        desciptors = np.empty((n_sources_, *shape), model["dtype"])
    else:
        desciptors = [0] * n_sources_

    if "chunk_size" in model:
        chunk_from = 0
        chunk_to = model["chunk_size"]
        while chunk_from < n_sources_:
            if chunk_to >= n_sources_:
                chunk_to = n_sources_
            desciptors[chunk_from:chunk_to] = computer.compute(sources_[chunk_from:chunk_to])
            chunk_from = chunk_to
            chunk_to += model["chunk_size"]
    else:
        for i, source_ in enumerate(sources_):
            desciptors[i] = computer.compute(source_)

    if not shape:
        desciptors = np.array(desciptors)

    return desciptors
