import numpy as np
from core.quantization.pq_quantizer import restore_from_clusters
from skimage.feature import greycomatrix, local_binary_pattern
from sklearn.preprocessing import normalize

import image_form_utils as iu
# from cbir_core.util import keras_utils
from cbir_core.util import ds_utils
from cbir_core.computer.computer import Computer


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


def gray_histogram_computer_factory(computer_func_params):
    n_bins = computer_func_params["n_bins"]
    density = computer_func_params["density"]
    n_values = 256
    bin_size = n_values // n_bins
    # edges = [i * bin_size for i in range(n_bins)]
    edges = np.arange(0, n_values - 1, bin_size)

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


"""
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
"""


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


descriptor_util__computer_factory = {
    "pq": pqcode_computer_factory,
    "glcm": glcm_computer_factory,
    "histogram": gray_histogram_computer_factory,
    "lbp": lbp_computer_factory,
    # "vgg16": vgg16_computer_factory,
    "pqcode": pqcode_computer_factory,
    "normalize": normalize_factory
}
