import unittest
from itertools import count

import numpy as np
from classificatiton_utils import knn_compute_classes
from computer_utils import compute_model
from model_generators import generate_histogram_model, generate_image_model
from model_generators.image_transform_model_generators import generate_pilimage_to_matrix_model
from model_generators.tiling_model_generators import generate_tiling_model


class OpensliderTest(unittest.TestCase):
    def test_openslided_tiler(self):
        # openslide_image_model = {
        #     "type": "slide_image",
        #     "image_path": "/home/dimathe47/Downloads/CMU-1-Small-Region.svs",
        #     "name": "CMU-1-Small-Region.svs"
        # }
        image_path = "/home/dimathe47/Downloads/CMU-1-Small-Region.svs"
        image_model = generate_image_model(image_path)
        zoom_factor = 1
        shape = (224, 224)
        step_shape = (224, 224)
        tiling_model = generate_tiling_model(image_model, zoom_factor, shape, step_shape)
        n_bins = 128
        pilimage_to_matrix_model = generate_pilimage_to_matrix_model(tiling_model)
        histogram_model = generate_histogram_model(pilimage_to_matrix_model, n_bins, "temp/histograms.hdf5")
        compute_model(histogram_model, verbose=2)


if __name__ == '__main__':
    unittest.main()
