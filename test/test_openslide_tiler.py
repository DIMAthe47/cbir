import unittest
from itertools import count

from computer_utils import compute_model
from model_generators.tiling_model_generators import generate_tiling_model


class OpensliderTest(unittest.TestCase):
    def test_openslided_tiler(self):
        # openslide_image_model = {
        #     "type": "slide_image",
        #     "image_path": "/home/dimathe47/Downloads/CMU-1-Small-Region.svs",
        #     "name": "CMU-1-Small-Region.svs"
        # }
        openslide_image_model = {
            "type": "string",
            "string": "/home/dimathe47/Downloads/CMU-1-Small-Region.svs",
            "name": "CMU-1-Small-Region.svs"
        }
        tiling_model = generate_tiling_model(openslide_image_model, 0, (500, 500), (500, 500))
        output_ = compute_model(tiling_model)
        # k = sum(1 for _ in output_)
        # print(output_, k)
        tiles = list(output_)
        n_tiles = len(tiles)
        print(n_tiles)

        tiles[0].show()
        tiles[1].show()
        tiles[2].show()
        tiles[3].show()
        tiles[4].show()



if __name__ == '__main__':
    unittest.main()
