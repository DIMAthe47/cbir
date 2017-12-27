import unittest

from cbir_core.computer.computer_utils import compute_model
from model_generators.tiling_model_generators import generate_tiling_model2


class OpensliderTest(unittest.TestCase):
    def test_openslided_tiler(self):
        # openslide_image_model = {
        #     "type": "slide_image",
        #     "image_path": "/home/dimathe47/Downloads/CMU-1-Small-Region.svs",
        #     "name": "CMU-1-Small-Region.svs"
        # }
        openslide_image_model = {
            "type": "string",
            # "string": "/home/dimathe47/Downloads/CMU-1-Small-Region.svs",
            "string": r"C:\Users\DIMA\Downloads\CMU-1-Small-Region.svs",
            "name": "CMU-1-Small-Region.svs"
        }
        rects_model = {
            "type": "list",
            "list": [(0, 0, 500, 500),
                     (100, 100, 500, 500),
                     ],
            "name": "rects"
        }
        tiling_model = generate_tiling_model2(rects_model, openslide_image_model, 1)
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
