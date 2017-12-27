import openslide

from cbir_core.util.openslide_utils import read_openslide_tile
from cbir_core.computer.computer import Computer


def openslide_tiles_generator_factory(computer_func_params):
    # tiles_rects = computer_func_params["tiles_rects"]
    downsample = computer_func_params["downsample"]
    image_path = computer_func_params["image_model"]["string"]
    slide = openslide.OpenSlide(image_path)
    level = slide.get_best_level_for_downsample(downsample)

    def computer_(tile_rect):
        tile = read_openslide_tile(slide, level, tile_rect)
        return tile

    return Computer(computer_)


openslide_util__computer_factory = {
    "openslide_tiler": openslide_tiles_generator_factory,
}