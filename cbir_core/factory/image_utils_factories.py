import openslide

from cbir_core.computer.computer import Computer
from factory_utils import factorify_as_computer
from image_form_utils import jpeg_to_matrix, pilimage_to_matrix, pure_pil_alpha_to_color_v2
from tiling_utils import generate_tiles_rects


def tiles_rects_computer_factory(computer_func_params):
    tile_shape = computer_func_params["tile_shape"]
    tile_step = computer_func_params["tile_step"]
    downsample = computer_func_params["downsample"]
    image_path = computer_func_params["image_path"]

    # def computer_(image_path):
    def computer_():
        img = openslide.OpenSlide(image_path)
        level = img.get_best_level_for_downsample(downsample)
        img_shape = img.level_dimensions[level]
        tiles_rects = generate_tiles_rects(img_shape, tile_shape, tile_step)
        return tiles_rects

    return Computer(computer_, None)


image_transform_type__computer_factory = {
    "jpeg_to_matrix": factorify_as_computer(jpeg_to_matrix),
    "pilimage_to_matrix": factorify_as_computer(pilimage_to_matrix),
    "tiles_rects": tiles_rects_computer_factory,
    "rgbapilimage_to_rgbpilimage": factorify_as_computer(pure_pil_alpha_to_color_v2)
}