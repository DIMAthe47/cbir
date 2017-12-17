import openslide
import rect_utils

from computer import Computer


def openslide_tiles_generator(slide_path, zoom_factor, shape, step_shape):
    slide = openslide.OpenSlide(slide_path)
    level = slide.get_best_level_for_downsample(zoom_factor)
    level_shape = slide.level_dimensions[level]
    x_max = level_shape[0]
    y_max = level_shape[1]
    x = 0
    y = 0
    x_step = step_shape[0]
    y_step = step_shape[1]
    # TODO stop when x>x_max, y>y_max
    while y < y_max:
        while x < x_max:
            if x + x_step < x_max and y + y_step < y_max:
                tile = slide.read_region((x, y), level, shape)
                yield tile
            x += x_step
        y += y_step
        x = 0
    del slide


def read_openslide_tile(slide: openslide.OpenSlide, downsample, tile_rect):
    level = slide.get_best_level_for_downsample(downsample)
    tile = slide.read_region((tile_rect[0], tile_rect[1]), level, (tile_rect[2], tile_rect[3]))
    return tile


def openslide_tiles_generator_factory(computer_func_params):
    zoom_factor = computer_func_params["zoom_factor"]
    shape = computer_func_params["shape"]
    step_shape = computer_func_params["step_shape"]

    # def computer_(slide):
    #     return openslide_tiles_generator(slide, zoom_factor, shape, step_shape)

    def computer_(slide_path):
        return openslide_tiles_generator(slide_path, zoom_factor, shape, step_shape)

    return Computer(computer_)


def openslide_tiles_generator2_factory(computer_func_params):
    # tiles_rects = computer_func_params["tiles_rects"]
    downsample = computer_func_params["downsample"]
    image_path = computer_func_params["image_model"]["string"]
    slide = openslide.OpenSlide(image_path)
    level = slide.get_best_level_for_downsample(downsample)

    def computer_(tile_rect):
        tile = read_openslide_tile(slide, level, tile_rect)
        return tile

    return Computer(computer_)


def rect_tiles_generator_factory(computer_func_params):
    rect_size = computer_func_params["rect_size"]
    tile_size = computer_func_params["tile_size"]
    tile_step = computer_func_params["tile_step"]

    def computer_():
        tile_generator = rect_utils.gen_slice_rect(rect_size, tile_size, tile_step)
        return tile_generator

    return Computer(computer_)


image_util_type__computer_factory = {
    "openslide_tiler": openslide_tiles_generator_factory,
    "openslide_tiler2": openslide_tiles_generator2_factory,
    "rect_tiles": rect_tiles_generator_factory,
    "tile": openslide_tiles_generator2_factory
}
