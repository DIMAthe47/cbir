import openslide

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


def openslide_tiles_generator_factory(computer_func_params):
    zoom_factor = computer_func_params["zoom_factor"]
    shape = computer_func_params["shape"]
    step_shape = computer_func_params["step_shape"]

    # def computer_(slide):
    #     return openslide_tiles_generator(slide, zoom_factor, shape, step_shape)

    def computer_(slide_path):
        return openslide_tiles_generator(slide_path, zoom_factor, shape, step_shape)

    return Computer(computer_)


image_util_type__computer_factory = {
    "openslide_tiler": openslide_tiles_generator_factory
}
