from cbir_core.computer.computer import Computer
from tiling_utils import generate_tiles_rects


def rect_tiles_generator_factory(computer_func_params):
    rect_size = computer_func_params["rect_size"]
    tile_size = computer_func_params["tile_size"]
    tile_step = computer_func_params["tile_step"]

    def computer_():
        tile_generator = generate_tiles_rects(rect_size, tile_size, tile_step, True)
        return tile_generator

    return Computer(computer_)


tiling_util__computer_factory = {
    "rect_tiles": rect_tiles_generator_factory,
}
