import openslide


def read_openslide_tile(slide: openslide.OpenSlide, downsample, tile_rect):
    level = slide.get_best_level_for_downsample(downsample)
    tile = slide.read_region((tile_rect[0], tile_rect[1]), level, (tile_rect[2], tile_rect[3]))
    return tile
