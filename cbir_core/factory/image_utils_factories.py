from cbir_core.factory.factory_utils import factorify_as_computer
from image_form_utils import jpeg_to_matrix, pilimage_to_matrix, pure_pil_alpha_to_color_v2

image_transform_type__computer_factory = {
    "jpeg_to_matrix": factorify_as_computer(jpeg_to_matrix),
    "pilimage_to_matrix": factorify_as_computer(pilimage_to_matrix),
    "rgbapilimage_to_rgbpilimage": factorify_as_computer(pure_pil_alpha_to_color_v2)
}
