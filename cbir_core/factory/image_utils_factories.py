from cbir_core.computer.computer import Computer
from cbir_core.factory.factory_utils import factorify_as_computer, factorify_as_computer_kw
from image_form_utils import jpeg_to_matrix, pilimage_to_matrix, pure_pil_alpha_to_color_v2, resize_pilimg


# def resizepilimage_factory(computer_func_params):
#     size = computer_func_params["size"]
#
#     def computer_(X):
#         return resize_pilimg(X, size=size)
#
#     return Computer(computer_, None)


image_transform_type__computer_factory = {
    "jpeg_to_matrix": factorify_as_computer(jpeg_to_matrix),
    "pilimage_to_matrix": factorify_as_computer(pilimage_to_matrix),
    "rgbapilimage_to_rgbpilimage": factorify_as_computer(pure_pil_alpha_to_color_v2),
    "pilimage_to_resizedpilimage": factorify_as_computer_kw(resize_pilimg)
}
