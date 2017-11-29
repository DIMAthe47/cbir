import io

import numpy as np
from PIL import Image

# def gen_img_tiles(img_or_bytes_or_path, height, width, print_boxes=False):
#    img_matrix=img_to_numpy_array(img_or_bytes_or_path)
#    img=img_matrix_to_pil_image(img_matrix)
#    imgwidth, imgheight = img.size
#    for i in range(0,imgheight,height):
#        for j in range(0,imgwidth,width):
#            box = (j, i, j+width, i+height)
#            if box[2]<=imgwidth and box[3]<=imgheight:
#                if print_boxes:
#            	    print(box)
#                a = img.crop(box)
#                yield a
from factory_utils import factorify_as_computer


def tiles(pilimg, height, width, print_boxes=False):
    #    img_matrix=img_to_numpy_array(img_or_bytes_or_path)
    #    img=img_matrix_to_pil_image(img_matrix)
    imgwidth, imgheight = pilimg.size
    cols = imgwidth // width
    rows = imgheight // height
    n_tiles = cols * rows
    tiles = []
    #   tiles=np.empty((n_tiles, width, height, len(img.getbands())), dtype='uint8')
    for i in range(0, rows):
        for j in range(0, cols):
            box = (j * width, i * height, j * width + width, i * height + height)
            if box[2] <= imgwidth and box[3] <= imgheight:
                if print_boxes:
                    print(box)
                # tiles[i*cols+j]=img.crop(box)
                tiles.append(pilimg.crop(box))
                #    print(n_tiles)
    return tiles


# def img_to_numpy_array(img_or_bytes_or_path):
#    if isinstance(img_or_bytes_or_path, str):
#        img = Image.open(img_or_bytes_or_path)
#    elif isinstance(img_or_bytes_or_path, bytes):
#        img=Image.open(io.BytesIO(img_or_bytes_or_path))
#    elif isinstance(img_or_bytes_or_path, PIL.Image.Image):
#        img=img_or_bytes_or_path
#    img_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
#    img_arr = img_arr.reshape((img.size[1], img.size[0], len(img.getbands())))
#    return img_arr

def img_matrix_to_gray_img_matrix(img_matrix):
    pilimg_gray = img_matrix_to_pil_image(img_matrix, grayscale=True)
    img_gray_matrix = pilimage_to_matrix(pilimg_gray)
    return img_gray_matrix


def img_matrix_to_pil_image(img_matrix, grayscale=False):
    img_matrix = img_matrix.squeeze()
    img = Image.fromarray(img_matrix)
    if grayscale:
        img = img.convert('L')
    return img


def path_to_pilimage(path_):
    return Image.open(path_)


def pilimage_to_matrix(pilimage):
    img_matrix = np.fromstring(pilimage.tobytes(), dtype=np.uint8)
    img_matrix = img_matrix.reshape((pilimage.size[1], pilimage.size[0], len(pilimage.getbands())))
    return img_matrix


def path_to_matrix(path_):
    return pilimage_to_matrix(path_to_pilimage(path_))


def pilimg_to_jpeg(pilimg):
    b = io.BytesIO()
    pilimg.save(b, format="jpeg")
    b.seek(0)
    return b.read()


def jpeg_to_pilimg(jpeg_):
    return Image.open(io.BytesIO(jpeg_))


def jpeg_to_matrix(jpeg_):
    return pilimage_to_matrix(jpeg_to_pilimg(jpeg_))


image_transform_type__computer_factory = {
    "jpeg_to_matrix": factorify_as_computer(jpeg_to_matrix),
    "pilimage_to_matrix": factorify_as_computer(pilimage_to_matrix)
}
