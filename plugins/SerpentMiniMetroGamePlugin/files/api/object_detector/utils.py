from PIL import Image, ImageColor, ImageDraw, ImageFont
from skimage.color import rgb2gray
import numpy as np

def draw_bounding_box(image_array, ymin, xmin, ymax, xmax, label='',
                      thickness=4, color='lime', normalized_coords=True):
    """
    Draw a single bounding box on an image array

    :param image_array: numpy array with shape [?,?,3]
    :param ymin, xmin, ymax, xmax: coordinates of bounding box limits
    :param label: bounding box text
    :param thickness: number of pixels to used for box border width
    :param color: string specificying box colour
    :param normalized_coords: boolean flags if coordinates are normalized (default) or absolute pixel values
    :returns: numpy array with shape [n,n,3]
    """
    assert image_array.ndim == 3 and image_array.shape[2] == 3, 'Image must have shape [?,?,3]'

    image_pil = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image_pil)

    image_height, image_width = image_array.shape[:2]
    
    if normalized_coords: 
        xmin *= image_width
        xmax *= image_width
        ymin *= image_height
        ymax *= image_height

    line_coords = [(xmin, ymax), (xmin, ymin), (xmax, ymin),(xmax, ymax), (xmin, ymax)]
    
    draw.line(line_coords, width=thickness, fill=color)

    # Add text with border
    size = 14
    textcolor = "white"
    bordercolor = "black"
    text_x, text_y = xmin, ymin - size

    draw.text((text_x-1, text_y-1), label, fill=bordercolor)
    draw.text((text_x+1, text_y-1), label, fill=bordercolor)
    draw.text((text_x-1, text_y+1), label, fill=bordercolor)
    draw.text((text_x+1, text_y+1), label, fill=bordercolor)
    draw.text((text_x, text_y), label, fill=textcolor)

    return np.array(image_pil)


def rgb2gray3d(image_array):
    ''' Convert RGB to grayscale whilst preserving 3 channels '''
    # Convert to grayscale 2d
    image_array_gray_2d = rgb2gray(image_array)
    # Add 3rd dimension and repeat along new axis 
    image_array_gray_3d = np.repeat(image_array_gray_2d[..., None], 3, axis=2)

    return image_array_gray_3d