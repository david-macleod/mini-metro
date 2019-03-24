from abc import ABC, abstractmethod
import json
from enum import Enum

import tensorflow as tf
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from skimage.color import rgb2gray 


class ObjectDetector(ABC):

    @abstractmethod
    def predict(self, image_array):
        pass

    @abstractmethod
    def _load_model(self, model_path):
        pass


class FastaiObjectDetector(ObjectDetector):

    def predict(self, image_array):
        pass

    def _load_model(self, model_path):
        pass


class PytorchObjectDetector(ObjectDetector):

    def predict(self, image_array):
        pass

    def _load_model(self, model_path):
        pass


class TensorflowObjectDetector(ObjectDetector):

    def __init__(self, model_path, category_labels_path):
        """
        :param model_path: string location of serialized frozen model
        :param category_labels_path: string location of class labels json
        """
        self.graph = self._load_model(str(model_path))
        self.category_labels  = self._load_category_labels(str(category_labels_path))
        # Creating shared session to get performance benefits of cached graph
        self.session = tf.Session(graph=self.graph)


    def predict(self, image_array):
        """
        Run object detection inference on a a single imgage

        :param image_array: numpy array with shape [?,?,3]
        :returns: dict containing lists of bounding box co-ordinates (ymin, xmin, ymax, xmax),
            scores (class probabilities) and class ids (integers)
        """
        assert image_array.ndim == 3 and image_array.shape[2] == 3, 'Input array must have shape [?,?,3]'
  
        # Get handles to input and output tensors
        ops = self.graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.graph.get_tensor_by_name(tensor_name)
                
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Create feed dict in format required by graph
        feed_dict={image_tensor: np.expand_dims(image_array, 0)}
        # Run inference
        output_dict = self.session.run(tensor_dict, feed_dict=feed_dict)

        # Returning boxes as dicts to avoid ambiguous list of coords
        raw_boxes = output_dict.pop('detection_boxes')[0]
        output_dict['boxes'] = self.map_boxes(raw_boxes)
        output_dict['scores'] = output_dict.pop('detection_scores')[0]
        output_dict['category_ids'] = output_dict.pop('detection_classes')[0].astype(np.uint8)

        return output_dict


    def _load_category_labels(self, labels_path):
        """
        Load class labels from json ("category_label": category_id) and convert to Enum mapping

        :returns: Enum object with containing label/id mappings for two-way lookup
        """
        with open(labels_path, 'rb') as json_file:
            labels_dict = json.load(json_file)

        return Enum('Labels', labels_dict)


    def _load_model(self, model_path):
        """
        Load serialized frozen graph model

        :param model_path: path to frozen graph
        :returns: Tensorflow Graph
        """
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            # Create GraphDef object to parse serialized frozen graph
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                # Import graph definition into default graph
                tf.import_graph_def(graph_def, name='')

                return detection_graph


    def draw_bounding_boxes(self, image_array, boxes, scores, category_ids, threshold=0.2, **draw_kwargs):
        """
        Draw all bounding boxes for a single image

        :param image_array: numpy array with shape [?,?,3]
        :param boxes: list of list of (ymin, xmin, ymax, xmax) box co-ordinates
        :param scores: list of class probabilities for boxes
        :param category_ids: list of class ids for boxes
        :returns: numpy array with all boxes annootated
        """

        for box, score, category_id in zip(boxes, scores, category_ids):
            if score >= threshold:
                #TODO add class labels to box
                category_label = self.category_labels(category_id).name
                label = f'{score:.0%}'
         
                image_array = draw_bounding_box(image_array=image_array, label=label,
                                                **box, **draw_kwargs)
            else:
                continue

        return image_array

    @staticmethod
    def map_boxes(boxes):
        ''' Convert list of box coords in TF format to dict '''
        return [dict(zip(['ymin', 'xmin', 'ymax', 'xmax'], box)) for box in boxes]


def object_detector(model_path, model_type, **kwargs):
    ''' Factory method for ObjectDetector '''

    if model_type == 'fastai':
        return FastaiObjectDetector(model_path, **kwargs)

    elif model_type == 'pytorch':
        return PytorchObjectDetector(model_path, **kwargs)

    elif model_type == 'tensorflow':
        return TensorflowObjectDetector(model_path, **kwargs)


def draw_bounding_box(image_array, ymin, xmin, ymax, xmax, label='',
                      thickness=4, color='lime', normalized_coords=True):
    """
    Draw a single bounding box on an image array

    :param image_array: numpy array with shape [?,?,3]
    :param ymin, xmin, ymax, xmax: coordinates of bounding box limits
    :param label: bounding box text
    :param thickness: number of pixels to used for box border width
    :param color: string specificying box colour
    :param normalized_coordas: boolean flags if coordinates are normalized (default) or absolute pixel values
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
    font = ImageFont.truetype('arialbd.ttf', size)
    textcolor = "white"
    bordercolor = "black"
    text_x, text_y = xmin, ymin - size

    draw.text((text_x-1, text_y-1), label, font=font, fill=bordercolor)
    draw.text((text_x+1, text_y-1), label, font=font, fill=bordercolor)
    draw.text((text_x-1, text_y+1), label, font=font, fill=bordercolor)
    draw.text((text_x+1, text_y+1), label, font=font, fill=bordercolor)
    draw.text((text_x, text_y), label, font=font, fill=textcolor)

    return np.array(image_pil)


def rgb2gray3d(image_array):
    ''' Convert RGB to grayscale whilst preserving 3 channels '''
    # Convert to grayscale 2d
    image_array_gray_2d = rgb2gray(image_array)
    # Add 3rd dimension and repeat along new axis 
    image_array_gray_3d = np.repeat(image_array_gray_2d[..., None], 3, axis=2)

    return image_array_gray_3d