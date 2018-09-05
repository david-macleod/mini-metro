import tensorflow as tf
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import json
from enum import Enum

class ObjectDetector(object):

    def __init__(self, frozen_graph_path, class_labels_path):

        self.graph = self._load_model(frozen_graph_path)
        self.class_labels  = self._load_class_labels(class_labels_path) 
        # Creating shared session to get performance benefits of cached graph
        self.session = tf.Session(graph=self.graph)


    def run_inference(self, image):

        assert image.ndim == 3 and image.shape[2] == 3, 'Input array must have shape [n,n,3]'
  
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
        feed_dict={image_tensor: np.expand_dims(image, 0)}
        # Run inference
        output_dict = self.session.run(tensor_dict, feed_dict=feed_dict)

        # All outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['boxes'] = output_dict.pop('detection_boxes')[0]
        output_dict['scores'] = output_dict.pop('detection_scores')[0]
        output_dict['class_ids'] = output_dict.pop('detection_classes')[0].astype(np.uint8)

        return output_dict


    def _load_class_labels(self, labels_path):

        with open(labels_path, 'rb') as json_file:
            labels_dict = json.load(json_file)

        return Enum('Labels', labels_dict)


    def _load_model(self, frozen_graph_path):

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            # Create GraphDef object to parse serialized frozen graph
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                # Import graph definition into default graph
                tf.import_graph_def(graph_def, name='')

                return detection_graph


    def draw_bounding_boxes(self, image_array, boxes, scores, class_ids, threshold=0.2, **draw_kwargs):
    
        assert boxes.ndim == 2 and boxes.shape[1] == 4, 'Bounding boxes must have shape [n,4]'

        for box, score, class_id in zip(boxes, scores, class_ids):
            if score >= threshold:
                class_label = self.class_labels(class_id).name
                label = '{}: {:.2f}'.format(threshold, score)
                coord_kwargs = dict(zip(['ymin', 'xmin', 'ymax', 'xmax'], box))
         
                image_array = draw_bounding_box(image_array=image_array, **coord_kwargs, **draw_kwargs)
            else:
                continue

        return image_array
   

def draw_bounding_box(image_array, ymin, xmin, ymax, xmax, thickness=4, color='lime', normalized_coords=True):
    """
    Draw a single bounding box on an image array

    :param image_array: numpy array with shape [n,n,3]
    :param ymin, xmin, ymax, xmax: coordinates of bounding box limits
    :param thickness: number of pixels to used for box border width
    :param color: string specificying box colour
    :param normalized_coordas: boolean flags if coordinates are normalized (default) or absolute pixel values
    :returns: numpy array with shape [n,n,3]
    """
    assert image_array.ndim == 3 and image_array.shape[2] == 3, 'Image must have shape [n,n,3]'

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

    return np.array(image_pil)