from abc import ABC, abstractmethod
import json
from enum import Enum

import torch
import numpy as np

from ptyolov3.models import Darknet
from ptyolov3.utils.utils import process_detections
from ptyolov3.utils.datasets import image_tensor_from_array

from .utils import draw_bounding_box


class PytorchObjectDetector:

    def __init__(self, model_path, config_path, labels_path, img_size):
        """
        :param model_path: location of serialised state_dict
        :param config_path: location of model config (darknet format)
        :param labels_path: location of category labels mapping
        :param img_size: image size used in input layer to model
        """
        self.model_path = model_path
        self.config_path = config_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.model = self._load_model()

    def predict(self, image_array, conf_threshold, nms_threshold):
        """
        Run object detection inference on a a single image

        :param image_array: numpy array with shape [?,?,3]
        :param conf_threshold: object confidence probability (P(object), not P(class|object))
        :param nms_threshold: iou threshold used in non-maximum supression
        :returns: dict containing lists of bounding box co-ordinates (ymin, xmin, ymax, xmax),
            object_conf (object probabilities), scores (class probabilities) and class ids (integers)
        """
        image_tensor = image_tensor_from_array(image_array, self.img_size)

        with torch.no_grad():
            model_output = self.model(image_tensor)
           
        detections = process_detections(
            raw_detections=model_output,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            img_size=self.img_size,
            orig_img_shape=image_array.shape[:2]
        ).numpy()

        output_dict = {}
        output_dict['boxes'] = self.map_boxes(detections[:, :4])
        output_dict['object_conf'] = detections[:, 4]
        output_dict['scores'] = detections[:, 5]
        output_dict['category_ids'] = detections[:, 6].astype(np.uint8)

        return output_dict

    def _load_model(self):
        ''' Deserialise state_dict and insantiate model '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Darknet(self.config_path, self.img_size)
        model.load_state_dict(torch.load(self.model_path, map_location=device))

        return model

    @staticmethod
    def map_boxes(boxes):
        ''' Convert list of box coords in PT format to dict '''
        return [dict(zip(['xmin', 'ymin', 'xmax', 'ymax'], box)) for box in boxes]

    def draw_bounding_boxes(self, image_array, boxes, object_conf, scores, category_ids, **draw_kwargs):
        """
        Draw all bounding boxes for a single image

        :param image_array: numpy array with shape [?,?,3]
        :param boxes: list of list of (ymin, xmin, ymax, xmax) box co-ordinates
        :param scores: list of class probabilities for boxes
        :param category_ids: list of class ids for boxes
        :returns: numpy array with all boxes annootated
        """

        for box, object_conf, score, category_id in zip(boxes, object_conf, scores, category_ids):
            
            label = f'{object_conf:.0%}'
            image_array = draw_bounding_box(image_array=image_array, label=label,
                                            **box, **draw_kwargs)

        return image_array