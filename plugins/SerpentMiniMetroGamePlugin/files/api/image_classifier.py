from abc import ABC, abstractmethod
from pathlib import Path

from fastai.vision import load_learner, Image
import torch

class ImageClassifier(ABC):

    @abstractmethod
    def predict(self, image_array):
        pass

    @abstractmethod
    def _load_model(self, model_path):
        pass


class FastaiImageClassifier(ImageClassifier):

    def __init__(self, model_path):
        self.model = self._load_model(model_path)

    def predict(self, image_array):
        '''
        Run inference on a a single image 

        :param image_array: numpy array with shape [?,?,3]
        :returns: dict containing category, category_id and scores
        '''
        # convert to (C,H,W) tensor, scale between 0-1 as per fastai.open_image
        image_tensor = torch.from_numpy(image_array).permute([2, 0, 1]).float().div_(255)

        # create Image object (this is expected by predict)
        image = Image(image_tensor)
        
        category, category_id, scores = self.model.predict(image)
        
        output_dict = {
            'category': category.obj, # fastai.core.Category to str
            'category_id': category_id.item(), # torch.Tensor to int
            'scores': scores.tolist() # torch.Tensor to List[float]
        }

        return output_dict

    @staticmethod
    def _load_model(model_path):
        model_path = Path(model_path)
        return load_learner(model_path.parent, model_path.name)


class PytorchImageClassifier(ABC):

    def predict(self, image_array):
        pass

    def _load_model(self, model_path):
        pass


class TensorflowImageClassifier(ABC):

    def predict(self, image_array):
        pass

    def _load_model(self, model_path):
        pass


def image_classifier(model_path, model_type):
    ''' Factory method for ImageClassifier '''

    if model_type == 'fastai':
        return FastaiImageClassifier(model_path)

    elif model_type == 'pytorch':
        return PytorchImageClassifier(model_path)

    elif model_type == 'tensorflow':
        return TensorflowImageClassifier(model_path)
