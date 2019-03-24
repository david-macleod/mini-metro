from pathlib import Path

from serpent.game_api import GameAPI
from .image_classifier import image_classifier
from .object_detector import object_detector

import offshoot


class MiniMetroAPI(GameAPI):

    def __init__(self, game=None):
        super().__init__(game=game)
        self.ml_models_dir = Path(offshoot.config['file_paths']['game_ml_models'])

        self.ml_station_detector = object_detector(
            model_path=self.ml_models_dir/'station_detector_tf_m6-1_frozen_graph.pb',
            model_type='tensorflow',
            category_labels_path=self.ml_models_dir/'station_detector_label_map.json'
        )
        self.ml_context_classifier = image_classifier(
            model_path=self.ml_models_dir/'context_classifier_fa_m0_learner.pkl',
            model_type='fastai'
        )

    def parse_game_state(self, frame):
        pass

    class MyAPINamespace:

        @classmethod
        def my_namespaced_api_function(cls):
            api = MiniMetroAPI.instance