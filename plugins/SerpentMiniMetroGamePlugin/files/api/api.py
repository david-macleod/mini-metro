from serpent.game_api import GameAPI
from .object_detector import ObjectDetector

import offshoot


class MiniMetroAPI(GameAPI):

    def __init__(self, game=None):
        super().__init__(game=game)
        self.ml_models_dir = offshoot.config["file_paths"]["game_ml_models"]
        self.ml_station_detector = ObjectDetector(
            frozen_graph_path=f"{self.ml_models_dir}/station_detector_m0_frozen_graph.pb",
            class_labels_path=f"{self.ml_models_dir}/station_detector_m0_label_map.json"
        )

    def my_api_function(self):
        pass

    class MyAPINamespace:

        @classmethod
        def my_namespaced_api_function(cls):
            api = MiniMetroAPI.instance
