from serpent.game import Game

from .api.api import MiniMetroAPI

from serpent.utilities import Singleton




class SerpentMiniMetroGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "steam"

        kwargs["window_name"] = "Mini Metro"

        kwargs["app_id"] = "287980"
        kwargs["app_args"] = None
        
        
        

        super().__init__(**kwargs)

        self.api_class = MiniMetroAPI
        self.api_instance = None

    @property
    def screen_regions(self):
        regions = {
            "GAME_PASSENGER_COUNTER": (35, 679, 21, 639),
            "GAME_WEEKDAY": (16, 706, 37, 743),
            "GAME_CLOCK": (9, 741, 45, 779),
            "GAME_ACTIVE_LINES": (509, 275, 551, 555)
        }

        return regions

    @property
    def ocr_presets(self):
        presets = {
            "SAMPLE_PRESET": {
                "extract": {
                    "gradient_size": 1,
                    "closing_size": 1
                },
                "perform": {
                    "scale": 10,
                    "order": 1,
                    "horizontal_closing": 1,
                    "vertical_closing": 1
                }
            }
        }

        return presets
