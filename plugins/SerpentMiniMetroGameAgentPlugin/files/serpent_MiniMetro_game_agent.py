from serpent.game_agent import GameAgent
from serpent.input_controller import MouseButton

class SerpentMiniMetroGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):
        # CLICK "Play"
        self.input_controller.move(x=270, y=210)
        self.input_controller.click()
        # CLICK "Play"
        self.input_controller.move(x=700, y=510)
        self.input_controller.click()


    def handle_play(self, game_frame):
        pass
