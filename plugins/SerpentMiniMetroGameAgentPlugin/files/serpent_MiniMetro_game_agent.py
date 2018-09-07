from serpent.game_agent import GameAgent


class SerpentMiniMetroGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):

        self.ml_station_detector = self.game.api.ml_station_detector

        # Should these setup actions be moved to Game api?
        # CLICK "Play"
        self.input_controller.move(x=270, y=210)
        self.input_controller.click()
        # CLICK "Play"
        self.input_controller.move(x=700, y=510)
        self.input_controller.click()



    def handle_play(self, game_frame):
        results = self.ml_station_detector.run_inference(game_frame.frame_array)

        for i, game_frame in enumerate(self.game_frame_buffer.frames):
            self.visual_debugger.store_image_data(
                self.ml_station_detector.draw_bounding_boxes(game_frame.frame, **results),
                game_frame.frame.shape,
                str(i)
            )
