from serpent.game_agent import GameAgent 


class SerpentMiniMetroGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play

        self.ml_station_detector = self.game.api.ml_station_detector
        self.ml_context_classifier = self.game.api.ml_context_classifier

        self.ctrl = self.input_controller

    def setup_play(self):

        # Start Game
        (self.game.context.click_play(self.ctrl)
                          .click_mode(self.ctrl)
                          .click_normal(self.ctrl)
                          .click_return(self.ctrl)
                          .click_play(self.ctrl))


    def handle_play(self, game_frame):
        
        results = self.ml_station_detector.predict(game_frame.frame)

        print(self.ml_context_classifier.predict(game_frame.frame)['category'])

        for i, game_frame in enumerate(self.game_frame_buffer.frames):
            self.visual_debugger.store_image_data(
                #self.ml_station_detector.draw_bounding_boxes(game_frame.frame, **results),
                game_frame.frame,
                game_frame.frame.shape,
                str(i)
            )
