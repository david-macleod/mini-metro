class GameState(object):

    def __init__(self, frame, station_detector):
        self.stations = self.extract_stations(frame, station_detector)
        self.lines = None
        self.terminals = None
        self.score = None

    def extract_stations(self, frame, station_detector):
        boxes = station_detector.run_inference(frame)['boxes']
        return self.find_box_centers(boxes)

    @staticmethod
    def find_box_centers(boxes):
        box_centers = list()
        for box in boxes:
            j = (box['xmax'] - box['xmin']) / 2
            i = (box['ymax'] - box['ymin']) / 2
            box_centers.append((i,j))
        return box_centers 