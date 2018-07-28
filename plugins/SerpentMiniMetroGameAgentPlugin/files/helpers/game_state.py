
class GameState(object):

    def __init__(self):
        pass


class Inventory(object):

    def __init__(self, trains=0, carriages=0, tunnels=0, stations=0):
        self.trains = trains
        self.carriages = carriages
        self.tunnels = tunnels
        self.stations = stations


class Line(object):

    def __init__(self, colour):
        self.colour = colour
        self.deployed = False

    def deploy(self):
        self.deployed = True


class Station(object):

    def __init__(self, shape, coords):
        self.shape = shape
        self.coords = coords

    def deploy(self):
        self.deployed = True

    def connect_stations(self):
        pass


class StationGraph(object):
    pass
