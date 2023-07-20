#import time

class Clock:
    def __init__(self, simulation_step):
        self.time = 0.0
        self.simulation_step = simulation_step

    def restart(self):
        self.time = 0.0

    def get_time(self):
        return self.time

    def get_step(self):
        return self.time/self.simulation_step

    def increment_time(self, inc):
        self.time = self.time + inc

