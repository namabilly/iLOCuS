import numpy as np
class DriverSim:
    def __init__(self):
        pass

    def react(self, pricing):
        return np.random.rand(4,15,15), False
    
    def reset(self):
        return np.random.rand(4,15,15)
