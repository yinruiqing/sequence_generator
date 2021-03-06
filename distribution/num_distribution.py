# uniform
import numpy as np
from numpy.random import randint
class NDistribution(object):
    def __init__(self, name):
        self.name = name

    def sample(self):
        raise NotImplementedError



class UniformDistribution(NDistribution):

    _NAME = 'Uniform'

    def __init__(self, start=1, stop=5):
        super(UniformDistribution, self).__init__(self._NAME)
        self.start = start
        self.stop = stop

    def sample(self):
        return randint(self.start, self.stop)