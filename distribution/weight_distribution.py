import numpy as np

class WDistribution(object):
    def __init__(self, name):
        self.name = name

    def sample(self):
        raise NotImplementedError



class UniformDistribution(WDistribution):

    _NAME = 'Uniform Weight'

    def __init__(self):
        super(UniformDistribution, self).__init__(self._NAME)
        

    def sample(self, num):
        from random import randint
        weights = np.random.rand(num)
        return weights/weights.sum()