# poisson
import numpy as np
class DDistribution(object):
    def __init__(self, name):
        self.name = name

    def sample(self):
        raise NotImplementedError



class PoissonDistribution(DDistribution):

    _NAME = 'Poisson'

    def __init__(self, lam):
        super(PoissonDistribution, self).__init__(self._NAME)
        self.lam = lam

    def sample(self, n=100):
        return np.random.poisson(self.lam, n)