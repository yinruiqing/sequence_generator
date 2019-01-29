# mm
import numpy as np
class ODistribution(object):
    def __init__(self, name):
        self.name = name

    def sample(self):
        raise NotImplementedError


class MarkovDistribution(ODistribution):
    _NAME = 'Markov'

    def __init__(self):
        super(MarkovDistribution, self).__init__(self._NAME)
       
    def sample(self, num, occup_weights):
        def transmat_from_weights(occup_weights):
            length = len(occup_weights)
            weight_mat = np.tile(occup_weights, length).reshape(length,length)
            for i in range(length):
                weight_mat[i][i]=0 
            transmat = weight_mat/weight_mat.sum(axis=-1).reshape(length,1)
            return transmat

        from hmmlearn.hmm import MultinomialHMM
        length = len(occup_weights)
        transmat = transmat_from_weights(occup_weights)
        model = MultinomialHMM(n_components=length)
        model.startprob_ = np.ones((length))/length
        model.transmat_ = transmat
        model.emissionprob_ = np.ones((length,length))/length

        _, Z = model.sample(num)
        return Z







