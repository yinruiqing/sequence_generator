import numpy as np
from numpy.random import randint
class SGenerator(object):
    def __init__(self, duration, ndistribution, wdistribution, ddistribution, odistribution, egenerator, random_seed=42):
        self.duration = duration
        self.ndistribution = ndistribution
        self.wdistribution = wdistribution
        self.ddistribution = ddistribution
        self.odistribution = odistribution
        self.egenerator = egenerator

        np.random.seed(random_seed)

    def re_order(self, seq):
        seen = set()
        seen_add = seen.add
        orders = [x for x in seq if not (x in seen or seen_add(x))]
        res = np.zeros(len(seq),dtype=int)
        for i,label in enumerate(orders):
            res[seq==label] = i+1
        return res

    def change_order(self, seq):
        label_ordered = self.re_order(seq)
        tmp = np.zeros(len(seq)-1,dtype=int)
        res = np.zeros(len(seq),dtype=int)
        
        label_orign = label_ordered[:-1]
        label_shift = label_ordered[1:]
        tmp[label_shift == label_orign] = 0
        tmp[label_shift < label_orign] = 1
        tmp[label_shift > label_orign] = 2
        res[0] = 2
        res[1:]=tmp
        
        return res

    def sample_one(self, label_len, reorder=True, change_label=False):
        #print(random_seed)
        num = self.ndistribution.sample()
        weights = self.wdistribution.sample(num)
        orders = self.odistribution.sample(label_len, weights)
        durations = self.ddistribution.sample(label_len)

        labels = [[label] * dur for label, dur in zip(orders, durations)]
        labels = [label for sublist in labels for label in sublist]
        labels = np.array(labels)

        if reorder:
            labels = self.re_order(labels)

        counter = [sum(labels==i) for i in range(1, num+1)]
        embedding_cands = self.egenerator.generate_points(counter)
        embeddings = np.zeros((len(labels), self.egenerator.n_features))
        for i, cand in enumerate(embedding_cands):
            embeddings[labels==(i+1)] = cand

        if change_label:
            labels = self.change_order(labels)
        return {'X':embeddings[:self.duration], 'y': labels[:self.duration]}


    def sample_same_batch(self, label_len, batch_size, change_label=False):
        Xs = []
        ys = []
        one = self.sample_one(label_len, change_label=change_label)
        for i in range(batch_size):
            Xs.append(one['X'])
            ys.append(one['y'])

        return {'X':np.array(Xs), 'y': np.array(ys)}


    def sample_batch(self, label_len, batch_size, change_label=False):
        Xs = []
        ys = []
        for i in range(batch_size):
            one = self.sample_one(label_len, change_label=change_label)
            Xs.append(one['X'])
            ys.append(one['y'])

        return {'X':np.array(Xs), 'y': np.array(ys)}

    def generator(self, label_len, batch_size, change_label=False):
        while True:
            yield self.sample_batch(label_len, batch_size, change_label)



