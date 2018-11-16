import numpy as np

class SGenerator(object):
    def __init__(self, duration, ndistribution, wdistribution, ddistribution, odistribution, egenerator):
        self.duration = duration
        self.ndistribution = ndistribution
        self.wdistribution = wdistribution
        self.ddistribution = ddistribution
        self.odistribution = odistribution
        self.egenerator = egenerator

    def re_order(self, seq):
        seen = set()
        seen_add = seen.add
        orders = [x for x in seq if not (x in seen or seen_add(x))]
        res = np.zeros(len(seq),dtype=int)
        for i,label in enumerate(orders):
            res[seq==label] = i+1
        return res

    def sample_one(self, label_len, reorder=True):
        num = self.ndistribution.sample()
        weights = self.wdistribution.sample(num)
        orders = self.odistribution.sample(label_len, weights)
        durations = self.ddistribution.sample(label_len)

        labels = [[label] * dur for label, dur in zip(orders, durations)]
        labels = [label for sublist in labels for label in sublist]
        labels = np.array(labels)

        if reorder:
            labels = self.re_order(labels)

        counter = [sum(labels==i) for i in range(num)]
        embedding_cands = self.egenerator.generate_points(counter)
        embeddings = np.zeros((len(labels), self.egenerator.n_features))
        for i, cand in enumerate(embedding_cands):
            embeddings[labels==i] = cand

        return {'X':embeddings[:self.duration], 'y': labels[:self.duration]}


    def sample_batch(self, label_len, batch_size):
        Xs = []
        ys = []
        for i in range(batch_size):
            one = self.sample_one(label_len)
            Xs.append(one['X'])
            ys.append(one['y'])

        return {'X':np.array(Xs), 'y': np.array(ys)}

    def generator(self, label_len, batch_size):
        while True:
            yield self.sample_batch(label_len, batch_size)



