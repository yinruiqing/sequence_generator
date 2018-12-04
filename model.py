from pyannote.pipeline import Pipeline
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.blocks.clustering import AffinityPropagationClustering
from pyannote.pipeline.blocks.clustering import HierarchicalAgglomerativeClustering

from pyannote.core import Segment, Annotation
import chocolate
from pyannote.pipeline import Optimizer

class ClusterPipeline(Pipeline):
    def __init__(self, name='ap', **params):
        super().__init__()
        if name not in ['ap', 'hac']:
            raise NotImplementedError
        if name == 'ap':
            self.clustering = AffinityPropagationClustering('euclidean')
        if name == 'hac':
            print(params)
            self.clustering = HierarchicalAgglomerativeClustering(**params)


    def __call__(self, item):
        X, _ = item
        return self.clustering(X)   # its parameters are already set by with_params

    def generate_annotation(self, uri, labels):
        res = Annotation(uri)
        for start, end, label in zip(range(0,len(labels)), range(1,len(labels)+1), labels):
            res[Segment(start,end)] = str(label)
        return res.support()

    def loss(self, item, y_pred):
        y_true = item[1]
        uri = 'tmp'
        der = GreedyDiarizationErrorRate()
        reference = self.generate_annotation(uri, y_true)
        hypothesis = self.generate_annotation(uri, y_pred)
            
        return abs(der(reference,hypothesis, uem=reference.get_timeline().extent()))



class ClusterOptimizer(object):
    def __init__(self, name, db_path, num_iter=100, **params):
        self.name = name
        self.num_iter = num_iter
        self.pipeline = ClusterPipeline(name, **params)
        self.optimizer = Optimizer(self.pipeline, db_path)


    def fit(self, data):
        result = self.optimizer.tune(data, n_iterations=self.num_iter)

    def predict(self, Xys):
        return [self.optimizer.best_pipeline(Xy) for Xy in Xys]
