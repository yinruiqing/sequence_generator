import warnings
import numpy as np
from typing import Optional

from scipy.cluster.hierarchy import fcluster
from pyannote.core.utils.hierarchy import linkage

import sklearn.cluster
from scipy.spatial.distance import squareform
from pyannote.core.utils.distance import pdist
from pyannote.core.utils.distance import dist_range
from pyannote.core.utils.distance import l2_normalize
from pyannote.core.utils.cluster import chinese_whispers_clustering



import chocolate
from pyannote.pipeline.pipeline import Pipeline


class ChineseWhispersClustering(Pipeline):
    def __init__(self, method: Optional[str] = 'distance',
                       metric: Optional[str] = 'euclidean',
                       normalize: Optional[bool] = False,
                       max_iter: Optional[int] = 1000):

        super().__init__()
        self.method = method
        self.metric = metric
        self.max_iter = max_iter
        self.normalize = normalize

        min_dist, max_dist = dist_range(metric=self.metric,
                                        normalize=self.normalize)
        if not np.isfinite(max_dist):
            # this is arbitray and might lead to suboptimal results
            max_dist = 20
            msg = (f'bounding distance threshold to {max_dist:g}: '
                   f'this might lead to suboptimal results.')
            warnings.warn(msg)
        self.threshold = chocolate.uniform(min_dist, max_dist)



    def __call__(self, X: np.ndarray) -> np.ndarray:
        clusters = chinese_whispers_clustering(X, self.threshold, method=self.method,
                                                metric=self.metric,
                                                max_iter=self.max_iter)

        return clusters

