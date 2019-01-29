import numpy as np
class EGenerator(object):
    def __init__(self, name):
        self.name = name


class GaussEGenerator(EGenerator):
	_NAME = 'Gauss'

	def __init__(self, n_features=2, center_box= (-10.0, 10.0), cluster_std=0.5, std_low=0.2, std_high=1.5):
		self.n_features = n_features
		self.center_box = eval(center_box)
		self.cluster_std = cluster_std
		self.std_low = std_low
		self.std_high = std_high


	def generate_points(self, nums):
		n_centers = len(nums)
		n_samples = n_centers * max(nums)
		if self.cluster_std is None:
			cluster_std = np.random.uniform(low=self.std_low, high=self.std_high, size=(n_centers,))
		else:
			cluster_std = self.cluster_std
		from sklearn.datasets.samples_generator import make_blobs
		X, y = make_blobs(n_samples=n_samples, centers=n_centers, cluster_std=cluster_std,
			n_features=self.n_features, center_box=self.center_box, shuffle=False)
			#n_features=self.n_features, center_box=self.center_box, shuffle=False, random_state=random_seed)
		return [X[y==label][:nums[label]] for label in range(n_centers)]

		

