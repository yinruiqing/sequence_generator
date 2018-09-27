class EGenerator(object):
    def __init__(self, name):
        self.name = name


class GaussEGenerator(EGenerator):
	_NAME = 'Gauss'

	def __init__(self, n_features=2, center_box=(-10.0, 10.0), cluster_std=0.5):
		self.n_features = n_features
		self.center_box = center_box
		self.cluster_std = cluster_std


	def generate_points(self, nums):
		n_centers = len(nums)
		n_samples = n_centers * max(nums)
		from sklearn.datasets.samples_generator import make_blobs
		X, y = make_blobs(n_samples=n_samples, centers=n_centers, cluster_std=self.cluster_std,
			n_features=self.n_features, shuffle=False)
		return [X[y==label][:nums[label]] for label in range(n_centers)]

		

