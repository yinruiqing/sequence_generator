from distribution.duration_distribution import PoissonDistribution
from distribution.num_distribution import UniformDistribution
from distribution.order_distribution import MarkovDistribution
from distribution.weight_distribution import UniformDistribution as WUniformDistribution
from embedding_generator import GaussEGenerator

uniform_distribution = UniformDistribution(start=2)
weight_distribution = WUniformDistribution()
order_distribution = MarkovDistribution()
duration_distribution = PoissonDistribution(5)
embedding_generator = GaussEGenerator()


generator = ToyIterator(30, uniform_distribution, weight_distribution, 
                duration_distribution, order_distribution, embedding_generator, max_spk = 10,
                batch_size=32, device='cuda')

# def sequence_generator():
#     while True:
#         yield X, y


# my_method = MyMethod()


# generator = sequence_generator()

# generate test data
X_test, y_test = [], []
for i in range(100):
    X, y = next(generator)
    X_test.append(X)
    y_test.append(y)


method1 = Method1()
method2 = Method2()
method3 = Method3()

for method in [method1, method2, method3]:

    method.fit(generator)

    error = []
    for X, y_true in X_test, y_test:
        y_pred = method.predict(X_test)
        error.append(metric(y_true, y_pred))

print(method.name, np.mean(error))