import random
import numpy as np


N = 1000
NUMBER_OF_EXPERIMENTS = 1000


def mapped(boo):
    if boo:
        return +1.
    else:
        return -1.


class Target():
    def output(self, x, y):
        return mapped(x**2 + y**2 - 0.6 > 0)


def training_point(target):
    x = random.uniform(-1., 1.)
    y = random.uniform(-1., 1.)
    return {'v': np.array([1., x, y]), 'y': target.output(x, y)}


def classify(v1, v2):
    return mapped(np.dot(v1, v2) > 0)


def LinearRegression_experiment():
    f = Target()
    sample = [training_point(f) for _ in range(N)]
    for _ in np.arange(N/10.):
        random.choice(sample)['y'] *= (-1.)  # we introduce noise with random flipping of 10% of the points

    LargeX = np.matrix([point['v'] for point in sample])
    yvec = np.array([point['y'] for point in sample])
    Square = LargeX.T * LargeX
    Operator = Square.I * LargeX.T
    w = np.dot(Operator, yvec.T)
    misclassified = [point for point in sample if not classify(w, point['v']) == point['y']]
    E_in = len(misclassified)/float(N)
    return E_in


print np.mean([LinearRegression_experiment() for i in range(NUMBER_OF_EXPERIMENTS)])

