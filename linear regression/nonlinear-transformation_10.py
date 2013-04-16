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
    def output(self, x1, x2):
        return mapped(x1**2 + x2**2 - 0.6 > 0)


def training_point(target):
    x1 = random.uniform(-1., 1.)
    x2 = random.uniform(-1., 1.)
    return {'v': np.array([1., x1, x2, x1*x2, x1**2, x2**2]), 'y': target.output(x1, x2)}


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

    out_sample = [training_point(f) for _ in range(1000)]
    for _ in np.arange(N/10.):
        random.choice(out_sample)['y'] *= (-1.)  # we introduce noise with random flipping of 10% of the points

    misclassified = [point for point in out_sample if not classify(w, point['v']) == point['y']]
    E = len(misclassified)/float(1000)
    return E


print np.mean([LinearRegression_experiment() for i in range(NUMBER_OF_EXPERIMENTS)])

