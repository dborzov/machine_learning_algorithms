import random
import numpy as np


N = 10
NUMBER_OF_EXPERIMENTS = 1000
MONTE_CARLO = 1000


def mapped(boo):
    if boo:
        return +1.
    else:
        return -1.


class Target():
    def __init__(self):
        self.generate()
    def generate(self):
        self.x1, self.y1=random.uniform(-1.,1.), random.uniform(-1.,1.)
        self.x2, self.y2=random.uniform(-1.,1.), random.uniform(-1.,1.)
    def output(self,x,y):
        return mapped(y*(self.x2-self.x1)-x*(self.y2-self.y1) - self.y1*self.x2 + self.y2*self.x1>0)


def training_point(target):
    x=random.uniform(-1.,1.)
    y=random.uniform(-1.,1.)
    return {'v':np.array([1.,x,y]),'y':target.output(x,y)}


def classify(v1, v2):
    return mapped(np.dot(v1, v2) > 0)


def LinearRegression_experiment():
    f=Target()
    sample=[training_point(f) for _ in range(N)]
    LargeX=np.matrix([point['v'] for point in sample])
    yvec=np.array([point['y'] for point in sample])
    Square = LargeX.T * LargeX
    Operator = Square.I * LargeX.T
    w = np.dot(Operator, yvec.T)


print np.mean([LinearRegression_experiment() for i in range(NUMBER_OF_EXPERIMENTS)])

