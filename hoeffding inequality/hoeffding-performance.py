import numpy as np
import random

TOTAL_NUMBER_OF_EXPERIMENTS = 100000
TOTAL_NUMBER_OF_COINS = 1000


class Coin(object):
    def __init__(self):
        self.record = [random.choice([0., 1.]) for _ in range(10)]

    def get_frequency(self):
        return np.mean(self.record)


Sum_n0, Sum_nRand, Sum_nMin = 0., 0., 0.
for i in range(TOTAL_NUMBER_OF_EXPERIMENTS):
    experiment = [Coin().get_frequency() for _ in range(TOTAL_NUMBER_OF_COINS)]
    Sum_nMin += min(experiment)
    if i% 500==0:
        print i/float(TOTAL_NUMBER_OF_EXPERIMENTS), " percent done. nMin so far: ",  Sum_nMin/float(i+1)


denominator = float(TOTAL_NUMBER_OF_EXPERIMENTS)
n0 = Sum_n0/denominator
nRand = Sum_nRand/denominator
nMin = Sum_nMin/denominator
print 'n0, nRand, nMin', n0, nRand, nMin
