import random
import numpy as np

a = np.array([{'l':float(i)} for i in range(10)])

for _ in range(3):
    random.choice(a)['l'] *= (-1.)

print a
