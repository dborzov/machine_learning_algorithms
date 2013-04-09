import random
import numpy as np


N=100
MONTE_CARLO=1000

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

def classify(v1,v2):
	return mapped(np.dot(v1,v2)>0)


def PLA_experiment():
	f=Target()
	training=[training_point(f) for _ in range(N)]
	w=np.array([0.,0.,0.])
	mismatched=[point for point in training if classify(w,point['v'])<>point['y']]
	iteration_count=0
	while len(mismatched)>0 or iteration_count>50000:
		iteration_count+=1
		#print 'iteration count is ',iteration_count, w,' Number of mismatched points:',len(mismatched)
		mismatched=[point for point in training if classify(w,point['v'])<>point['y']]
		if len(mismatched)>0:
			correcting_point=random.choice(mismatched)
			w=w+correcting_point['y']*correcting_point['v']

    
	miss=0.
	for i in range(MONTE_CARLO):
		a=training_point(f)
		if classify(w,a['v'])<>a['y']:
			miss+=1.
	probability=np.float(miss/np.float(MONTE_CARLO))	
	return {'probability':probability,'iterations':iteration_count}



experiments=[PLA_experiment() for i in range(1000)]
its=np.array([each['iterations'] for each in experiments])
print 'probability average ',np.mean(its)
area=np.array([each['probability'] for each in experiments])
print 'probability average ',np.mean(area)



