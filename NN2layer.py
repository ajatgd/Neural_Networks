import numpy as np

def sigmoidfn(x,derivative=False):
    if(derivative==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]) #input dataset
y = np.array([[0,0,1,1]]).T #output dataset

np.random.seed(1) #seed random numbers so that the weights are random but remain same for further opn

weights = 2*np.random.random((3,1)) - 1 #initialize weights randomly

for iter in xrange(10000):
    l0 = X #forward propogation
    l1 = sigmoidfn(np.dot(l0,weights)) #calculating sigmoidal values of dot product
    l1_error = y - l1 #calculating error from actual value
    l1_delta = l1_error * sigmoidfn(l1,True) #slope of the sigmoid at the values in l1
    weights += np.dot(l0.T,l1_delta) #update weights

print "Output After Training is",l1
