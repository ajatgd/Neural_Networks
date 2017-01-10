import numpy as np

def sigmoidfn(x,derivative=False):
    if (derivative==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0,1,1,0]]).T

np.random.seed(1)

weight1=2*np.random.random((3,4))-1
weight2=2*np.random.random((4,1))-1

for iter in xrange(60000):
    l0=X
    l1=sigmoidfn(np.dot(l0,weight1))
    l2=sigmoidfn(np.dot(l1,weight2))
    l2_error=y-l2
    if (iter% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))
    l2_delta=l2_error*sigmoidfn(l2,True)
    l1_error=l2_delta.dot(weight2.T)
    l1_delta=l1_error*sigmoidfn(l1,True)
    weight2 += l1.T.dot(l2_delta)
    weight1 += l0.T.dot(l1_delta)

print "Output after traing is:",l2
