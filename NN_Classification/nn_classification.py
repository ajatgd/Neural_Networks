import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
dim_inp=2
dim_out=2
hidden_layer=3
alpha=0.01#learning rate
reg_lambda=0.01#regularization parameter

def main():
    X,y= datasets.make_moons(200,noise=0.20)#choosing a prdefined dataset in sklearn
    num_examples=len(X)
    np.random.seed(0)
    weight1=np.random.random((dim_inp,hidden_layer))-1
    bias_weight1=np.zeros((1,hidden_layer))
    weight2=np.random.random((hidden_layer,dim_out))-1
    bias_weight2=np.zeros((1,dim_out))
    model={}
    for i in range(0,20000):
        z1=X.dot(weight1)+bias_weight1#forward propagation
        a1=np.tanh(z1)#using tanh as activation fn.
        z2=a1.dot(weight2)+bias_weight2
        #calculating and reducing error, using Softmax fn.
        exp_scores=np.exp(z2)
        probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
        delta3=probs#backpropagation
        delta3[range(num_examples),y] -=1#as y is 0 or 1, 0 for red in graph and 1 for blue
        dw2=(a1.T).dot(delta3)
        db2=np.sum(delta3,axis=0,keepdims=True)
        delta2=delta3.dot(weight2.T)*(1-np.power(a1,2))#(1-np.power(a1,2) is derivative of tanh )
        dw1=(X.T).dot(delta2)
        db1=np.sum(delta2,axis=0)
        #adding regularisation parameter
        dw2 += reg_lambda*weight2
        dw1 += reg_lambda*weight1
        #updating weight using gradient descent learning rate
        weight1 += -alpha * dw1
        bias_weight1 += -alpha * db1
        weight2 += -alpha * dw2
        bias_weight2 += -alpha * db2
        #updating model
        model={'W1':weight1,'b1':bias_weight1,'W2':weight2,'b2':bias_weight2}
        if i%1000 ==0:
            print("loss after %i iteration:%f"%(i, calculate_loss(model, X, y)))
    visualize(X,y,model)
#Calculating loss using regularisation parameter
def calculate_loss(model, X, y):
    num_examples = len(X)  # training set size
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss

#visualizing the decision boundary for same
def visualize(X, y, model):
    plot_decision_boundary(lambda x:predict(model,x), X, y)
    plt.title("NN")


def plot_decision_boundary(pred_func, X, y):
    # Setting min and max values and some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


if __name__=="__main__":
    main()
