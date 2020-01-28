import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_regression


def modele(X,theta): 
    return X.dot(theta)
    
def cost_function(X,y,theta): 
    m=len(y)
    return 1/(2*m) * np.sum((modele(X,theta)-y)**2)
 
def grad (X,y,theta): 
    m=len(y)
    return 1/m * X.T.dot(modele(X,theta)-y)
    
    
def gradient_descent(X,y,theta,learning_rate,n_iterration):
    cost_history=np.zeros(n_iterration)
    for i in range(0,n_iterration): 
        theta=theta - learning_rate * grad(X,y,theta)
        cost_history[i]=cost_function(X,y,theta)
    return theta,cost_history
 
def coef_determination(y,prediction): 
    u=((y-prediction)**2).sum()
    v=((y-y.mean())**2).sum()
    return 1- u/v
    
    
    
    




x, y = make_regression(n_samples=100,n_features=1, noise=10)
plt.scatter(x,y)
y=y.reshape(y.shape[0],1)
X=np.hstack((x,np.ones(x.shape)))
theta=np.random.randn(2,1)
cost_function(X,y,theta)
theta_final,cost_history= gradient_descent(X,y,theta,0.01,1000)
prediction=modele(X,theta_final)
plt.scatter(x,y)
plt.plot(x,prediction,c='r')
plt.plot(range(1000),cost_history)
coef_determination(y,prediction)
