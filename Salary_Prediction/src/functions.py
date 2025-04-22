import numpy as np
import copy
import matplotlib.pyplot as plt
import math

def zscorenormalization(x):
  mean = np.mean(x,axis=0)
  std = np.std(x,axis=0)
  return (x-mean)/std, std, mean


def compute_cost(x,y,w,b):
    m=x.shape[0]
    f=np.dot(w,x.T)+b
    cost=np.sum((f-y)**2)
    return cost/(2*m)

def compute_gradient(x,y,w,b):
    m,n=x.shape
    d_dw=np.zeros((n,))
    d_db=0
    for i in range(m):
        err=(np.dot(w,x[i])+b)-y[i]
        for j in range(n):
            d_dw[j]+=err*x[i,j]
        d_db+=err
    d_dw=d_dw/m
    d_db=d_db/m
    
    return d_dw, d_db

def gradient_descent(x,y,w_in,b_in,alpha,num_iters):
    j_history=[]
    w=copy.deepcopy(w_in)
    b=b_in
    
    for i in range(num_iters):
        d_dw,d_db=compute_gradient(x,y,w,b)
        w=w-alpha*d_dw
        b=b-alpha*d_db
        
        if i <100000:
            j_history.append(compute_cost(x,y,w,b))
        
        if i%math.ceil(num_iters/10)==0:
            print(f"Iterations {i:4d} : cost {j_history[-1]:8.3f}")
            
    return w,b,j_history

def predictions(x,y,w,b,xstd,xmean,ystd,ymean,x_train,y_train):
    y_pred_norm = np.dot(x,w)+b
    y_pred = (y_pred_norm*ystd)+ymean
    m=x_train.shape[0]
    print(f"{'Years of Experience':<25}{'Salary':<20}{'Predicted Salary':<5}")
    for i in range(m):
        print(f"{x_train[i].item():<25.1f}{y_train[i]:<20.2f}{y_pred[i].item():<5.2f}")
    r2 = R2(x,y,y_pred,w,b,ymean)
    return r2

def R2(x,y,y_pred,w,b,ymean):
    y=y.reshape(-1,1)
    pred=(y-y_pred)**2
    m=(y-ymean)**2
    numerator = np.sum(pred)
    denominator= np.sum(m)
    rs=numerator/denominator
    Rsquare=1-rs
    return Rsquare

def plotting(x, y, w, b, xstd, xmean, ystd, ymean):
  xo = (x*xstd)+xmean
  yo = (y*ystd)+ymean
  plt.scatter(xo, yo, color='blue', label='Actual Data')
  y_vals = np.dot(x, w) + b
  y_vals = (y_vals*ystd)+ymean
  plt.plot(xo, y_vals, color='red', label='Predicted Line')
  plt.xlabel('Years of Experience')
  plt.ylabel('Salary')
  plt.legend()
  plt.show()
  