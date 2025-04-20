import numpy as np
import functions

data = np.array(np.genfromtxt("Data/Salary_Data.csv",delimiter=',',skip_header=1))
x_train=data[:,0].reshape(-1,1)
y_train=data[:,1]

x_norm, xstd, xmean = functions.zscorenormalization(x_train)
y_norm, ystd, ymean = functions.zscorenormalization(y_train)
w_in = np.array((0,))  # Randomly initialize weights
b_in = 0  # Randomly initialize bias

print(w_in)
print(b_in)
alpha=0.0001
Iterations=100000

w_final,b_final,J_hist=functions.gradient_descent(x_norm,y_norm,w_in,b_in,alpha,Iterations)
w_final=w_final.item()
print("W_final : ",w_final)
print("B_final",b_final)
r2 = functions.predictions(x_norm,y_train,w_final,b_final,xstd,xmean,ystd,ymean,x_train,y_train)
print("R² Score:", r2)
functions.plotting(x_norm,y_norm,w_final,b_final,xstd, xmean, ystd, ymean)