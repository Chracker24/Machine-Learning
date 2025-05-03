from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

data = np.array(np.genfromtxt("Salary_Prediction/Data/Salary_Data.csv",delimiter=',',skip_header=1))
X=data[:,0].reshape(-1,1)
Y=data[:,1]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
model = linear_model.LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
print('R2 score: %.2f' % r2_score (Y_test,Y_pred))

plt.scatter(X_train, Y_train, color='blue', label="Actual Data")
plt.scatter(X_test, Y_test, color="black",label="Test Data")
plt.plot(X_test, Y_pred, color="red", label="Predicted Line")
plt.show()