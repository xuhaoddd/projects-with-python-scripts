# -*- coding: utf-8 -*-
# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(12).reshape(2,6)
x1=x[0,:]
x2=x[1,:]
y = 3 - 2 * x1 + x1 ** 2 - x1 ** 3- 4 * x2 + x2 ** 2 - x2 ** 3

dataframe = pd.DataFrame({'a_name':x1,'b_name':x2,'c_name':y})
dataframe.to_csv("test.csv",index=False,sep=',')

# Importing the dataset
dataset = pd.read_csv('test.csv')


#X = dataset.iloc[:, 1:2].values  # 自变量应该是矩阵
#y = dataset.iloc[:, 2].values    # 因变量应该是向量

x=dataset.iloc[:,0:2].values
y=dataset.iloc[:,2].values


# Fitting Polynomail Regression to the dataset
model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression(fit_intercept=False))])

model = model.fit(x, y)

#model.predict(np.array([[1,7]]))

fig = plt.figure(1)
ax = Axes3D(fig)
X1_grid = np.arange(min(x1), max(x1), 0.1)
X2_grid = np.arange(min(x2), max(x2), 0.1)
X1, X2 = np.meshgrid(X1_grid, X2_grid)
array_combine = np.dstack((X1,X2)).reshape(2500,2)

Z=model.predict(array_combine)

Z_=Z.reshape(50,50)

ax.plot_surface(X1, X2 , Z_, rstride=1, cstride=1)

#plt.show(1)


plt.figure(2)
plt.contourf(X1, X2, Z_, 20, alpha=1.0, cmap='jet')
C = plt.contour(X1, X2, Z_, 20, colors='black', linewidth=.5)
plt.clabel(C, inline=True, fontsize=10)
plt.show(2)

#error analysis
x1_=array_combine[:,0]
x2_=array_combine[:,1]
y_ = 3 - 2 * x1_ + x1_ ** 2 - x1_ ** 3- 4 * x2_ + x2_ ** 2 - x2_ ** 3

Z_predict=model.predict(array_combine)

error = 1 / len(y_) * np.sum(np.power((Z_predict-y_),2))

print("predict_error:",error)
























