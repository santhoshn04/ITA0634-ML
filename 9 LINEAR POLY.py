import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv(r"C:\Users\Raju\Downloads\DATASET\DATASET\Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Linear Regression classifier model
#(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)


lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
#Polynomial Regression classifier model
#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#Visualising the Polynomial Regression results

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

