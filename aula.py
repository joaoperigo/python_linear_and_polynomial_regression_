import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('HeightVsWeight.csv')
ind = dataset.iloc[:, 0:-1].values
dep = dataset.iloc[:, -1].values
print("ind:\n", ind)
print("dep:\n", dep)


# obter modelos de regressão linear simples e polinomial.
linearRegression = LinearRegression()
linearRegression.fit(ind, dep)

poly_features = PolynomialFeatures (degree= 2)
ind_poly = poly_features.fit_transform(ind)
polyLinearRegression = LinearRegression()
polyLinearRegression.fit(ind_poly, dep)

# exibe grafico simples
plt.scatter(ind, dep, color="red")
plt.plot(ind, linearRegression.predict(ind), color="blue")
plt.title("Regressão Linear Simples")
plt.xlabel("Nível")
plt.ylabel("Salário")
plt.show()

# exibe grafico polinomial
plt.scatter(ind, dep, color="red")
plt.plot(ind, polyLinearRegression.predict(ind_poly),
color="blue")
plt.title("Regressão Linear Polinomial")
plt.xlabel("Nível")
plt.ylabel("Salário")
plt.show()


# Obtenha um novo modelo de regressão polinomial utilizando degree=4. Exiba o gráfico
poly_features = PolynomialFeatures (degree= 4)
ind_poly = poly_features.fit_transform(ind)
polyLinearRegression = LinearRegression()
polyLinearRegression.fit(ind_poly, dep)

# exibe grafico simples"
plt.scatter(ind, dep, color="red")
plt.plot(ind, linearRegression.predict(ind), color="blue")
plt.title("Regressão Linear Simples")
plt.xlabel("Nível")
plt.ylabel("Salário")
plt.show()

# exibe grafico polinomial
plt.scatter(ind, dep, color="red")
plt.plot(ind, polyLinearRegression.predict(ind_poly),
color="blue")
plt.title("Regressão Linear Polinomial")
plt.xlabel("Nível")
plt.ylabel("Salário")
plt.show()


