# import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

mpa = 130

# Read the dataset from csv file
df1 = pd.read_csv('knVSmpa.csv')
df1.head()

df2 = pd.read_csv('mmpmVSmpa.csv')
df2.head()

df3 = pd.read_csv('rpmVSmpa.csv')
df3.head()


y1 = df1['kn'].values.reshape(-1, 1)  #target y
X1 = df1['mpa'].values.reshape(-1, 1)  #feature x

y2 = df2['mmpm'].values.reshape(-1, 1)  #target y
X2 = df2['mpa'].values.reshape(-1, 1)  #feature x

y3 = df3['rpm'].values.reshape(-1, 1)  #target y
X3 = df3['mpa'].values.reshape(-1, 1)  #feature x





SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.2, random_state = SEED)
regressor1 = LinearRegression()
regressor1.fit(y1, X1)
print(regressor1.intercept_)
print(regressor1.coef_)
kn = regressor1.predict([[mpa]])
print("kn -")
print(kn) # 94.80663482

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.2, random_state = SEED)
regressor2 = LinearRegression()
regressor2.fit(y2, X2)
print(regressor2.intercept_)
print(regressor2.coef_)
mmpm = regressor2.predict([[mpa]])
print(" mmpm -")
print(mmpm) # 94.80663482


X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size = 0.2, random_state = SEED)
regressor3 = LinearRegression()
regressor3.fit(y3, X3)

print(regressor3.intercept_)
print(regressor3.coef_)

rpm = regressor3.predict([[mpa]])

print(" mmpm -")
print(rpm) # 94.80663482