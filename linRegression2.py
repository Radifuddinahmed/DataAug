# import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

mpa = 139

# Read the dataset from csv file


df2 = pd.read_csv('mmpmVSmpa.csv')
df2.head()






y2 = df2['mmpm'].values.reshape(-1, 1)  #target y
X2 = df2['mpa'].values.reshape(-1, 1)  #feature x







SEED = 42


X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.01, random_state = SEED)
regressor2 = LinearRegression()
regressor2.fit(y2, X2)
print(regressor2.intercept_)
print(regressor2.coef_)
mmpm = regressor2.predict([[mpa]])
print(" mmpm -")
print(mmpm) # 94.80663482


