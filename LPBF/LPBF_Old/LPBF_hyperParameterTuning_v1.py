from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,

)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic



# Load the dataset as an example
df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_Width.csv')
X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
y = df['Max Melt Pool Width [um]'].values



#     ('Multi Layer Perce,alpha=0.001,random_state=20,early_stopping=False)),
# (kernel=DotProduct() + WhiteKernel(),random_state=0).fit(X, y)),

model_params = {


    # 'Guassian Process': {
    #     'model': GaussianProcessRegressor(),
    #     'params': {
    #         'kernel': [DotProduct() + WhiteKernel(),RationalQuadratic(), RBF()],
    #     }
    # },
    # 'Linear Regression': {
    #     'model': LinearRegression(),
    #     'params': {
    #          # 'lambda': [0.001, 0.01, 0.1, 1, 10, 100],
    #     }
    # },
    # 'Polynomial Regression': {
    #     'model': SVR(),
    #     'params': {
    #         'C': [.2, .5, 1],
    #         'kernel': ['rbf', 'linear']
    #     }
    # },
    # 'Support Vector Machine': {
    #     'model': SVR(),
    #     'params': {
    #         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #         'C': [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1],
    #     }
    # },
    # 'KNN': {
    #     'model': KNeighborsRegressor(),
    #     'params': {
    #          'n_neighbors': [1,2,3,4,5,10,15,20,25,30,35,40,45,50],
    #                 }
    # },
    'Multi Layer Perception': {
        'model': MLPRegressor(),
        'params': {
            'activation': ['tanh', 'logistic', 'relu', 'identity'],
            'hidden_layer_sizes': [50,100,150,200,300,400,500,600],
            'alpha': [.00001, .0001, .001, .01, .1, 1],
            'max_iter': [600],
        }
    },
    # 'Random Forest': {
    #     'model': RandomForestRegressor(),
    #     'params': {
    #         'n_estimators': [10, 20, 30, 40, 50, 100, 200],
    #     }
    # },
    # 'Gradient Boosting': {
    #     'model': GradientBoostingRegressor(),
    #     'params': {
    #         'n_estimators': [10, 20, 30, 40, 50, 100, 200],
    #     }
    # },
    # 'AdaBoost': {
    #     'model': AdaBoostRegressor(),
    #     'params': {
    #         'n_estimators': [10, 20, 30, 40, 50, 100, 200],
    #     }
    # },
    # 'Bagging': {
    #     'model': BaggingRegressor(),
    #     'params': {
    #         'n_estimators': [10, 20, 30, 40, 50, 100, 200],
    #     }
    # },
    # 'extraTreesRegressor': {
    #     'model': ExtraTreesRegressor(),
    #     'params': {
    #         'n_estimators': [10, 20, 30, 40, 50, 100, 200],
    #     }
    # },

}



scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], scoring='neg_mean_absolute_error', cv=10, return_train_score=False)
    clf.fit(X, y)

    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)
df.to_csv('LPBF_HpTuning_Values_width.csv', index=False)