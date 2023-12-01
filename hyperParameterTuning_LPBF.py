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


    'Guassian Process': {
        'model': GaussianProcessRegressor(),
        'params': {
            # 'kernel': ['RationalQuadratic'],
        }
    },
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {
             'lambda': [0.001, 0.01, 0.1, 1, 10, 100],
        }
    },
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
    #         'C': np.logspace(-5, 3, num=9), ##
    #         'gamma': np.logspace(-5, 3, num=9),
    #         'kernel': [ 'linear', 'polynomial','rbf', 'sigmoid'],
    #     }
    # },
    # 'KNN': {
    #     'model': KNeighborsRegressor(),
    #     'params': {
    #          'n_neighbors': [2,3,4,5 ,10,15,20,25,30,35,40,45,50], ## range(5,50,5) gives warning
    #                 }
    # },
    # 'Multi Layer Perception': {
    #     'model': MLPRegressor(),
    #     'params': {
    #         'activation': ['tanh', 'logistic', 'relu', 'identity'],
    #         'learning_rate': ['invscaling', 'constant', 'adaptive'],
    #         'max_iter': [200],
    #         # 'dropout_rate': [.01,.1,.2,.3],
    #         'batch_size': [32,64,128],
    #         'hidden_layer_sizes': [2,3,4],
    #         'early_stopping': [False],
    #     }
    # },
    # 'Random Forest': {
    #     'model': RandomForestRegressor(),
    #     'params': {
    #         'n_estimators': [ 10, 50, 100, 200],
    #         'max_depth' : [ 3, 5, 7, 10],
    #         'min_samples_split' : [2 , 5, 10],
    #     }
    # },
    # 'Gradient Boosting': {
    #     'model': GradientBoostingRegressor(),
    #     'params': {
    #         'n_estimators': [ 10, 50, 100, 200],
    #         'max_depth' : [ 3, 5, 7, 10],
    #         'min_samples_split' : [2 , 5, 10],
    #         #alpha
    #         'learning_rate': np.logspace(-5, 3, num=9),
    #     }
    # },
    # 'AdaBoost': {
    #     'model': AdaBoostRegressor(),
    #     'params': {
    #         'base_estimator': [DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5) , LinearRegression()],
    #         'n_estimators': [10,50,100,200],
    #         'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],  # Adjust as needed
    #
    #     }
    # },
    # 'Bagging': {
    #     'model': BaggingRegressor(),
    #     'params': {
    #         'base_estimator' : [DecisionTreeRegressor(), LinearRegression()],
    #         'n_estimators': [10,50,100,200],
    #     }
    # },
    # 'extraTreesRegressor': {
    #     'model': ExtraTreesRegressor(),
    #     'params': {
    #         'n_estimators': [10, 50, 100, 200],
    #         'max_depth': [3, 5, 7, 10],
    #         'min_samples_split': [2, 5, 10]
    #     }
    # },

}



scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X, y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)
df.to_csv('LPBF_HpTuning_Values_width.csv', index=False)