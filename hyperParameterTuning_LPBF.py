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

# Load the dataset as an example
df = pd.read_csv('D:\PhD_ResearchWork\ASME_Journal\datasets\LPBF_Dataset_Combined_v2_python_Depth.csv')
X = df[['Laser Power [W]', 'Scanning Speed [mm/s]', 'Layer Thickness [um]', 'Spot Size [um]', 'Porosity [%]']].values
y = df['Max Melt Pool Depth [um]'].values


# Create a list of ensemble regressors
# ensemble_methods = [

#     ('Multi Layer Perce,alpha=0.001,random_state=20,early_stopping=False)),
#     ('Gaussian Process', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),random_state=0).fit(X, y)),








# ]


model_params = {

    'Multi Layer Perception': {
        'model': MLPRegressor(),
        'params': {
            'activation':['relu'],
            'alpha':[0.001],
            # 'batch_size',
            # 'beta_1',
            # 'beta_2',
            'early_stopping': [False],
            # 'epsilon',
            'hidden_layer_sizes': [10, 100],
            # 'learning_rate',
            # 'learning_rate_init',
            # 'max_fun',
            # 'max_iter',
            # 'momentum',
            # 'n_iter_no_change',
            # 'nesterovs_momentum',
            # 'power_t',
            'random_state': [20],
            # 'shuffle',
            # 'solver',
            # 'tol',
            # 'validation_fraction',
            # 'verbose',
            # 'warm_start'
        }
    },
    'Guassian Process': {
        'model': GaussianProcessRegressor(),
        'params': {
            'alpha',
            'copy_X_train',
            'kernel',
            'n_restarts_optimizer',
            'normalize_y',
            'optimizer',
            'random_state'
        }
    },
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {
            'copy_X',
            'fit_intercept',
            'n_jobs',
            'positive'
        }
    },
    # 'Polynomial Regression': {
    #     'model': SVR(),
    #     'params': {
    #         'C': [.2, .5, 1],
    #         'kernel': ['rbf', 'linear']
    #     }
    # },
    'Support Vector Machine': {
        'model': SVR(),
        'params': {
            'C': [.2, .5, 1],
            # 'cache_size',
            # 'coef0',
            # 'degree',
            # 'epsilon',
            # 'gamma',
            'kernel': ['rbf', 'linear'],
            # 'max_iter',
            # 'shrinking',
            # 'tol',
            # 'verbose'
        }
    },
    'KNN': {
        'model': KNeighborsRegressor(),
        'params': {
            # 'algorithm',
            # 'leaf_size',
            # 'metric',
            # 'metric_params',
            # 'n_jobs',
            # 'n_neighbors',
            # 'p',
            # 'weights'
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(),
        'params': {
            # 'bootstrap',
            # 'ccp_alpha',
            # 'criterion',
            # 'max_depth',
            # 'max_features',
            # 'max_leaf_nodes',
            # 'max_samples',
            # 'min_impurity_decrease',
            # 'min_samples_leaf',
            # 'min_samples_split',
            # 'min_weight_fraction_leaf',
            'n_estimators': [100],
            # 'n_jobs',
            # 'oob_score',
            # 'random_state',
            # 'verbose',
            # 'warm_start'
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(),
        'params': {
            # 'alpha',
            # 'ccp_alpha',
            # 'criterion',
            # 'init',
            # 'learning_rate',
            # 'loss',
            # 'max_depth',
            # 'max_features',
            # 'max_leaf_nodes',
            # 'min_impurity_decrease',
            # 'min_samples_leaf',
            # 'min_samples_split',
            # 'min_weight_fraction_leaf',
            'n_estimators': [100],
            # 'n_iter_no_change',
            # 'random_state',
            # 'subsample',
            # 'tol',
            # 'validation_fraction',
            # 'verbose',
            # 'warm_start'
        }
    },
    'AdaBoost': {
        'model': AdaBoostRegressor(),
        'params': {
            # 'base_estimator',
            # 'estimator',
            # 'learning_rate',
            # 'loss',
            'n_estimators': [100],
            # 'random_state'
        }
    },
    'Bagging': {
        'model': BaggingRegressor(),
        'params': {
            # 'base_estimator',
            # 'bootstrap',
            # 'bootstrap_features',
            # 'estimator',
            # 'max_features',
            # 'max_samples',
            'n_estimators': [100],
            # 'n_jobs',
            # 'oob_score',
            # 'random_state',
            # 'verbose',
            # 'warm_start'
        }
    },
    'extraTreesRegressor': {
        'model': ExtraTreesRegressor(),
        'params': {
            #'bootstrap': [True, False],
            # 'ccp_alpha',
            # 'criterion',
            # 'max_depth',
            # 'max_features',
            # 'max_leaf_nodes',
            # 'max_samples',
            # 'min_impurity_decrease',
            # 'min_samples_leaf',
            # 'min_samples_split',
            # 'min_weight_fraction_leaf',
            'n_estimators': [100],
            # 'n_jobs',
            # 'oob_score',
            #'random_state': [100, 200, 300],
            # 'verbose',
            #'warm_start': [True, False],
            ## 'kernel': ['rbf', 'linear']
        }
    },

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