
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import make_regression
from datetime import datetime
# Import necessary libraries
import numpy as np
import pandas as pd
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
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from datetime import datetime

# output='depth'
# input_file='DataAug/LPBF_Final_Dataset/LPBF_Dataset_Normalized_Depth.csv'

output = 'Depth'
input_file = f'D:\PhD_ResearchWork\ASME_Journal\datasets\Final\LPBF_Dataset_Normalized_{output}.csv'

df = pd.read_csv(input_file)

X=df.iloc[:, :-1]
y=df.iloc[:, -1]

# Specify different hyperparameter combinations
hidden_layer_sizes_values = [50, 100] #default 100
activation_values = ['tanh', 'logistic', 'relu', 'identity'] #default relu
solver_values = ['lbfgs', 'sgd', 'adam'] #default adam
alpha_values = [.00001, .0001, .001, .01] #default 0.0001
batch_size_values = [200] #default auto
learning_rate_values = ['constant', 'invscaling', 'adaptive'] #default constant


poly_list=[]
cv = KFold(n_splits=10, shuffle=True, random_state=42)
# Perform 10-fold cross-validation for each combination of hyperparameters
for hidden_layer_sizes in hidden_layer_sizes_values:
    for activation in activation_values:
        for solver in solver_values:
            for alpha in alpha_values:
                for batch_size in batch_size_values:
                    for learning_rate in learning_rate_values:
                        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                                      activation=activation,
                                                      solver=solver,
                                                      alpha=alpha,
                                                      batch_size=batch_size,
                                                      learning_rate=learning_rate,
                                                      max_iter=600,
                                                      early_stopping=False,
                                                      random_state=42)


                        mse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
                        mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
                        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')


                        # Calculate metrics
                        mse = mse_scores.mean()
                        mae = mae_scores.mean()
                        corr = r2_scores.mean()

                        # Calculate additional metrics
                        rmse = np.sqrt(mse)
                        rae = mae / np.mean(np.abs(y - np.mean(y)))
                        rrse = rmse / np.sqrt(np.mean((y - np.mean(y))**2))

                        poly_list.append({
                            'hidden_layer_sizes': hidden_layer_sizes,
                            'activation': activation,
                            'solver': solver,
                            'alpha': alpha,
                            'batch_size': batch_size,
                            'learning_rate': learning_rate,
                            'R-squared': corr,
                            'MAE': mae,
                            'MSE': mse,
                            'RMSE': rmse,
                            'RAE': rae,
                            'RRSE': rrse
                        })

                        scores = cross_val_score(model, X, y, cv=cv,
                                                 scoring='neg_mean_squared_error')

                        # Display the results
                        print(f"hidden_layer_sizes: {hidden_layer_sizes}, activation: {activation}, solver: {solver}, "
                              f"alpha: {alpha}, batch_size: {batch_size}, learning_rate: {learning_rate}, Mean Squared Error: {-scores.mean()}"
                              )

poly_results = pd.DataFrame(poly_list)
# Save results to a CSV file

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f'LPBF_HP_MultiLayerPerception_results_{output}_{timestamp}.csv'

poly_results.to_csv(filename, index=False)