


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


model_params = {

    'svm': {
        'model': SVR(),
        'params': {
            'C': [.2, .5, 1],
            'kernel': ['rbf', 'linear']
        }
    },

    # 'svm': {
    #     'model': svm.SVC(gamma='auto'),
    #     'params': {
    #         'C': [1, 10, 20],
    #         'kernel': ['rbf', 'linear']
    #     }
    # },
    # 'random_forest': {
    #     'model': RandomForestClassifier(),
    #     'params': {
    #         'n_estimators': [1, 5, 10]
    #     }
    # },
    # 'logistic_regression': {
    #     'model': LogisticRegression(solver='liblinear', multi_class='auto'),
    #     'params': {
    #         'C': [1, 5, 10]
    #     }
    # },
    # 'naive_bayes_gaussian': {
    #     'model': GaussianNB(),
    #     'params': {}
    # },
    # 'naive_bayes_multinomial': {
    #     'model': MultinomialNB(),
    #     'params': {}
    # },
    # 'decision_tree': {
    #     'model': DecisionTreeClassifier(),
    #     'params': {
    #         'criterion': ['gini', 'entropy'],
    #
    #     }
    # }
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