import matplotlib.pyplot as plt
from numpy import mean, std

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RepeatedStratifiedKFold, GridSearchCV, cross_val_score)

from resources.properties import RANDOM_STATE
from resources.models import decision_tree, k_nearest_neighbors, random_forest, svm_svc, svm_nu, mlpc

# generate stacking model by combining multiple classification models
def get_stacked_model():
    level0 = [
        ('DecisionTree', decision_tree),
        ('K_NearestNeighbors', k_nearest_neighbors),
        ('RandomForest', random_forest),
        ('SVM_SVC', svm_svc),
        ('SVM_NuSVC', svm_nu),
        ('MLPClassifier', mlpc),
    ]
    level1 = LogisticRegression(random_state=RANDOM_STATE)
    return StackingClassifier(estimators=level0, final_estimator=level1, cv=10)

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    return cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# evaluate the models
def compare_models(models, X, y, scoring='accuracy'):
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        print('>>> %s %.4f (%.4f)' % (name, mean(scores), std(scores)))
    # plot model performance for comparison
    plt.boxplot(results, labels=names, showmeans=True)
    plt.title('Model Comparison')
    plt.xticks(rotation='60')
    plt.show()

def get_grid(model, cv, params):
    return GridSearchCV(
        estimator=model, 
        param_grid=params, 
        cv=cv,
        scoring="roc_auc",
        verbose=10,
        n_jobs=-1
    )

def get_grid_predictions(grid_model, X_test):
    return grid_model.predict(X_test)

def model_pipeline(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def get_model_predictions(models:dict, X_train, y_train, X_test):
    predictions = {}
    for model_name, model in models.items():
        y_pred = model_pipeline(model, X_train, y_train, X_test)
        predictions[model_name] = {
            'predictions': y_pred,
        }
    return predictions