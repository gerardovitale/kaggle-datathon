import pickle

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from resources.properties import RANDOM_STATE, PATH_MODEL

decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=8, max_leaf_nodes=26,min_samples_leaf=4, random_state=RANDOM_STATE)
k_nearest_neighbors = KNeighborsClassifier(n_neighbors=5)
random_forest = RandomForestClassifier(n_estimators=600, criterion="gini", max_depth=20, max_features="auto", min_samples_leaf=1, min_samples_split=2, n_jobs=-1,random_state=RANDOM_STATE, bootstrap=False,)
svm_svc = svm.SVC(C=50, degree=1, gamma="auto", kernel="rbf", probability=True, random_state=RANDOM_STATE)
svm_nu = svm.NuSVC(degree=1, kernel="rbf", nu=0.25, probability=True, random_state=RANDOM_STATE)
mlpc = MLPClassifier(activation="relu", alpha=0.1, hidden_layer_sizes=(10,10,10), learning_rate="constant", max_iter=3000, random_state=RANDOM_STATE)

xgboost = XGBClassifier(n_estimators=600, objective='multi:softmax', use_label_encoder=False, nthread=1)
xgforest = XGBRFClassifier(n_estimators=600, objective='multi:softmax', subsample=0.9, colsample_bynode=0.2, use_label_encoder=False)

model_name = PATH_MODEL + 'RandomForestOptimized.sav'
loaded_model = pickle.load(open(model_name, 'rb'))