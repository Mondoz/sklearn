import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF

def open_sing_cat(category):
    if category == 1:
        return 1
    else:
        return 0


# train = pd.read_csv('data/0209_train.csv')
train = pd.read_csv('data/0209_train_origin.csv')
target = 'outcome1'
multi_target = 'outcome2'
pat_id = 'patient_id'

x_columns = [x for x in train.columns if x not in [target, pat_id, multi_target, 'sex','smoking','upper_lbe',
                                                   'left_side','squamous','open_sing_multi_123','Lnstation','Lnnumbers',
                                                   'T','N']]
lnnumber_mean = train['Lnnumbers'].mean()
lnstation_mean = train['Lnstation'].mean()

def lnnumber_cat(lnnumber):
    if lnnumber < lnnumber_mean:
        return 0
    else:
        return 1


def lnstation_cat(lnsnumber):
    if lnsnumber < lnstation_mean:
        return 0
    else:
        return 1

for stp in ['age','BMI','FEV1e','surgtime','intrablood']:
    train[stp] = (train[stp] - train[stp].min())/(train[stp].max() - train[stp].min())

X = train[x_columns]
y = train[target]
sex_dummy_train = pd.get_dummies(train['sex'], prefix='sex')
X = X.join(sex_dummy_train)
smoke_dummy_train = pd.get_dummies(train['smoking'], prefix='smoke')
X = X.join(smoke_dummy_train)
squamous_dummy_train = pd.get_dummies(train['squamous'], prefix='squamous')
X = X.join(squamous_dummy_train)
train['open_sing_multi_123'] = train['open_sing_multi_123'].map(open_sing_cat)
open_dummy_train = pd.get_dummies(train['open_sing_multi_123'],prefix='open_sing_multi')
X = X.join(open_dummy_train)
train['Lnstation'] = train['Lnstation'].map(lnstation_cat)
lnstation_dummy_train = pd.get_dummies(train['Lnstation'],prefix='Lnstation')
X = X.join(lnstation_dummy_train)
train['Lnnumbers'] = train['Lnnumbers'].map(lnnumber_cat)
lnnumber_dummy_train = pd.get_dummies(train['Lnnumbers'],prefix='Lnnumbers')
X = X.join(lnnumber_dummy_train)
# X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
print(np.array(X.iloc[0]))

names = ['dt','nn','knn', "lsvm", "svm",'rf','gp','nb','ada']
classifiers = [tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=4, min_samples_leaf=6),
                MLPClassifier(alpha=0.0001, max_iter=3000, activation='relu', solver='adam'),
               KNeighborsClassifier(3),
               SVC(kernel="linear", C=0.025),
               SVC(gamma=2, C=1),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               GaussianProcessClassifier(1.0 * RBF(1.0)),
               GaussianNB(),
               AdaBoostClassifier(),
            ]

for name, clf in zip(names,classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    result = clf.predict(X_test)
    auc = roc_auc_score(np.array(y_test), result)
    print(name + ' ' + str(score) + ' auc ' + str(auc))





