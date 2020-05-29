import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
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

##open-vat分类
def open_sing_cat(category):
    if category == 1:
        return 1
    else:
        return 0
##lnnumber分类
def lnnumber_cat(lnnumber):
    if lnnumber < lnnumber_mean:
        return 0
    else:
        return 1

train = pd.read_csv('data/train_0313.csv')
print(train)
target = 'outcome1'
pat_id = 'patient_id'
x_columns = ['LYM3','Lnnumbers','stAging','open-vats','approach',
                            'surgtime','LYM1','NEUT1','NEUT3','T']
X = train[x_columns]
X = X.dropna(axis=0, how='any') ##去除空数据
train = train.dropna(subset=['LYM3','Lnnumbers','stAging','open-vats','approach',
                            'surgtime','LYM1','NEUT1','T'])
y = train[target]

lnnumber_mean = X['Lnnumbers'].mean()  #lnnumber均值

## 数值数据区间缩放
for stp in ['surgtime','LYM1','NEUT1','NEUT3']:
    X[stp] = (X[stp] - X[stp].min()) / (X[stp].max() - X[stp].min())

## 类别类型数据处理 one-hot表示
for dum in ['open-vats','stAging','approach','T','Lnnumbers']:
    if dum != 'Lnnumbers':
        dummy_train = pd.get_dummies(train[dum], prefix=dum)
    else:
        dummy_train = pd.get_dummies(train[dum].map(lnnumber_cat), prefix=dum)
    X = X.join(dummy_train)
    X = X.drop([dum], axis=1)

X.to_csv('result.csv') #输出训练数据输出文件

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23) #测试集训练集区分 test_size 训练测试比例设置

##分类器
names = ['dt','nn','knn', 'rf','gp','nb','ada']
classifiers = [tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=4, min_samples_leaf=6),
                MLPClassifier(alpha=0.0001, max_iter=3000, activation='relu', solver='adam'),
               KNeighborsClassifier(3),
               RandomForestClassifier(max_depth=5, n_estimators=30, max_features=1),
               GaussianProcessClassifier(1.0 * RBF(1.0)),
               GaussianNB(),
               AdaBoostClassifier()
            ]

for name, clf in zip(names,classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test) #准确率
    y_prob = clf.predict_proba(X_test) ##预测每个类型的概率
    y_pred = clf.predict(X_test)  ## 预测的类别
    auc = roc_auc_score(np.array(y_test), y_prob[:,1]) ##auc计算
    # print(y_prob[:,1])
    # print(np.array(y_test)) ##测试数据真实类别
    # print(y_pred)
    print(name + ' ' + str(score) + ' auc ' + str(auc))
