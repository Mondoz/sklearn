from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib

clf = svm.SVC(gamma=0.001, C=100.)

iris = datasets.load_iris()
digits = datasets.load_digits()

clf.fit(digits.data[:-1],digits.target[:-1])
print(clf.predict(digits.data[-1:]))

joblib.dump(clf,'svc_model.pkl')