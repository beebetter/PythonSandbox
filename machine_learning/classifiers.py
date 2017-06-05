#from sklearn import datasets
#iris = datasets.load_iris()

#X = iris.data
#y = iris.target
import pandas as pd
data = pd.read_csv('./data/AAPL.csv')[150:0:-1]
X = data.ix[:, 'Open':'Volume'].tolist()
y = data.ix[:, 'Adj Close'].tolist()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

classifiers = []
classifiers_names = []

from sklearn.neighbors import KNeighborsClassifier
classifiers.append(KNeighborsClassifier())
classifiers_names.append("KNeighborsClassifier")

from sklearn import tree
classifiers.append(tree.DecisionTreeClassifier())
classifiers_names.append("DecisionTreeClassifier")

from sklearn.linear_model import LogisticRegression
classifiers.append(LogisticRegression())
classifiers_names.append("LogisticRegression")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
classifiers.append(LinearDiscriminantAnalysis())
classifiers_names.append("LinearDiscriminantAnalysis")

from sklearn.naive_bayes import GaussianNB
classifiers.append(GaussianNB())
classifiers_names.append("GaussianNB")

from sklearn.svm import SVC
classifiers.append(SVC())
classifiers_names.append("SVC")

from sklearn.metrics import accuracy_score

for i in range(len(classifiers)):
    classifiers[i].fit(X_train, y_train)
    predictions = classifiers[i].predict(X_test)
    print (classifiers_names[i], accuracy_score(y_test, predictions))