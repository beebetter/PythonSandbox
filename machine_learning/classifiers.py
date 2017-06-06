import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

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

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#classifiers.append(LinearDiscriminantAnalysis())
#classifiers_names.append("LinearDiscriminantAnalysis")

from sklearn.naive_bayes import GaussianNB
classifiers.append(GaussianNB())
classifiers_names.append("GaussianNB")

from sklearn.svm import SVC
classifiers.append(SVC())
classifiers_names.append("SVC")

data = pd.read_csv('./data/data1_0.csv')#[0:150:1]#[150:0:-1]
col = ['price', 'open', 'high', 'low']

X = data.as_matrix(col)
y = data.ix[:, 'change'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

for i in range(len(classifiers)):
    classifiers[i].fit(X_train, y_train)
    predictions = classifiers[i].predict(X_test)
    print (classifiers_names[i], accuracy_score(y_test, predictions))