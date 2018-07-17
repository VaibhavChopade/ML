import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics.classification import accuracy_score, confusion_matrix,\
    classification_report
import numpy as np

# aa = ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol","quality"]
# 
# rwine= pandas.read_csv("winequality-red.csv")
# 
# dframe = pandas.DataFrame(rwine)

url = "flowers.csv"
names = ["sepal-length","sepal-Width","petal-length", "petal-Width","class"]
dataset = pandas.read_csv(url,names=names)

array = dataset.values

X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
scoring = 'accuracy'
X_train,X_validation,Y_train,Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)

# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVC', SVC()))
# # evaluate each model in turn
# results = []
# names = []
# for name,model in models:
#     kfold = model_selection.KFold(n_splits=10,random_state=seed)
#     cv_results = model_selection.cross_val_score(model, X_train , Y_train, cv=kfold, scoring=scoring)
#     results.append(name)
#     msg="%s:%f (%f)" %(name,cv_results.mean(),cv_results.std())   
#     print msg

knn = KNeighborsClassifier()
#knn = SVC()
knn.fit(X_train,Y_train)
print X_train,Y_train
print type(X_validation)
temp = [[6.4,2.9,4.3,1.3]]
#temp = [[6.4,2.9,8.3,1]]
a = np.asarray(temp)
print type(a)
prediction = knn.predict(a)
print accuracy_score([["Iris-versicolor"]],prediction)



#predictions = knn.predict(X_validation)
# print accuracy_score(Y_validation,predictions)
# print confusion_matrix(Y_validation,predictions)
# print classification_report(Y_validation,predictions)
