import commun
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import sys
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import accuracy

def logisticRegressionPrediction(X_train, y_train, X_test):
    LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
    yhat = LR.predict(X_test)
    yhat_prob = LR.predict_proba(X_test)
    return yhat, yhat_prob

if __name__ == '__main__':
    print sys.path
    data, churn, features = commun.readCsv('ChurnData.csv')
    churn = churn.astype('int')
    data = preprocessing.StandardScaler().fit(data).transform(data)
    X_trainset, X_testset, y_trainset, y_testset = commun.splitData(data, churn, 0.2)
    yhat, yhat_prob = logisticRegressionPrediction(X_trainset, y_trainset, X_testset)
    cnf_matrix = confusion_matrix(y_testset, yhat, labels=[1,0])
    np.set_printoptions(precision=2)

    plt.figure()
    accuracy.plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'])
    print accuracy.logloss(y_testset, yhat_prob)
