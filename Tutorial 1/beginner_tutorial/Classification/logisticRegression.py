import commun
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import sys
import pkgutil

def logisticRegressionPrediction(X_train, y_train, X_test):
    LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
    yhat = LR.predict(X_test)
    yhat_prob = LR.predict_proba(X_test)


if __name__ == '__main__':
    print sys.path
    search_path = ['.']  # set to None to see all modules importable from sys.path
    all_modules = [x[1] for x in pkgutil.iter_modules(path=search_path)]
    print(all_modules)
    # data, churn, features = commun.readCsv('ChurnData.csv')
    # churn = churn.astype('int')
    # data = preprocessing.StandardScaler().fit(data).transform(data)
    # X_trainset, X_testset, y_trainset, y_testset = commun.splitData(data, churn, 0.2)
    # logisticRegressionPrediction(X_trainset, y_trainset, X_testset)

