import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import commun
from sklearn import preprocessing



def predictKneighClassifier(k,attribut_train,attribut_test,result_train,result_test):

    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=k).fit(attribut_train, result_train)
    yhat = neigh.predict(attribut_test)
    print("Test set Accuracy: ", metrics.accuracy_score(result_test, yhat))

def findMaxAccuracy(Ks,attribut_train,attribut_test,result_train,result_test):
    mean_acc = np.zeros((Ks - 1))

    for n in range(1, Ks):
        # Train Model and Predict
        neigh = KNeighborsClassifier(n_neighbors=n).fit(attribut_train, result_train)
        yhat = neigh.predict(attribut_test)
        mean_acc[n - 1] = metrics.accuracy_score(result_test, yhat)

    print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax() + 1)
    return mean_acc

def plotAccuracy(Ks,mean_acc):
    plt.plot(range(1, Ks), mean_acc, 'g')
    plt.legend(('Accuracy ', '+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Nabors (K)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    attribut,result,features= commun.readCsv('teleCust1000t.csv')
    attribut = preprocessing.StandardScaler().fit(attribut).transform(attribut.astype(float))
    attribut_train,attribut_test,result_train,result_test= commun.splitData(attribut, result, 0.2)
    predictKneighClassifier(4,attribut_train,attribut_test,result_train,result_test)
    mean_acc=findMaxAccuracy(10,attribut_train,attribut_test,result_train,result_test)
    plotAccuracy(10,mean_acc)
