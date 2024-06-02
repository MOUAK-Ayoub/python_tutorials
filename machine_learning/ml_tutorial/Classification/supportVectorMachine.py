import commun
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import accuracy

def plotCellClasses(cell_df):
    ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue',label='malignant');
    cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign',ax=ax);
    plt.show()

def predictSVM(X_train, X_test, y_train):
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    return yhat

if __name__ == '__main__':
    cell_data = commun.readCsvReturnDF('cell_samples.csv')
    plotCellClasses(cell_data)
    cell_data = cell_data[pd.to_numeric(cell_data['BareNuc'], errors='coerce').notnull()]
    X = cell_data[cell_data.columns[0:cell_data.columns.size - 1]]
    y = cell_data[cell_data.columns[cell_data.columns.size - 1]].astype('int')
    X = np.asarray(X)
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = commun.splitData(X, y, 0.2)
    yhat = predictSVM(X_train, X_test, y_train)
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[2, 4])
    np.set_printoptions(precision=2)
    print (classification_report(y_test, yhat))
    plt.figure()
    accuracy.plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'], normalize=False, title='Confusion matrix')
    print(accuracy.f1Score(y_test, yhat))
    #print(accuracy.jaccardScore(y_test, yhat))