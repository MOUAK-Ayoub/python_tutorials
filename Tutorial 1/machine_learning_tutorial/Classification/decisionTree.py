import commun
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
import numpy as np


def transformData(data):
    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['F', 'M'])
    data[:, 1] = le_sex.transform(data[:, 1])

    le_BP = preprocessing.LabelEncoder()
    le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
    data[:, 2] = le_BP.transform(data[:, 2])

    le_Chol = preprocessing.LabelEncoder()
    le_Chol.fit(['NORMAL', 'HIGH'])
    data[:, 3] = le_Chol.transform(data[:, 3])

    return data

def decisionTreeClassifier(X_trainset, X_testset, y_trainset, y_testset):
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    drugTree.fit(X_trainset, y_trainset)
    predTree = drugTree.predict(X_testset)

    print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
    return drugTree

def plotResults(filename,featureNames,drugTree):
    dot_data = StringIO()

    out = tree.export_graphviz(drugTree, feature_names=featureNames, out_file=dot_data,
                               class_names=np.unique(y_trainset), filled=True, special_characters=True, rotate=False)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    graph.write_png(filename)
    img = mpimg.imread(filename)
    plt.figure(figsize=(100, 200))
    plt.imshow(img, interpolation='nearest')

if __name__ == '__main__':
    data, y, columns= commun.readCsv('drug200.csv')
    data = transformData(data)
    X_trainset, X_testset, y_trainset, y_testset= commun.splitData(data, y, 0.3)
    drugTree=decisionTreeClassifier(X_trainset, X_testset, y_trainset, y_testset)
    plotResults('drugtree.png', columns, drugTree)


