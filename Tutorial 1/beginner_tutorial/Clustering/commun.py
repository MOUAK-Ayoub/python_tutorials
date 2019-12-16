import wget
import pandas as pd
from sklearn.model_selection import train_test_split

def importFile():

    url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv'
    wget.download(url, 'drug200.csv')

def readCsv(fileName):
    df=pd.read_csv(fileName)
    X = df[df.columns[0:df.columns.size-1]].values
    y = df[df.columns[df.columns.size-1]].values
    return X,y,df.columns[0:df.columns.size-1]

def readCsvReturnDF(fileName):
    df=pd.read_csv(fileName)
    return df

def splitData(attributs,result,testSize):
    attribut_train,attribut_test,result_train,result_test=train_test_split(attributs, result, test_size=testSize, random_state=4)
    return attribut_train,attribut_test,result_train,result_test