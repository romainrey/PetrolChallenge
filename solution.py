# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 13:53:29 2017

@author: Romain Rey
"""

import numpy as np
import pandas
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

def score_function(y_true, y_pred):
    """Gives our prediction score, assuming that we have y_true"""
    return roc_auc_score(y_true, y_pred)
    
def printTitle(s):
    """Useful to make some nice print"""
    print("\n\n########## "+s+" ##########\n")
    
def importData(path, title):
    """imports csv file into matrix, ignoring the first column (id in our case)"""
    importedData = pandas.read_csv(path,sep=";",usecols=[1,2]+[i for i in range(84,123)]).as_matrix()
    printTitle(title)
    print(importedData)
    return importedData

def importLabels():
    """import csv file into an array, ignoring the first column (id in our case)"""
    importedLabels = pandas.read_csv("challenge_output_data_training_file_predict_the_crude_oil_production_trend.csv",sep=";",usecols=range(1,2)).as_matrix()
    importedLabels = np.concatenate(importedLabels)
    printTitle("Label of Training Data")
    print(importedLabels)
    return importedLabels
    
def cleanData(dataToClean):
    """We remplace the NaN elements in the dataToClean by 0"""
    for i in range(len(dataToClean)):
        for j in range(len(dataToClean[0])):
            if np.isnan(dataToClean[i][j]):
                dataToClean[i][j]=0

def constructCountry(matrix,id):
    array = []
    for i in range(len(matrix)):
        if matrix[i][1]==id:
            array.append(i)
    return matrix.choose(choices=array)
    
def normalizeData(dataToNormalize, title):
    """We normalize data from dataToNormalize"""
    numberRows = len(dataToNormalize)
    numberColumns = len(dataToNormalize[0])
    # We do not normalize teh country and the mounth so we start at 2
    for i in range(2,numberColumns):
        if i!=1:
            column = dataToNormalize[:,i]
            # We calcul the average and the standart deviation
            average = np.average(column)
            std = np.std(column)
            for j in range(numberRows):
                oldCoeff=dataToNormalize[j][i]
                # We bring back minimum to average-3*std and maximum to average+3*std
                # This will reduce the influence of the extremes
                oldCoeff=min(oldCoeff,average+3*std)
                oldCoeff=max(oldCoeff,average-3*std)
                # Finally we normalize each coeff
                newCoeff=(oldCoeff-average)/(6*std)
#                newCoeff=giveGrade(oldCoeff,average,std)
                dataToNormalize[j][i]=newCoeff
    printTitle(title)
    print(dataToNormalize)
    
def giveGrade(coeff,average,sigma):
    a=average-3*sigma
    grade=0
    while a<coeff:
        a+=sigma/2
        grade+=1
    return grade/100
        
    
    
#### IMPLEMENTATION ####    
    
    
    
########### Importing
dataI = importData("Train.csv", "Training Data extracted")
dataTestI = importData("Test.csv", "Test Data extracted")
data = dataI[:8000]
dataTest = dataI[8001:]
    
labelTrainI = importLabels()
labelTrain = labelTrainI[:8000]
labelTest = labelTrainI[8001:]


########### Cleaning
cleanData(data)
cleanData(dataTest) 
    
########### Normalizing Data
normalizeData(data,"Training Data normalized")
normalizeData(dataTest,"Test Data normalized")


########### Diminution of the dimension with PCA algorithm          
# We choose 2 final dimensions (because after experimentation it gives best results)
pca = PCA(n_components=2)
## We fit on our Training data
pca.fit(data)
## Then transform both Training and Test Data
printTitle("PCA Training Data")
dataReduced = pca.transform(data)
print(dataReduced)
printTitle("PCA Test Data")
dataTestReduced = pca.transform(dataTest)
print(dataTestReduced)
#
#dataReduced = data
#dataTestReduced = dataTest
########### Training of the SVM
# We choose a Support Vector Regression, with an optimal number of support vector
# (found by experimentation) and we train on training data
clf = svm.NuSVR(nu=0.55)
clf.fit(dataReduced, labelTrain)          

########### Predicting labels
printTitle("Predicted Labels")                
labelPredicted=clf.predict(dataTestReduced)
# We correct the prediction: if there are negative probas they become 0
# and probas over 1 become 1
for i in range(len(labelPredicted)):    
    value = labelPredicted[i]
    if value<0:    
        labelPredicted[i]=0
    elif value>1:
        labelPredicted[i]=1
print(labelPredicted)

########### Making answer csv
answer = []
# We import the IDs
TestID = pandas.read_csv("Test.csv",sep=";",usecols=range(0,1)).as_matrix()
#We create the answer array
#for i in range(len(labelPredicted)):
#    answer.append([TestID[i][0],labelPredicted[i]])
    
print(score_function(labelTest,labelPredicted))
# Then we create the answer csv calles "answer.csv"
#dataFrame = pandas.DataFrame(data=answer)
#dataFrame.to_csv(path_or_buf="answer2.csv",header=["ID","Target"],index=False)
#printTitle("File answer.csv created")
    

