# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:17:04 2016

# -*- coding: utf-8 -*-

@author: Su YANG
"""


for name in dir():
    if not name.startswith('_'):
        del globals()[name]
        
        
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

trainA=pd.read_csv("classificationA.train", header=None, delim_whitespace=True, decimal='.')
trainB=pd.read_csv("classificationB.train", header=None, delim_whitespace=True, decimal='.')
trainC=pd.read_csv("classificationC.train", header=None, delim_whitespace=True, decimal='.')
train=pd.concat([trainA,trainB,trainC])

testA=pd.read_csv("classificationA.test", header=None, delim_whitespace=True, decimal='.')
testB=pd.read_csv("classificationB.test", header=None, delim_whitespace=True, decimal='.')
testC=pd.read_csv("classificationC.test", header=None, delim_whitespace=True, decimal='.')
test=pd.concat([testA,testB,testC])



#initialization

def linearRegression(train):
    
    
    #getting data to the right dimension.
    # we will be extensively use the transposee better compute it here:
    
    x = train[[0,1]]
    N = len(x)
    x = np.append(x, np.ones((N,1)),axis=1)
    y = train[[2]]
    xTranspose = np.transpose(x)
    w =  np.linalg.inv(xTranspose.dot(x)).dot(xTranspose).dot(y)
    sigma2 = y-x.dot(w)
    sigma2 = np.sum(sigma2*sigma2)/N
    return w, sigma2

     

def linearPredictProb(w,x,sigma2):
    error = 1-x.dot(w)
    output = np.exp(-error*error/(2*sigma2))/np.sqrt(2*np.pi*sigma2)
    return output

def linearPredict(w,x,sigma2):
    if linearPredictProb(w,x,sigma2).values>=0.5:
        return 1
    else:
        return 0
        
def linearRegressionPlot(train,w,sigma2):
    plt.scatter(train[0],train[1], c=train[2], marker='o',cmap=plt.cm.autumn)
    plt.show()
    rangeX0 = np.linspace(-10,10)
    rangeX1 = np.linspace(-10,10)[:, None]
    proba= np.array(map(lambda x1: map(lambda x0: 
        linearPredictProb(w,np.array([x0,x1,1]),sigma2)[2],rangeX0),rangeX1))
    plt.contour(rangeX0,rangeX1.ravel(),proba,[0.5])
    plt.show()
    return proba #Here we have proba density and not probabilities like in 
    #other two cases, therefore we have bigger than 1 numbers (essentially
    #it means that Gaussian is not suited for such a discrete model)



def linearPredictError(w,sigma2,train,test):
    xTrain = train[[0,1]]
    N = len(xTrain)
    xTrain = np.append(xTrain, np.ones((N,1)),axis=1)
    yTrain = train[2]
    xTest= test[[0,1]]
    N = len(xTest)
    xTest= np.append(xTest, np.ones((N,1)),axis=1)
    yTest = test[2]
    trainPredict =map(lambda x: linearPredict(w,x,sigma2), xTrain)
    testPredict  = map(lambda x: linearPredict(w,x,sigma2), xTest)
    errorTrain = np.mean(trainPredict != yTrain)*100
    errorTest = np.mean(testPredict != yTest)*100
    return errorTrain, errorTest,trainPredict,testPredict
    
w_A, sigma2_A = linearRegression(trainA)
w_B, sigma2_B = linearRegression(trainB)   
w_C, sigma2_C = linearRegression(trainC)   
w, sigma2 = linearRegression(train)  

prob_A= linearRegressionPlot(trainA,w_A,sigma2_A)
prob_B= linearRegressionPlot(trainB,w_B,sigma2_B)
prob_C= linearRegressionPlot(trainC,w_C,sigma2_C)
prob= linearRegressionPlot(train,w,sigma2)


eTrain_A,eTest_A,trainPredict_A,testPredict_A = linearPredictError(w_A,sigma2_A,trainA,testA)  
eTrain_B,eTest_B,trainPredict_B,testPredict_B = linearPredictError(w_B,sigma2_B,trainB,testB)
eTrain_C,eTest_C,trainPredict_C,testPredict_C = linearPredictError(w_C,sigma2_C,train,testC)
eTrain,eTest,trainPredict,testPredict= linearPredictError(w,sigma2,train,test)
    