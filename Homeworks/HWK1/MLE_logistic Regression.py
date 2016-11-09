# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:08:35 2016

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


def logisticPredictProb(w,xTranspose):
    return 1/(1+np.exp(-((w.dot(xTranspose)))))

def logisticPredict(w,xTranspose):
    if logisticPredictProb(w,xTranspose)>=0.5:
        return 1
    else:
        return 0

def irls(train,nbIteration):
    
    #getting data to the right dimension.
    
    x = train[[0,1]]
    N = len(x)
    x = np.append(x, np.ones((N,1)),axis=1)
    y = train[[2]]
    # we will be extensively use the transposee better compute it here:
    w = np.matrix('0.0,0.0,0.0')
#    lW = [w]
#    lH = []
#    lG = []
    xTranspose = np.transpose(x)
    for i in range (nbIteration):
        
        #for each step we compute nu then
        #the gradient and the inverse of the hessian
        #nu et w are a row vectors not column vectors here
        
        nu = np.array(logisticPredictProb(w,xTranspose))
        nuDiag = np.diag((nu*(1-nu))[0]) 
        gradient = xTranspose.dot(y-np.transpose(nu))
        hessianInv = np.linalg.inv(-xTranspose.dot(nuDiag).dot(x))
        
        #the Newton's method update
        w = w - np.transpose(hessianInv.dot(gradient))
#        lW.append(w)
#        lH.append(hessianInv)
#        lG.append(gradient)
    return w
  

def logisticRegressionPlot(train,w):
    plt.scatter(train[0],train[1], c=train[2], marker='o',cmap=plt.cm.autumn)
    plt.show()
    rangeX0 = np.linspace(-10,10)
    rangeX1 = np.linspace(-10,10)[:, None]
    proba= np.array(map(lambda x1: map(lambda x0:
        logisticPredictProb(w,[x0,x1,1])[0,0],rangeX0),rangeX1))
    plt.contour(rangeX0,rangeX1.ravel(),proba,[0.5])
    plt.show()
    return proba

def logisticPredictError(w,train,test):
    xTrain = train[[0,1]]
    N = len(xTrain)
    xTrain = np.append(xTrain, np.ones((N,1)),axis=1)
    yTrain = train[2]
    xTest= test[[0,1]]
    N = len(xTest)
    xTest= np.append(xTest, np.ones((N,1)),axis=1)
    yTest = test[2]
    trainPredict =map(lambda x: logisticPredict(w,x), xTrain)
    testPredict  = map(lambda x: logisticPredict(w,x), xTest)
    errorTrain = np.mean(trainPredict != yTrain)*100
    errorTest = np.mean(testPredict != yTest)*100
    return errorTrain, errorTest,trainPredict,testPredict

w_A = irls(trainA,100)    
w_B = irls(trainB,100)    
w_C = irls(trainC,100)    
w = irls(train,100)  

prob_A= logisticRegressionPlot(trainA,w_A)
prob_B= logisticRegressionPlot(trainB,w_B)
prob_C= logisticRegressionPlot(trainC,w_C)
prob= logisticRegressionPlot(train,w)


eTrain_A,eTest_A,trainPredict_A,testPredict_A = logisticPredictError(w_A,trainA,testA)  
eTrain_B,eTest_B,trainPredict_B,testPredict_B = logisticPredictError(w_B,trainB,testB)
eTrain_C,eTest_C,trainPredict_C,testPredict_C = logisticPredictError(w_C,trainC,testC)
eTrain,eTest,trainPredict,testPredict= logisticPredictError(w,train,test)
    
