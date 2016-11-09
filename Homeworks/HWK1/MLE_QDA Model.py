# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:50:58 2016

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
#declare different parameters and a variable that will store:
#row slices in for loops.

testA=pd.read_csv("classificationA.test", header=None, delim_whitespace=True, decimal='.')
testB=pd.read_csv("classificationB.test", header=None, delim_whitespace=True, decimal='.')
testC=pd.read_csv("classificationC.test", header=None, delim_whitespace=True, decimal='.')
test=pd.concat([testA,testB,testC])


def qdaModel(train):
    #getting the covariance matrix:
    mu0=np.matrix('0.0,0.0')
    mu1=np.matrix('0.0,0.0')
    sigma0=np.matrix('0.0,0.0;0.0,0.0')
    sigma1=np.matrix('0.0,0.0;0.0,0.0')
    N1=0
    N=len(train)
    rowSlice=np.matrix('0.0,0.0,0.0')
    
    #getting mu0, mu1, N1 and pi:
    
    for i in range(N):
        rowSlice = train[i:i+1]
        if rowSlice[2].values==1:
            N1 += 1 
            mu1 += rowSlice[[0,1]]
        else:
            mu0 += rowSlice[[0,1]]
            
    pi = N1/float(N)
    mu0 /= (N-N1)
    mu1 /= N1
    diff0=np.matrix('0.0,0.0;0.0,0.0')
    diff1=np.matrix('0.0,0.0;0.0,0.0')    
    for i in range(N):    
        rowSlice = train[i:i+1]
        if rowSlice[2].values==1:
            diff1 = rowSlice[[0,1]]-mu1 
            sigma1 += np.transpose(diff1).dot(diff1)
        else:
            diff0 = rowSlice[[0,1]]-mu0            
            sigma0 += np.transpose(diff0).dot(diff0)
    sigma1 /= N1
    sigma0 /= (N-N1)
    return pi,mu0,mu1,sigma0,sigma1
    



def qdaPredictProb(x, mu0, mu1, sigma0Inv,sigma1Inv, offset):
    return 1/(1+offset*np.exp((x-mu1).dot(sigma1Inv).dot(np.transpose(x-mu1))
    -(x-mu0).dot(sigma0Inv).dot(np.transpose(x-mu0))))

def qdaPredict(x,mu0,mu1,sigma0Inv,sigma1Inv,offset):
    if qdaPredictProb(x,mu0,mu1,sigma0Inv,sigma1Inv,offset)>=0.5:
        return 1
    else:
        return 0 
        
def qdaPlot(pi_v,mu0_v,mu1_v,sigma0_v, sigma1_v, train_v):
    #now defining p(y=1|x)=f(x)
    #we compute the inverse of sigma and the factor (1-pi)/pi so as to
    #optimize the plotting:
    sigma0Inv = np.linalg.inv(sigma0_v)
    sigma1Inv = np.linalg.inv(sigma1_v)
    offset = (1-pi_v)/(pi_v)
    #plotting the data points:

    plt.scatter(train_v[0],train_v[1], c=train_v[2], marker='o',cmap=plt.cm.autumn)
    plt.show()
        
    #plotting the implicit function f(x)=0.5, we set the range of x0 and x1
    #rangeX0 and rangeX1, then we compute f(x0,x1) for all combinations of
    #x0 and x1, then we draw the line using contour by comparing the 
    #oriba array to 0.5.
        
    rangeX0 = np.linspace(-10,10)
    rangeX1 = np.linspace(-10,10)[:, None]
    proba= np.array(map(lambda x1: map(lambda x0: qdaPredictProb(
    [x0,x1], mu0_v, mu1_v, sigma0Inv,sigma1Inv, offset)[0,0], rangeX0),rangeX1))
    plt.contour(rangeX0,rangeX1.ravel(),proba,[0.5])
    plt.show()
    return proba
    
    
def qdaPredictError(pi,mu0,mu1,sigma0,sigma1,train,test):
    sigma0Inv = np.linalg.inv(sigma0)
    sigma1Inv = np.linalg.inv(sigma1)
    offset = (1-pi)/(pi)
    xTrain = train[[0,1]]
    yTrain = train[2]
    xTest = test[[0,1]]
    yTest = test[2]
    trainPredict = xTrain.apply(lambda x: qdaPredict([x[0],x[1]],mu0,mu1,sigma0Inv,sigma1Inv,offset), axis=1)
    testPredict  = xTest.apply(lambda x: qdaPredict([x[0],x[1]],mu0,mu1,sigma0Inv,sigma1Inv,offset), axis=1) 
    errorTrain = np.mean(trainPredict != yTrain)*100
    errorTest = np.mean(testPredict != yTest)*100
    return errorTrain, errorTest,trainPredict,testPredict
    
    
    
pi_A, mu0_A, mu1_A, sigma0_A, sigma1_A= qdaModel(trainA)
pi_B, mu0_B, mu1_B, sigma0_B, sigma1_B = qdaModel(trainB)
pi_C, mu0_C, mu1_C, sigma0_C, sigma1_C = qdaModel(trainC)
pi, mu0, mu1, sigma0, sigma1 = qdaModel(train)    


prob_A= qdaPlot(pi_A, mu0_A, mu1_A, sigma0,sigma1_A, trainA)
prob_B= qdaPlot(pi_B, mu0_B, mu1_B, sigma0,sigma1_B, trainB)
prob_C= qdaPlot(pi_C, mu0_C, mu1_C, sigma0,sigma1_C, trainC)
prob= qdaPlot(pi, mu0, mu1, sigma0,sigma1, train)

eTrain_A,eTest_A,trainPredict_A,testPredict_A = qdaPredictError(pi_A,mu0_A,mu1_A,sigma0,sigma1_A,trainA,testA)  
eTrain_B,eTest_B,trainPredict_B,testPredict_B = qdaPredictError(pi_B,mu0_B,mu1_B,sigma0,sigma1_B,trainB,testB)
eTrain_C,eTest_C,trainPredict_C,testPredict_C = qdaPredictError(pi_C,mu0_C,mu1_C,sigma0,sigma1_C,trainC,testC)
eTrain,eTest,trainPredict,testPredict= qdaPredictError(pi,mu0,mu1,sigma0,sigma1,train,test)
