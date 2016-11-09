# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:39:57 2016

@author: Su YANG
"""

for name in dir():
    if not name.startswith('_'):
        del globals()[name]
        
        
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#pd.options.display.float_format = '{:20,.2f}'.format

#getting data from 3 files and putting it in a single dataframe:


        
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


def generativeModel(train):
    #getting the covariance matrix:
    mu0=np.matrix('0.0,0.0')
    mu1=np.matrix('0.0,0.0')
    sigma=np.matrix('0.0,0.0;0.0,0.0')
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
    diff=np.matrix('0.0,0.0;0.0,0.0')
    for i in range(N):    
        rowSlice = train[i:i+1]
        if rowSlice[2].values==1:
            diff = rowSlice[[0,1]]-mu1 
        else:
            diff = rowSlice[[0,1]]-mu0            
        sigma += np.transpose(diff).dot(diff)
        
    sigma /= N
    return pi,mu0,mu1,sigma


def generativePredictProb(x, mu0, mu1, sigmaInv, offset):
    return 1/(1+offset*np.exp((x-mu1).dot(sigmaInv).dot(np.transpose(x-mu1))
    -(x-mu0).dot(sigmaInv).dot(np.transpose(x-mu0))))

def generativePredict(x,mu0,mu1,sigmaInv,offset):
    if generativePredictProb(x,mu0,mu1,sigmaInv,offset)>=0.5:
        return 1
    else:
        return 0 
        
def generativePlot(pi_v,mu0_v,mu1_v,sigma_v, train_v):
    #now defining p(y=1|x)=f(x)
    #we compute the inverse of sigma and the factor (1-pi)/pi so as to
    #optimize the plotting:
    sigmaInv = np.linalg.inv(sigma_v)
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
    proba= np.array(map(lambda x1: map(lambda x0: generativePredictProb(
    [x0,x1], mu0_v, mu1_v, sigmaInv, offset)[0,0], rangeX0),rangeX1))
    
    plt.contour(rangeX0,rangeX1.ravel(),proba,[0.5])
    plt.show()
    return proba
    
    #one can check that the lower space delimited by the drawen line 
    #is the space of higher probability for p(y=1|x) by setting the implicit
    #function right side to a number higher than 0.5 
    


def generativePredictError(pi,mu0,mu1,sigma,train,test):
    sigmaInv = np.linalg.inv(sigma)
    offset = (1-pi)/(pi)
    xTrain = train[[0,1]]
    yTrain = train[2]
    xTest = test[[0,1]]
    yTest = test[2]
    trainPredict = xTrain.apply(lambda x: generativePredict([x[0],x[1]],mu0,mu1,sigmaInv,offset), axis=1)
    testPredict  = xTest.apply(lambda x: generativePredict([x[0],x[1]],mu0,mu1,sigmaInv,offset), axis=1) 
    errorTrain = np.mean(trainPredict != yTrain)*100
    errorTest = np.mean(testPredict != yTest)*100
    return errorTrain, errorTest,trainPredict,testPredict
    


pi_A, mu0_A, mu1_A, sigma_A = generativeModel(trainA)
pi_B, mu0_B, mu1_B, sigma_B = generativeModel(trainB)
pi_C, mu0_C, mu1_C, sigma_C = generativeModel(trainC)
pi, mu0, mu1, sigma = generativeModel(train)

    
prob_A= generativePlot(pi_A, mu0_A, mu1_A, sigma_A, trainA)
prob_B= generativePlot(pi_B, mu0_B, mu1_B, sigma_B, trainB)
prob_C= generativePlot(pi_C, mu0_C, mu1_C, sigma_C, trainC)
prob= generativePlot(pi, mu0, mu1, sigma, train)


eTrain_A,eTest_A,trainPredict_A,testPredict_A = generativePredictError(pi_A,mu0_A,mu1_A,sigma_A,trainA,testA)  
eTrain_B,eTest_B,trainPredict_B,testPredict_B = generativePredictError(pi_B,mu0_B,mu1_B,sigma_B,trainB,testB)
eTrain_C,eTest_C,trainPredict_C,testPredict_C = generativePredictError(pi_C,mu0_C,mu1_C,sigma_C,trainC,testC)
eTrain,eTest,trainPredict,testPredict= generativePredictError(pi,mu0,mu1,sigma,train,test)

# plt.scatter(testC[0],testC[1], c=testC[2], marker='o',cmap=plt.cm.autumn)
# plt.show()