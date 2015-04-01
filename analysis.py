# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
# need to add the utilities class. Want 'home' to be platform independent
from os.path import expanduser
home = expanduser("~")
# get the utilties directory (assume it lives in ~/utilities/python)
# but simple to change
path= home +"/utilities/python"
import sys
sys.path.append(path)
# import the patrick-specific utilities
import GenUtilities  as pGenUtil
import PlotUtilities as pPlotUtil
import CheckpointUtilities as pCheckUtil
import csv as csv 

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from dataToMat import ShipData
from utilIO import getData

def predict(fitter,x,yReal,rawDat,label,saveDir,colNames,saveBad=True):
    yPred = fitter.predict(x.toarray())
    cm = confusion_matrix(yReal,yPred)
    acc= accuracy_score(yReal,yPred)
    badIdx = [ i  for i,pred in enumerate(yPred) if pred != yReal[i]]
    # Show confusion matrix in a separate window
    badVals = rawDat[badIdx,:]
    if (saveBad):
        # XXX could profile? 
        np.savetxt(saveDir + 'debug_{:s}.csv'.format(label),badVals,fmt="%s",
                   delimiter=',')
    fig = pPlotUtil.figure()
    ax = plt.subplot(1,1,1)
    numCols = colNames.size
    xRange = range(numCols)
    ax.bar(xRange,fitter.coef_[0],align='center')
    ax.set_xticks(xRange)
    ax.set_xticklabels(colNames,rotation='vertical')
    plt.xlabel("coefficient name")
    plt.ylabel("beta")
    pPlotUtil.savefig(fig,saveDir + label + "coeffs")
    return acc

def analyze(dataObj,dataDir,outDir,testFile):
    nEst = np.logspace(-2,1,10)
    numTrials = len(nEst)
    acc = np.zeros((numTrials,2))
    predictDir = pGenUtil.ensureDirExists(outDir + "predictions/")
    testDat = getData(outDir,dataDir + testFile,test=True)
    # assume train/test names are the same
    colNames = np.array([str(c) for c in dataObj._trainObj ],dtype=np.object)
    for i,n in enumerate(nEst):
        fitter = LogisticRegression(C=n)
        mLabel = "_iter{:d}".format(i)
        fitter.fit(dataObj._trainX.toarray(),dataObj._trainY)
        accTrain = predict(fitter,dataObj._trainX,dataObj._trainY,
                           dataObj._trainRaw,"Train" +mLabel,outDir,colNames)
        accTest= predict(fitter,dataObj._validX,dataObj._validY,
                         dataObj._validRaw,"Valid" + mLabel,outDir,colNames)
        acc[i,:] = accTrain,accTest
        print("{:d}/{:d} : {:s}".format(i+1,numTrials,acc[i,:]))
        # save the data
        testY = fitter.predict(testDat._trainX.toarray())
        with open(predictDir+ "test{:d}.csv".format(i),"w") as fh:
            fh.write("PassengerId,Survived\n")
            for idV,pred in zip(testDat._id,testY):
                fh.write("{:d},{:d}\n".format(idV,pred))
    fig = pPlotUtil.figure()
    plt.semilogx(nEst,acc[:,0],'ro-',label="Training Set")
    plt.semilogx(nEst,acc[:,1],'kx-',label="Validation Set")
    plt.axhline(1,color='b',linestyle='--',label='max')
    plt.xlabel("Fit parameter")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.title('Accuracy vs fit parameter')
    pPlotUtil.savefig(fig,outDir + "accuracies")
