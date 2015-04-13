# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
# need to add the utilities class. Want 'home' to be platform independent
from os.path import expanduser
import sys
sys.path.append("./util/")
sys.path.append("../util/")
# import the patrick-specific utilities
import GenUtilities  as pGenUtil
import PlotUtilities as pPlotUtil
import CheckpointUtilities as pCheckUtil
import csv as csv 

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from utilIO import getData

def predict(fitter,x,yReal,rawDat,label,saveDir,colNames,fitterCoeff,
            saveBad=False,saveCoeffs=False):
    try:
        yPred = fitter.predict(x)
    except TypeError:
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
    coeffs = fitterCoeff(fitter)
    nCoeffs = coeffs.size
    xRange = range(nCoeffs)
    if( numCols == nCoeffs):
        # then we have a coefficient per feature (column), so use them for ticks
        ax.bar(xRange,coeffs,align='center')
        ax.set_xticks(xRange)
        ax.set_xticklabels(colNames,rotation='vertical')
        plt.xlabel("coefficient name")
    else:
        plt.plot(xRange,coeffs,'ro-')
        plt.xlabel("Fitter Coefficients")
    plt.ylabel("Predictor strength")
    pPlotUtil.savefig(fig,saveDir + label + "coeffs")
    return acc

def fitAndPredict(outDir,predictDir,fitter,dataObj,testDat,thisTrial,coeffFunc,
                  params):
    colNames = np.array([str(c)for i,c in enumerate(dataObj._trainObj)],
                        dtype=np.object)
    mLabel = "_iter{:d}".format(thisTrial)
    try:
        fitter.fit(dataObj._trainX,dataObj._trainY)
    except TypeError:
        # Boosting needs a dense matrix... :-(
        fitter.fit(dataObj._trainX.toarray(),dataObj._trainY)
    accTrain = predict(fitter,dataObj._trainX,dataObj._trainY,
                       dataObj._trainRaw,"Train" +mLabel,outDir,
                       colNames,coeffFunc)
    # only  get the accuracy for the validation set if it exists
    if (dataObj._valid > 0.):
        accVld= predict(fitter,dataObj._validX,dataObj._validY,
                        dataObj._validRaw,"Valid" + mLabel,outDir,
                        colNames,coeffFunc)
    else:
        accVld = -1
    # save the data
    thisCoef =params[thisTrial]
    try:
        testY = fitter.predict(testDat._trainX)
    except TypeError:
    # Boosting!
        testY = fitter.predict(testDat._trainX.toarray())
    with open(predictDir+ "test{:d}_{:.3g}.csv".format(thisTrial,thisCoef),
              "w")as fh:
        fh.write("PassengerId,Survived\n")
        for idV,pred in zip(testDat._id,testY):
            fh.write("{:d},{:d}\n".format(idV,pred))
    return accTrain,accVld

def plotAccuracies(outDir,label,accMean,accStd,fitParam):
    fig = pPlotUtil.figure()
    plt.errorbar(fitParam,accMean[:,0],accStd[:,0],fmt='ro-',
                 label="Training Set")
    plt.errorbar(fitParam,accMean[:,1],accStd[:,1],fmt='kx-',
                 label="Validation Set")
    plt.xscale('log', nonposy='clip')
    plt.axhline(1,color='b',linestyle='--',label='max')
    plt.xlabel("Fit parameter")
    plt.ylabel("Accuracy")
    plt.xlim([min(fitParam)*0.7,max(fitParam)*1.3])
    plt.legend(loc='best')
    plt.title('Accuracy vs fit parameter for fitter: {:s}'.format(label))
    pPlotUtil.savefig(fig,outDir + "accuracies")


def getTrialMeanStd(outDir,predictDir,dataObj,testDat,i,nTrials,
                    fitterCoeff,createFitter,params):
    fitter = createFitter(i)
    accTrain = []
    accValid = []
    for j in range(nTrials):
        # re-shuffle the data for the next run
        dataObj._shuffleAndPopulate()
        # get the accuracy for this fit and data object (Train/validation)
        trainTmp,vldTmp=  fitAndPredict(outDir,predictDir,fitter,
                                        dataObj,testDat,i,fitterCoeff,params)
        print("Trial {:d}/{:d} (repeat {:d}/{:d}) : {:.3f}/{:.3f}".\
              format(i+1,len(params),j+1,nTrials,trainTmp,vldTmp))

        accTrain.append(trainTmp)
        accValid.append(vldTmp)
    return [np.mean(accTrain),np.mean(accValid)],\
        [np.std(accTrain),np.std(accValid)]
    
def getAllTrials(params,outDir,predictDir,dataObj,testDat,nTrials,
                 fitterCoeff,createFitter):
    nParams = len(params)
    meanTrainValid = np.zeros((nParams,2))
    stdTrainValid = np.zeros((nParams,2))
    # get the parameters for fitting
    # assume train/test names are the same
    for i,n in enumerate(params):
        meanTrainValid[i,:],stdTrainValid[i,:] = getTrialMeanStd(outDir,\
        predictDir,dataObj,testDat,i,nTrials,fitterCoeff,createFitter,params)
    return meanTrainValid,stdTrainValid

def analyze(dataObj,dataDir,outDir,testFile,createFitter,fitterParams,
            fitterCoeff,label,dataClass,nTrials):
    # 'createfitter' takes in the current iteration 'i', and returns a fitter
    # e.g. "return LogisticRegression(C=[10,30,40][i])"
    # 'fitterParams' gives the value of the parameters used at each iter.
    predictDir = pGenUtil.ensureDirExists(outDir + "predictions/")
    testDat = getData(dataClass,outDir,dataDir + testFile,test=True)
    params = fitterParams()
    fName = outDir+"accuracyTrials_{:d}repeats_{:d}params.pkl".format(nTrials,
                                                            len(params))
    means,std=pCheckUtil.getCheckpoint(fName,getAllTrials,
            True,params,outDir,predictDir,dataObj,testDat,nTrials,
                                       fitterCoeff,createFitter)
    # plot the accuracies versus the fit parameter.
    plotAccuracies(outDir,label,means,std,params)
    return means,std
