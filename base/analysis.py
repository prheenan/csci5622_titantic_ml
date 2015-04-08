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
    nCoeffs = len(coeffs)
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

def fitAndPredict(outDir,predictDir,fitter,dataObj,testDat,thisTrial,coeffFunc):
    colNames = np.array([str(c) for c in dataObj._trainObj ],dtype=np.object)
    mLabel = "_iter{:d}".format(thisTrial)
    fitter.fit(dataObj._trainX.toarray(),dataObj._trainY)
    accTrain = predict(fitter,dataObj._trainX,dataObj._trainY,
                       dataObj._trainRaw,"Train" +mLabel,outDir,
                       colNames,coeffFunc)
    accVld= predict(fitter,dataObj._validX,dataObj._validY,
                     dataObj._validRaw,"Valid" + mLabel,outDir,
                    colNames,coeffFunc)
    print("Trial {:d} : {:.3f}/{:.3f}".format(thisTrial+1,
                                              accTrain,accVld))
    # save the data
    testY = fitter.predict(testDat._trainX.toarray())
    with open(predictDir+ "test{:d}.csv".format(thisTrial),"w") as fh:
        fh.write("PassengerId,Survived\n")
        for idV,pred in zip(testDat._id,testY):
            fh.write("{:03d},{:03d}\n".format(idV,pred))
    return accTrain,accVld

def plotAccuracies(outDir,label,acc,fitParam):
    fig = pPlotUtil.figure()
    plt.semilogx(fitParam,acc[:,0],'ro-',label="Training Set")
    plt.semilogx(fitParam,acc[:,1],'kx-',label="Validation Set")
    plt.axhline(1,color='b',linestyle='--',label='max')
    plt.xlabel("Fit parameter")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.title('Accuracy vs fit parameter for fitter: {:s}'.format(label))
    pPlotUtil.savefig(fig,outDir + "accuracies")


def analyze(dataObj,dataDir,outDir,testFile,createFitter,fitterParams,
            fitterCoeff,label,dataClass):
    # 'createfitter' takes in the current iteration 'i', and returns a fitter
    # e.g. "return LogisticRegression(C=[10,30,40][i])"
    # 'fitterParams' gives the value of the parameters used at each iter.
    predictDir = pGenUtil.ensureDirExists(outDir + "predictions/")
    testDat = getData(dataClass,outDir,dataDir + testFile,test=True)
    accTrain = []
    accValid = []
    # get the parameters for fitting
    params = fitterParams()
    # assume train/test names are the same
    for i,n in enumerate(params):
        fitter = createFitter(i)
        # get the accuracy for this fit and data object (Train/validation)
        trainTmp,vldTmp=  fitAndPredict(outDir,predictDir,fitter,
                                        dataObj,testDat,i,fitterCoeff)
        accTrain.append(trainTmp)
        accValid.append(vldTmp)
    # create a single, two column array. columns are train/valid,
    # rows correspond to a single iteration
    acc = np.array([accTrain,accValid]).T
    # plot the accuracies versus the fit parameter.
    plotAccuracies(outDir,label,acc,params)

