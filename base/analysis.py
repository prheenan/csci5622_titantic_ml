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

def profileLosers(saveDir,label,yPred,yReal,rawDat,dataClass):
    badIdx = [ i  for i,pred in enumerate(yPred) if pred != yReal[i]]
    badVals = rawDat[badIdx,:]
    np.savetxt(saveDir + 'debug_{:s}.csv'.format(label),badVals,fmt="%s",
               delimiter=',')
    badObj = dataClass(saveDir,rawDat,valid=0.0,test=False,
                       profileName=saveDir + label)
    

def predict(fitter,x,yReal,rawDat,label,saveDir,colNames,fitterCoeff,objClass,
            featureObjects,saveBad=False,saveCoeffs=False,plot=True):
    try:
        yPred = fitter.predict(x)
    except TypeError:
        yPred = fitter.predict(x.toarray())
    cm = confusion_matrix(yReal,yPred)
    acc= accuracy_score(yReal,yPred)
    # Show confusion matrix in a separate window
    if (saveBad):
        # XXX could profile?
        profileLosers(saveDir,label,yPred,yReal,rawDat,objClass)
    if (plot):
        fig = pPlotUtil.figure()
        ax = plt.subplot(1,1,1)
        numCols = colNames.size
        coeffs = fitterCoeff(fitter)
        nCoeffs = coeffs.size
        xRange = range(nCoeffs)
        saveName = saveDir + label + "coeffs"
        sortIdx = np.argsort(coeffs)[::-1]
        sortedCoeffs = coeffs[sortIdx]
        sortedNames = colNames[sortIdx]
        sortedFeatures = [featureObjects[s] for s in sortIdx]
        stacked = np.vstack((sortedNames,sortedCoeffs)).T
        np.savetxt(saveName,stacked,fmt=["%s","%.3g"],delimiter="\t")
        maxToPlot = min(numCols//2,25) # on each side

        if( numCols == nCoeffs):
    # then we have a coefficient per feature (column), so use them for ticks
            coeffsToPlot = list(sortedCoeffs[:maxToPlot]) + \
                           list(sortedCoeffs[-maxToPlot:])
            labelsToPlot = list(sortedNames[:maxToPlot]) +\
                           list(sortedNames[-maxToPlot:])
            featuresPlotted = list(sortedFeatures[:maxToPlot]) + \
                              list(sortedFeatures[-maxToPlot:])
            xToPlot = range(len(coeffsToPlot))
            ax.bar(xToPlot,coeffsToPlot,align='center')
            ax.set_xticks(xToPlot)
            ax.set_xticklabels(labelsToPlot,rotation='vertical')
            plt.xlabel("coefficient name")
        else:
            plt.plot(xRange,coeffs,'ro-')
            plt.xlabel("Fitter Coefficients")
            plt.ylabel("Predictor strength")
        pPlotUtil.savefig(fig,saveName)
    return acc

def fitAndPredict(outDir,predictDir,fitter,dataObj,testDat,thisTrial,coeffFunc,
                  params,plot,dataClass):
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
                       colNames,coeffFunc,dataClass,dataObj._trainObj,plot=plot)
    # only  get the accuracy for the validation set if it exists
    if (dataObj._valid > 0.):
        accVld= predict(fitter,dataObj._validX,dataObj._validY,
                        dataObj._validRaw,"Valid" + mLabel,outDir,
                        colNames,coeffFunc,dataClass,dataObj._validObj,
                        plot=plot)
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
                    fitterCoeff,createFitter,params,plot,dataClass):
    fitter = createFitter(i)
    accTrain = []
    accValid = []
    for j in range(nTrials):
        # re-shuffle the data for the next run
        dataObj._shuffleAndPopulate()
        # get the accuracy for this fit and data object (Train/validation)
        trainTmp,vldTmp=  fitAndPredict(outDir,predictDir,fitter,
                                        dataObj,testDat,i,fitterCoeff,params,
                                        plot,dataClass)
        print("Trial {:d}/{:d} (repeat {:d}/{:d}) : {:.3f}/{:.3f}".\
              format(i+1,len(params),j+1,nTrials,trainTmp,vldTmp))

        accTrain.append(trainTmp)
        accValid.append(vldTmp)
    return [np.mean(accTrain),np.mean(accValid)],\
        [np.std(accTrain),np.std(accValid)]
    
def getAllTrials(params,outDir,predictDir,dataObj,testDat,nTrials,
                 fitterCoeff,createFitter,plot,dataClass):
    nParams = len(params)
    meanTrainValid = np.zeros((nParams,2))
    stdTrainValid = np.zeros((nParams,2))
    # get the parameters for fitting
    # assume train/test names are the same
    for i,n in enumerate(params):
        meanTrainValid[i,:],stdTrainValid[i,:] = getTrialMeanStd(outDir,\
        predictDir,dataObj,testDat,i,nTrials,fitterCoeff,createFitter,params,
        plot,dataClass)
    return meanTrainValid,stdTrainValid

def analyze(dataObj,dataDir,outDir,testFile,createFitter,fitterParams,
            fitterCoeff,label,dataClass,nTrials,force,plot):
    # 'createfitter' takes in the current iteration 'i', and returns a fitter
    # e.g. "return LogisticRegression(C=[10,30,40][i])"
    # 'fitterParams' gives the value of the parameters used at each iter.
    predictDir = pGenUtil.ensureDirExists(outDir + "predictions/")
    testDat = getData(dataClass,outDir,dataDir + testFile,test=True)
    params = fitterParams()
    fName = outDir+"accuracyTrials_{:d}repeats_{:d}params.pkl".format(nTrials,
                                                            len(params))
    means,std=pCheckUtil.getCheckpoint(fName,getAllTrials,
            force,params,outDir,predictDir,dataObj,testDat,nTrials,
                                       fitterCoeff,createFitter,plot,dataClass)
    # plot the accuracies versus the fit parameter.
    if (plot):
        plotAccuracies(outDir,label,means,std,params)
    return means,std

def plotErrorAnalysis(mean,std,params,labels,fullOutput):
    rowsPerPlot = min(4,len(mean))
    fig = pPlotUtil.figure(xSize=rowsPerPlot*6,ySize=len(mean)*4)
    nTrials = len(mean)
    colors = pPlotUtil.cmap(nTrials)
    minP = min([ min(p) for p in params] )
    maxP = max([ max(p) for p in params] )
    lowerAcc = min([min(acc.flatten()) for acc in mean])
    lowerBounds = [(meanV[:,1]-stdV[:,1]) for meanV,stdV in zip(mean,std) ]
    validLowerBound = np.array([np.max(bound) for bound in lowerBounds ])
    bestIdx = np.array([np.argmax(bound) for bound in lowerBounds ] )
    sortedBestValid = np.argsort(validLowerBound)[::-1]
    for idx in sortedBestValid:
        print("{:s} has lower accuracy of {:.3f} at condition {:.2g}".\
            format(labels[idx],validLowerBound[idx],bestIdx[idx]))
    i=0
    fontsize=20
    for meanV,stdV,pVals,lab in zip(mean,std,params,labels):
        ax=plt.subplot(np.ceil(nTrials/rowsPerPlot),rowsPerPlot,i+1)
        plt.errorbar(pVals,meanV[:,0],stdV[:,0],fmt='o-',color=colors[i],
                     label='train')
        plt.errorbar(pVals,meanV[:,1],stdV[:,1],fmt='x--',color=colors[i],
                     label='vld')
        ax.set_xscale('log')
        plt.axhline(0.8,color='r',linestyle='--')
        plt.ylim([lowerAcc*0.9,1])
        plt.xlim([minP*0.7,maxP*1.3])
        plt.title(lab,fontsize=fontsize)
        i+=1
        plt.xlabel('Classifier parameter')
        plt.ylabel('Accuracy')
    pPlotUtil.savefig(fig,fullOutput + 'allAcc')
