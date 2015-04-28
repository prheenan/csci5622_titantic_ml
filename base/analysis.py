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
from scipy.sparse import csr_matrix

def getNormalizedFeatureMatrix(badIdx,featureMat,sortFunc):
    minByCol = featureMat.min(axis=0).toarray()[0]
    maxByCol = featureMat.max(axis=0).toarray()[0]
    nFeatures = len(minByCol)
    nBad = len(badIdx)
    toRet=  csr_matrix((nBad,nFeatures))
    score = np.zeros(nBad)
    fudge = 0.01
    for i,personRow in enumerate(badIdx):
        # this is the row for an individual
        idx = slice(featureMat.indptr[personRow],featureMat.indptr[personRow+1])
        featureCols = featureMat.indices[idx]
        nDefFeatures = len(featureCols)
        featureDat = featureMat.data[idx]
        featMins = np.array(minByCol[featureCols])
        featMax = np.array(maxByCol[featureCols])
        featureNormalized = featureDat-featMins+fudge
        colorProp = featureNormalized/(featMax-featMins)
        toRet[i,featureCols] = colorProp
    toRet = sortFunc(toRet)
    return toRet

def plotFeatMatr(toPlot,featureObjects,featureMat,saveDir,label,badIdx):
    nFeats = featureMat.shape[1]
    nnzPerFeature = toPlot.getnnz(0)
    # get the indices to sort this ish.
    # how many should we use?...
    # get the top N most common
    mostCommon = np.argsort(nnzPerFeature)[-nFeats//7:]
    # get their labels
    featLabels = [f.label() for f in featureObjects]
    # get a version imshow can handle
    matImage = toPlot.todense()
    # fix the aspect ratio
    aspectSkew = len(badIdx)/nFeats
    aspectStr = 1./aspectSkew
    # plot everything
    ax = plt.subplot(1,1,1)
    cax = plt.imshow(matImage,cmap=plt.cm.hot_r,aspect=aspectStr,
                     interpolation="nearest")
    plt.spy(toPlot,marker='s',markersize=1.0,color='b',
            aspect=aspectStr,precision='present')
    cbar = plt.colorbar(cax, ticks=[0, 1], orientation='vertical')
    # horizontal colorbar
    cbar.ax.set_yticklabels(['Min Feature Value', 'Max Feature Value'])
    ax.set_xticks(range(nFeats))
    ax.set_xticklabels(featLabels,rotation='vertical')
    plt.xlabel("Feature Number")
    plt.ylabel("Individual")
    return aspectStr

def getIdxMistakes(yPred,yActual):
    badIdx = [ i  for i,pred in enumerate(yPred) if pred != yActual[i] ]
    predictedDeath = [ i for i,predIdx in enumerate(badIdx) \
                       if yPred[predIdx]==0]
    predictedSurv = [ i for i,predIdx in enumerate(badIdx) \
                       if yPred[predIdx]==1]
    return badIdx,predictedDeath,predictedSurv

def sortByPred(matrix,yPred,yActual):
    # 1 if predicted death, actually survived, 
    # 0 if prediced survivial, actually dead
    badIdx,predDeath,predSurv = getIdxMistakes(yPred,yActual)
    score = [ 1 if i in predDeath else 0 for i in range(matrix.shape[0])]
    sortIdx = np.argsort(score)
    # we now have the array like [1,2,...,len(predDeath),...,len(N)]
    # sort *within* in the dead and the survivors
    return matrix[sortIdx,:]

def profileLosers(saveDir,label,yPred,yActual,rawDat,dataClass,featureMat,
                  featureObjects):
    # get what we got wrong
    badIdx,predictedDeath,predictedSurv = getIdxMistakes(yPred,yActual)
    nSurv = len(predictedSurv)
    nDead = len(predictedDeath)
    fig = pPlotUtil.figure(xSize=16,ySize=12,dpi=200)
    # get the matrix, all features 0 --> 1
    toPlot = getNormalizedFeatureMatrix(badIdx,featureMat,
                                        lambda x: sortByPred(x,yPred,yActual))
    # get the number of non-zero elements in each column
    aspectStr = plotFeatMatr(toPlot,featureObjects,featureMat,saveDir,label,
                             badIdx)
    plt.axhline(len(predictedSurv),linewidth=3,color='c',
                label="Divides {:d} actual deceased from {:d} actual survived".\
                format(nSurv,nDead))
    plt.legend(loc="upper right", bbox_to_anchor=(0.4, -0.4))
    badVals = rawDat[badIdx,:]
    np.savetxt(saveDir + 'debug_{:s}.csv'.format(label),badVals,fmt="%s",
               delimiter=',')
    pPlotUtil.savefig(fig,saveDir + "mOut" + label,tight=True)


def predict(fitter,x,yReal,rawDat,label,saveDir,colNames,fitterCoeff,objClass,
            featureObjects,saveBad=True,saveCoeffs=False,plot=True):
    try:
        yPred = fitter.predict(x)
    except TypeError:
        yPred = fitter.predict(x.toarray())
    cm = confusion_matrix(yReal,yPred)
    acc= accuracy_score(yReal,yPred)
    # Show confusion matrix in a separate window
    if (saveBad):
        # XXX could profile?
        profileLosers(saveDir,label,yPred,yReal,rawDat,objClass,x,
                      featureObjects)
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
    colNames = np.array([c.label() for i,c in enumerate(dataObj._trainObj)],
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
