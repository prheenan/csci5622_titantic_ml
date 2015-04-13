# force floating point division. Can still use integer with //
from __future__ import division
# need to add the utilities class. Want 'home' to be platform independent
import sys
sys.path.append("./util/")
sys.path.append("../util/")
# import the patrick-specific utilities
import GenUtilities  as pGenUtil
import PlotUtilities as pPlotUtil
import CheckpointUtilities as pCheckUtil

from dataToMat import ShipData
from sklearn.linear_model import LogisticRegression
from utilIO import getData,getDirsFromCmdLine
from analysis import analyze
import numpy as np

def defaultFitterParams():
    return np.logspace(-2,1,10)

def defaultFitter(iterNum):
    nEst = defaultFitterParams()[iterNum]
    return LogisticRegression(C=nEst)

def defaultCoeff(fitter):
    return fitter.coef_[0]

def run(fitter,fitterParams,fitterCoeff,dataClass,label="LogReg",valid=0.05,
        train="train.csv",test="test.csv",profile=True,nTrials=1):
    trainFile = train
    testFile = test
    inDir,cacheDir,outDir = getDirsFromCmdLine()
    # add the label for this run (ie: SVM/Boost/LogisticRegression)
    outDir = pGenUtil.ensureDirExists(outDir + label +"/")
    # get the directories we want
    predictDir = pGenUtil.ensureDirExists(outDir + "predictions")
    if (profile):
        profileDir = pGenUtil.ensureDirExists(outDir + "profile")
    else:
        profileDir = None
    # get the data object, by cache or otherwise 
    dataObj = \
    pCheckUtil.pipeline([[cacheDir+'data.pkl',getData,dataClass,outDir,
                          inDir+trainFile,valid,False,profileDir,]],True)
    return analyze(dataObj,inDir,outDir,testFile,fitter,fitterParams,
                   fitterCoeff,label,dataClass,nTrials)

if __name__ == "__main__":
    run(defaultFitter,defaultFitterParams,defaultCoeff)
