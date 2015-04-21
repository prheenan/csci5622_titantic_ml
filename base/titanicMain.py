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
from utilIO import getData,getDirsFromCmdLine
from analysis import analyze
import numpy as np

def run(fitter,fitterParams,fitterCoeff,dataClass,label,valid=0.05,
        train="train.csv",test="test.csv",profile=False,nTrials=1,
        force=True,forceFeat=True,plot=False):
    trainFile = train
    testFile = test
    inDir,cacheDir,outDir = getDirsFromCmdLine()
    # add the label for this run (ie: SVM/Boost/LogisticRegression)
    outDir = pGenUtil.ensureDirExists(outDir + label +"/")
    # get the directories we want
    predictDir = pGenUtil.ensureDirExists(outDir + "predictions")
    if (profile and plot):
        profileDir = pGenUtil.ensureDirExists(outDir + "profile")
    else:
        profileDir = None
    # get the data object, by cache or otherwise 
    dataObj = \
    pCheckUtil.pipeline([[cacheDir+'data.pkl',getData,dataClass,outDir,
                          inDir+trainFile,valid,False,profileDir,]],forceFeat)
    return analyze(dataObj,inDir,outDir,testFile,fitter,fitterParams,
                   fitterCoeff,label,dataClass,nTrials,force,plot)

if __name__ == "__main__":
    # lol, patrick broke this 
    pass