# force floating point division. Can still use integer with //
from __future__ import division
# need to add the utilities class. Want 'home' to be platform independent
import sys
sys.path.append("./util/")
# import the patrick-specific utilities
import GenUtilities  as pGenUtil
import PlotUtilities as pPlotUtil
import CheckpointUtilities as pCheckUtil

from dataToMat import ShipData
from utilIO import getData,getDirsFromCmdLine
from analysis import analyze

valid = 0.05 # percentage to use for validation.
trainFile = "train.csv"
testFile = "test.csv"
inDir,cacheDir,outDir = getDirsFromCmdLine()
predictDir = pGenUtil.ensureDirExists(outDir + "predictions")
profileDir = pGenUtil.ensureDirExists(outDir + "profile")

# get the data object, by cache or otherwise 
dataObj = \
    pCheckUtil.pipeline([[cacheDir+'data.pkl',getData,outDir,inDir+trainFile,
                          valid,False,profileDir]],True)


analyze(dataObj,inDir,outDir,testFile)
