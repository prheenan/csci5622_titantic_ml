import base.titanicMain as main
from base.analysis import plotErrorAnalysis
from sklearn.svm import SVC as SVC
from base.dataToMat_svm import dataToMat_svm as lrClass
import numpy as np
import sys
sys.path.append("./util/")
sys.path.append("../util/")
# import the patrick-specific utilities
import GenUtilities  as pGenUtil
import PlotUtilities as pPlotUtil
import CheckpointUtilities as pCheckUtil
import matplotlib.pyplot as plt

rbfStr = 'rbf'
linStr = 'linear'
polyStr = 'poly'

class svmObj:
    def __init__(self,kernel,params,deg,gam,c0):
        self._ker = kernel
        self._params = params
        self._deg = deg
        self._gam = gam
        self._c0 = c0
    def SVC_params(self):
        return self._params
    def SVC_fit(self,iterNum):
        nEst =  self._params[iterNum]
        mKer = self._ker
        return SVC(kernel=mKer,C=nEst,degree=self._deg,gamma=self._gam,
                   coef0 = self._c0)
    def SVC_coeffs(self,fitter):
        mKer = self._ker
        if (mKer == linStr):
            return fitter.coef_[0].toarray()[0]
        else: 
            return fitter.dual_coef_.toarray()[0]
            
def getTrialStats(svmObjs,label,valid,nTrials,forceRun,forcePlot):
    numMeanStd = len(svmObjs)
    means = []
    std = []
    params = []
    labels = []
    i = 0
    for ker,pVals,degree,gamma,c0 in svmObjs:
        obj = svmObj(ker,pVals,degree,gamma,c0)
        # only profile the first..
        profile = (i == 0)
        labelStr = label + "_{:s}{:d}_deg{:d}_gamma_{:.3g}".\
                   format(ker,i,degree,gamma)
        meanTmp, stdTmp =main.run(obj.SVC_fit,obj.SVC_params,obj.SVC_coeffs,
            lrClass,label=labelStr,valid=valid,profile=profile,nTrials=nTrials,
                                  force=forceRun,plot=forcePlot,
                                  forceFeat=True)
        means.append(meanTmp)
        std.append(stdTmp)
        params.append(pVals)
        labels.append(labelStr)
        i += 1
    return means,std,params,labels

def run(labels,valid,nTrials,forceRun,forcePlot):
    fullOutput = "./work/out/"+label+"-full/"
    mean,std,params,labels=pCheckUtil.getCheckpoint(fullOutput + 'stats.pkl',
            getTrialStats,forceRun,svmObjs,label,valid,nTrials,forceRun,
                                                    forcePlot)
    plotErrorAnalysis(mean,std,params,labels,fullOutput)

if __name__ == '__main__':

# give functions for generating a fitter, parameters, and coefficients
    defParams =[0.01,0.025,0.05,0.1,0.2,0.5,2.5,10,20,40,80,160,300,500]
    polyParams = defParams[:-4]
    #svmObj: formatted like [kernelStr,params,degree,gamma]
    svmObjs = [ (linStr,defParams,0,0,0),
                (rbfStr,defParams,0,0,0),
                (rbfStr,defParams,0,1e-3,0),
                (rbfStr,defParams,0,5e-3,0),
                (rbfStr,defParams,0,7e-3,0),
                (rbfStr,defParams,0,9e-3,0),
                (rbfStr,defParams,0,1e-2,0),
                (rbfStr,defParams,0,1.5e-2,0),
                (rbfStr,defParams,0,2e-2,0),
                (rbfStr,defParams,0,3e-2,0),
                (rbfStr,defParams,0,5e-2,0),
                (rbfStr,defParams,0,7e-2,0),
                (rbfStr,defParams,0,1e-1,0)]
    runFull = False
    forcePlot = False
    forceRun = True
    if (runFull):
        # running on Kaggle, grab the entire training set
        label='svm-tosubmit'
        valid = 0.
        nTrials = 1
    else:
        label = 'svm-masked'
        valid = 0.1
        nTrials = int(np.ceil(1/valid))
    run(label,valid,nTrials,forceRun,forcePlot)
