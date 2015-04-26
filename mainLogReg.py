import base.titanicMain as main
from base.dataToMat_logreg import dataToMat_logreg  as lrClass
from base.analysis import plotErrorAnalysis
import numpy as np
from sklearn.linear_model import LogisticRegression


def defaultFitterParams():
    return np.logspace(-3,2,10)

def defaultFitter(iterNum):
    nEst = defaultFitterParams()[iterNum]
    return LogisticRegression(C=nEst)

def defaultCoeff(fitter):
    return fitter.coef_[0]

label='logreg'
forceRun = True # otherwise, use checkpoint (cached file)
fullOutput = "./work/out/"+label+"/"
mean,std= main.run(defaultFitter,defaultFitterParams,
                   defaultCoeff,lrClass,label=label,valid=0.1,nTrials=4,
                   force=forceRun,plot=True)
plotErrorAnalysis([mean],[std],[defaultFitterParams()],[label],fullOutput)
    
