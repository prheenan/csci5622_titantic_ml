import base.titanicMain as main
from base.dataToMat_logreg import dataToMat_logreg  as lrClass
from base.analysis import plotErrorAnalysis
import numpy as np
from sklearn.linear_model import LogisticRegression


def defaultFitterParams():
    return np.logspace(-3,2,1)

def defaultFitter(iterNum):
    nEst = defaultFitterParams()[iterNum]
    return LogisticRegression(C=nEst,penalty='l2')

def defaultCoeff(fitter):
    return fitter.coef_[0]

label='logreg'
forceRun = True # otherwise, use checkpoint (cached file)
fullOutput = "./work/out/"+label+"/"
mean,std= main.run(defaultFitter,defaultFitterParams,
                   defaultCoeff,lrClass,label=label,valid=0.1,nTrials=1,
                   force=forceRun,plot=True,profile=True)
plotErrorAnalysis([mean],[std],[defaultFitterParams()],[label],fullOutput)
    
