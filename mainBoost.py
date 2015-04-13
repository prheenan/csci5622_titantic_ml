import base.titanicMain as main
from base.analysis import plotErrorAnalysis
from sklearn.ensemble import AdaBoostClassifier as Boost
from base.dataToMat_boosting import dataToMat_boosting as lrClass


def boostParams():
    return [1,5,10,20,50]

def boost_fit(iterNum):
    nEst = boostParams()[iterNum]
    return Boost(n_estimators=nEst)

def getCoef(fitter):
    return fitter.estimator_weights_ 

# give functions for generating a fitter, parameters, and coefficients
numRepeatTrials = 5
label = 'boost'
fullOutput = "./work/out/"+label+"/"
mean,std = main.run(boost_fit,boostParams,getCoef,lrClass,label=label,
                    valid=0.05,nTrials=numRepeatTrials)
plotErrorAnalysis([mean],[std],[boostParams()],['boostExample'],fullOutput)
