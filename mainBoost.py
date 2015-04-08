import base.titanicMain as main

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
main.run(boost_fit,boostParams,getCoef,lrClass,label='boost',valid=0.05)
