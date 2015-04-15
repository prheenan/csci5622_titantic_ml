import base.titanicMain as main
from sklearn.svm import SVC as SVC
from base.dataToMat_svm import dataToMat_svm as lrClass

def SVC_params():
    return [1]

def SVC_fit(iterNum):
    nEst =  SVC_params()[iterNum]
    return SVC(kernel='linear',C=nEst)

def SVC_coeffs(fitter):
    return fitter.coef_

# give functions for generating a fitter, parameters, and coefficients
main.run(SVC_fit,SVC_params,SVC_coeffs, lrClass,label='svm',valid=0.05)