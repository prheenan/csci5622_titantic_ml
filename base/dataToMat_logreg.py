from dataToMat import ShipData as superClass
import numpy as np
import abc # import the abstract base class

class dataToMat_logreg(superClass):
    def __init__(self,dataInfoDir,data,valid=None,test=False,profileName=None):
        super(dataToMat_logreg, self).__init__(dataInfoDir,
                                               data,valid,test,profileName)
    def _makeBigram(self,x,bcol1,bcol2):
        return x[:,bcol1].toarray() * x[:,bcol2].toarray()
    def _makeTrigram(self,x,bcol1,bcol2,bcol3):
        return x[:,bcol1] * x[:,bcol2] * x[:,bcol3]
		
    def _getXandY(self,data,test=False):
        return self._defaultXY(data,test)




