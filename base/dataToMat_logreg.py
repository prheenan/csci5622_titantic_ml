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
        x,y,labels,cols = self._defaultXY(data,test)
        colMask = [0,1,2,3,5,6,7,8,9,10,11,15,18,21,22,25,26,29,31,32,33]
        colMask.extend([a for a in range(colMask[-1]+1,cols)])
        return self._maskArr(x,colMask),y,[labels[i] for i in colMask],\
            len(colMask)



