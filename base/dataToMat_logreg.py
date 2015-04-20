from dataToMat import ShipData as superClass
import numpy as np
import abc # import the abstract base class

class dataToMat_logreg(superClass):
    def __init__(self,dataInfoDir,data,valid=None,test=False,profileName=None):
        super(dataToMat_logreg, self).__init__(dataInfoDir,
                                               data,valid,test,profileName)

    def _makeBigram(self,x,bcol1,bcol2):
        return x[:,bcol1] * x[:,bcol2]

    def _getXandY(self,data,test=False):
        # could call mask on this
        x,y,labels,col = self._defaultXY(data,test)
        # add more engineered features for just LogReg
        col = self._addEngr(x,col,self._makeBigram(x.toarray(),2,3),
                            labels,'bigram1')
        cols = [0,1,2,3,5,6,7,8,9,10,11,15,18,21,22,25,26,29,32]
        return x,y,labels#self._maskArr(x,cols),y, labels[cols]




