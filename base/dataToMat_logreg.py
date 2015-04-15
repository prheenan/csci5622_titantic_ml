from dataToMat import ShipData as superClass
import abc # import the abstract base class 

class dataToMat_logreg(superClass):
    def __init__(self,dataInfoDir,data,valid=None,test=False,profileName=None):
        super(dataToMat_logreg, self).__init__(dataInfoDir,
                                               data,valid,test,profileName)
    def _getXandY(self,data,test=False):
        # could call mask on this
        x,y,labels = self._defaultXY(data,test)
        # could also return
        cols = [0,1,2,3,5,7,8,9,10,11,15,22]
        return self._maskArr(x,cols),y, labels[cols]




