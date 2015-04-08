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
        # return self._maskArr(x,[0,1,2,3]),y,labels
        return x,y,labels

