from dataToMat import ShipData as superClass
import abc # import the abstract base class 

class dataToMat_boosting(superClass):
    def __init__(self,dataInfoDir,data,valid=None,test=False,profileName=None):
        super(dataToMat_boosting, self).__init__(dataInfoDir,
                                               data,valid,test,profileName)
    def _getXandY(self,data,test=False):
        return self._defaultXY(data,test)
