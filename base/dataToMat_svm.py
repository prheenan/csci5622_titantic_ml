from dataToMat import ShipData as superClass
import abc # import the abstract base class 

class dataToMat_svm(superClass):
    def __init__(self,dataInfoDir,data,valid=None,test=False,profileName=None):
        super(dataToMat_svm, self).__init__(dataInfoDir,data,valid,test,
                                            profileName)
    def _getXandY(self,data,test=False):
        x,y,labels,cols = self._defaultXY(data,test)
        colMask = [0,1,2,3,5,6,7,8,9,10,11,15,18,21,22,25,26,29,31,31]
        colMask.extend([a for a in range(31,cols)])
        return self._maskArr(x,colMask),y,[labels[i] for i in colMask],len(colMask)
