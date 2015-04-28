from dataToMat import ShipData as superClass
import abc # import the abstract base class 

class dataToMat_svm(superClass):
    def __init__(self,dataInfoDir,data,valid=None,test=False,profileName=None):
        super(dataToMat_svm, self).__init__(dataInfoDir,data,valid,test,
                                            profileName)
    def _getXandY(self,data,test=False):
        x,y,labels,cols = self._defaultXY(data,test)
        colKeep = [7,11,1,13,5,9,22,28,31,3,15,35,2,37,24,8,0]
        return self._maskArr(x,colKeep),y,[labels[i] for i in colKeep],\
            len(colKeep)
