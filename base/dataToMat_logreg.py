from dataToMat import ShipData as superClass
import numpy as np
import abc # import the abstract base class

class dataToMat_logreg(superClass):
    def __init__(self,dataInfoDir,data,valid=None,test=False,profileName=None):
        super(dataToMat_logreg, self).__init__(dataInfoDir,
                                               data,valid,test,profileName)

    def _makeBigram(self,x,bcol1,bcol2):
        return x[:,bcol1] * x[:,bcol2]

    def _makeTrigram(self,x,bcol1,bcol2,bcol3):
        return x[:,bcol1] * x[:,bcol2] * x[:,bcol3]
    def _getXandY(self,data,test=False):
        # could call mask on this
        x,y,labels,col = self._defaultXY(data,test)
        # add more engineered features for just LogReg
        bg = 0 
        col = self._addEngr(x,col,self._makeBigram(x.toarray(),0,0),
                                    labels,'bigram' + str(bg))
        col = self._addEngr(x,col,self._makeBigram(x.toarray(),0,2),
                                    labels,'bigram' + str(bg))
        col = self._addEngr(x,col,self._makeBigram(x.toarray(),0,2),
                                    labels,'bigram' + str(bg))
        for iterx in xrange(col):
            for itery in xrange(col):
                if iterx <=itery:
                    bg = bg + 1
                    print bg
                    col = self._addEngr(x,col,self._makeBigram(x.toarray(),
                                                               iterx,itery),
                                        labels,'bigram' + str(bg))
        # could also return
        # return self._maskArr(x,[0,1,2,3]),y,labels
        cols = [0,1,2,3,5,7,8,9,10,11,15,22]
        return self._maskArr(x,cols),y, labels[cols]




