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
		
    def _getXandY(self,data,test=False,allBigrams=False):
        if not allBigrams:
            x,y,labels,cols = self._defaultXY(data,test)
            colKeep = [7,11,1,13,5,9,22,28,31,3,15,35,2,37,24,8,0]
            return self._maskArr(x,colKeep),y,[labels[i] for i in colKeep],\
                len(colKeep)
        else:
            # could call mask on this
            x,y,labels,col = self._defaultXY(data,test)
            # add more engineered features for just LogReg
            nStats = 28
            #labels = np.empty((nStats),dtype=np.object)
            bg = 0 
            loop = col
            print 'analyzing bigrams...'
            for iterx in range(loop):
                for itery in range(loop):
                    if iterx <=itery:
                        if x[:,iterx].size == x[:,itery].size:
                            col = self._addEngr(x,col,
        self._makeBigram(x, iterx,itery),labels,'bigram' + str(labels[iterx]) +
                                '+' + str(labels[itery]))
            # could also return
            # return self._maskArr(x,[0,1,2,3]),y,labels
            # cols = [0,1,2,3,5,7,8,9,10,11,15,22]
            # return self._maskArr(x,cols),y, [labels[c] for c in cols]
            x,y,labels,cols = self._defaultXY(data,test)
            colMask = [0,1,2,3,5,6,7,8,9,10,11,15,18,21,22,25,26,29,31,31]
            colMask.extend([a for a in range(31,cols)])
            return self._maskArr(x,colMask),y,[labels[i] for i in colMask],\
                len(colMask)



