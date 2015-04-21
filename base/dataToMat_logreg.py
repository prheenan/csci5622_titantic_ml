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
        print x[:,bcol1].size
        print x[:,bcol1].size
        return x[:,bcol1] * x[:,bcol2] * x[:,bcol3]
		
    def _getXandY(self,data,test=False):
        # could call mask on this
        x,y,labels,col = self._defaultXY(data,test)
        # add more engineered features for just LogReg
        nStats = 28
        #labels = np.empty((nStats),dtype=np.object)
        bg = 0 
        # instead of the following, pick and choose since
        # some features are not represented throughout
        loop = col
        print 'analyzing bigrams...'
        for iterx in range(loop):
            for itery in range(loop):
                if iterx <=itery:
                    if x[:,iterx].size == x[:,itery].size:
                        col = self._addEngr(x,col,
                                self._makeBigram(x, iterx,itery),
								labels,'bigram' + str(labels[iterx]) +
                                '+' + str(labels[itery]))
        # could also return
        # return self._maskArr(x,[0,1,2,3]),y,labels
        cols = [0,1,2,3,5,7,8,9,10,11,15,22]
        return self._maskArr(x,cols),y, [labels[c] for c in cols]



