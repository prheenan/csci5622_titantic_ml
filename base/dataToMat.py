# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
# need to add the utilities class. Want 'home' to be platform independent
import sys
sys.path.append("./util/")
sys.path.append("../util/")
# import the patrick-specific utilities
import GenUtilities  as pGenUtil
import PlotUtilities as pPlotUtil
import CheckpointUtilities as pCheckUtil
from scipy.sparse import csr_matrix
import re
# use an abstract class for the actual features
import abc


class Feat:
    def __init__(self,data,name,col,isNorm=False,mean=None,std=None,
                 categoryLab=None,bigramData=None):
        if (mean is None or std is None):
            self._mean = np.mean(data)
            self._std  = np.std(data)
        else:
            self._mean = mean
            self._std = std
        self._name = name
        self._norm = isNorm
        self._col = col
        # labels for the x axis of a histogram
        self._labels = categoryLab
        self._big = bigramData
    def isNorm(self):
        return self._norm
    def statStr(self):
        return "Mean/std: {:.2g}/{:.1g}".format(self._mean,self._std)
    def label(self):
        return str(self._col) + self._name

class ShipData(object):
    __metaclass__ = abc.ABCMeta
    cabLevels = {'X': 0, 'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8}
    def _getCabinLevelNum(self,dCab):
        # append all of the cabin "letters" for each person (may be more than 1)
        toRet = []
        for i,c in enumerate(dCab):
            allCabins = c.split(' ')
            toRet.append([self.cabLevels[s[0]] for s in allCabins])
        return toRet
    # transformation functions
    def _intTx(self,arr):
        # cast all elements to an int
        return [int(a) for a in arr]
    def _floatTx(self,arr):
        # cast all elements to a float
        return [float(a) for a in arr]
    def _dictTx(self,arr,dictV):
        # use a dict to transform the data
        return [ dictV[a] for a in arr]
    def _numCabins(self,cabins):
        return [ len(c.split(" ")) for c in cabins]
    def _cabLevel(self,cabins):
        toRet = np.zeros(len(cabins))
        allV = self._getCabinLevelNum(cabins)
        for i,c in enumerate(allV):
            meanLevel = np.mean(c)
            toRet[i] = meanLevel
        return toRet
    def _gendTx(self,arr):
        # binary male/female
        return self._dictTx(arr,{'male':0, 'female':1 })
    def _embTx(self,arr):
        # embark destination
        return self._dictTx(arr,{'C':0, 'Q':1, 'S':2})
    def _embFirst(self,arr):
        emb = self._embTx(arr)
        embTarget = min(emb)
        toRet = [ int(e == embTarget) for e in emb]
        return toRet
    def _decPrefix(self):
        return ['SOTONOQ','A4','SOP','SOPP', 'WEP','WC','SOTONO2', 
                'SP','CA','SOC']
    def _undetPrefix(self):
        return ['A5', 'SCPARIS','STONO','C', 'STONO2','PP']
    def _surPrefix(self):
        return [  'CASOTON', 'FA', 'FC', 'PC', 
                 'PPP','SC', 'SCA4',  'SCOW', 
                'FCC','SWPP','SCAH']        
    def _getPrefixes(self):
        # return dec, undec, sur
        return self._decPrefix() + self._undetPrefix() + self._surPrefix()
    def _processPrefix(self,x):
        return  x.replace('.','').replace('/','').upper()
    def _hasPrefixIdx(self,dTickets):
        tmp = []
        prefixes = self._getPrefixes()
        for j,a in enumerate(dTickets):
            split = a.split(" ")
            if (len(split) == 0 or self._processPrefix(split[0]) 
                not in prefixes):
                continue
            else:
                tmp.append(j)
        return tmp
    def _hasPrefixFeature(self,dTickets):
        hasPrefix = [False for d in dTickets]
        prefixIdx = self._hasPrefixIdx(dTickets)
        for i in prefixIdx:
            hasPrefix[i] = True
        return hasPrefix
    def _genericPrefix(self,dTickets,func):
        # XXX assume we have called 'hasPrefix' before this
        # this converts everything ot a prefix, then calls the function we want 
        prefixes =  [self._processPrefix(self._processPrefix(a.split(" ")[0]))
                     for a in dTickets]
        return [func(pref) for pref in prefixes]
    def _smallNameLen(self,dName):
        return [len(name) > 13 and len(name) < 25 for name in dName]
    def _longNameLen(self,dName):
        return [len(name) > 40 for name in dName]
    def _prefixA5(self,arr):
        return self._genericPrefix(arr, lambda x: x == 'A5') 
    def _prefixInSurv(self,arr):
        return self._genericPrefix(arr, lambda x: x in self._surPrefix()) 
    def _prefixInDec(self,arr):
        return self._genericPrefix(arr, lambda x: x in self._decPrefix()) 
    def _getTicketPrefix(self,arr):
        # transform ticket destinations. Must call 'hasPrefix'
        prefixes = self._getPrefixes()
        numTix = len(arr)
        tmp = np.zeros(numTix)
        for j,a in enumerate(arr):
            split = a.split(" ")
            # XXX assume we are using 'hasPrefix'
            tmp[j] = prefixes.index(self._processPrefix(split[0]))
        return tmp
    def _toIdx(self,arr):
        return [i for i,v in enumerate(arr) if v]
    def _allIdx(self,arr):
        # allow all indices XXX change into something for efficient?
        return [ i for i in range(len(arr)) ]
    def _empIdx(self,arr):
        # pick out indices with non-zero lengths
        return [ i for i in range(len(arr)) if len(arr[i]) != 0 ]
    # below are feature engineering methods
    def _ageEstimated(self,ages):
        # only estimated if like xx.5, where xx is >= 1.
        tol = 0.1
        toRet = np.zeros(len(ages))
        for i,age in enumerate(ages):
            toRet[i] = 0
            if (len(age) == 0):
                continue
            asFloat = float(age)
            if (asFloat > 1.0 and (asFloat - int(asFloat) >= tol)):
                toRet[i] = 1
        return toRet
    def _portUnknown(self,ports):
        return [ len(s) == 0 for s in ports ]
    def _genMatch(self,regex,names):
        return [ re.findall(regex,s) for s in names ]
    def _genLength(self,mFunc,names):
        return [ len(matchV[0]) for matchV in mFunc(names)]
    def _nickNameMatch(self,names):
        nickNameReg = re.compile(r'''\"(\w+)\"''',re.VERBOSE)
        return self._genMatch(nickNameReg,names)
    def _hasNickName(self,names):
        matches = self._nickNameMatch(names)
        return [len(n) > 0  for n in matches]
    def _nickNameLength(self,names):
        toRet = self._genLength(self._nickNameMatch,names)
        return toRet
    def _secondNameMatch(self,names):
        secondNameRegex = re.compile(r'''\([^\(\"]+\)''',re.VERBOSE)
        return self._genMatch(secondNameRegex,names)
    def _secondNameLen(self,names):
        # assumes that IdxFunc is hasSecondName (ie: everthing in names has
        # a lenght
        return self._genLength(self._secondNameMatch,names)
    def _hasSecondName(self,names):
        # match an enclosed name in parenthesis. don't match quotes
        toRet = [ len(matchV) > 0 for matchV in self._secondNameMatch(names)]
        return toRet
    def _secondNameIdx(self,names):
        toRet = self._toIdx(self._hasSecondName(names))
        return toRet
    def _isInfant(self,ages):
        return [1 if len(a) >0 and float(a) < 2.0 else 0 for a in ages]
    def _isChild(self,ages):
        return [1 if len(a) >0 and float(a) < 10.0 else 0 for a in ages]
    def _hasSiblings(self,count):
        return [1 if int(c) > 0 else 0 for c in count]
    def _hasParents(self,count):
        return [1 if int(c) > 0 else 0 for c in count]
    def _ageKnown(self,ages):
        toRet =  [len(a) > 0  for a in ages ]
        return toRet
    def _isElderly(self,ages):
        return [1 if len(a) != 0 and float(a) >=65 else 0 for a in ages]
    def _classGen(self,classes,num):
        return [int(c) == int(num) for c in classes]
    def _isFirstClass(self,classes):
        return self._classGen(classes,1)
    def _isSecondClass(self,classes):
        return self._classGen(classes,2)
    def _isThirdClass(self,classes):
        return self._classGen(classes,3)  
    def _highFare(self,highFare):
        return [len(fare) > 0 and float(fare) > 200.  for fare in highFare ] 
    def _highSiblings(self,siblings):
        return [ int(s) >= 2.5 for s in siblings ]
    def _fareUnknown(self,fares):
        toRet = [len(f.strip()) == 0 or float(f.strip()) < 1.e-6
                 for f in fares ]
        return toRet
    def _hasCabin(self,dCab):
        return [ len(d) > 0 for d in dCab ]
    def _highCab(self,dCab):
        return [ int(d > 1) for d in self._cabLevel(dCab)]
    def _safeNorm(self,data):
        stdV = np.std(data)
        mean = np.mean(data)
        delta = (data - mean)
        if (data.size == 0):
            raise StandardError 
        # make toAdd normalized
        if (stdV != 0):
            return delta/stdV,mean,stdV
        else:
            return delta,mean,stdV
    def _labelStat(self,label,mean,std):
        return label + "{:.2g}_{:.1g}".format(mean,std)
    def _add(self,mArr,arrCol,data,nameArr,name,idxFunc = None,
             txFunc = None,norm=False,categoryLabels=None):
        # add to the sparse internal matrix at col 'column' using data from
        # col 'column'
        if txFunc is None:
            txFunc = self._intTx
        if idxFunc is None:
            idxFunc = self._allIdx
        goodIndices = idxFunc(data)
        goodData = np.array(txFunc(data[goodIndices]))
        return self._addEngr(mArr,arrCol,goodData,nameArr,name,norm,
                             goodIndices,categoryLabels)

    def _safeMatrixAdd(self,toAddTo,col,indices,finalDat):
        toAddTo[indices,col] = finalDat
    def _addEngr(self,toAddTo,col,data,nameArr,name,norm=False,indices=None,
                 categoryLabels = None,bigramData=None):
        # add 'data' at 'col' of 'toAddTo', save 'name' in 'nameArr',
        # and normalize or take specifi indices according to 'indices'
        toAdd = np.array(data)
        mean = np.mean(toAdd)
        std = np.std(toAdd)
        if (norm):
            toAdd,mean,std = self._safeNorm(toAdd)
        finalDat = np.reshape(toAdd,((toAdd.size),1))
        # add an extra column if we need it
#http://stackoverflow.com/questions/4695337/expanding-adding-a-row-or-column-a-scipy-sparse-matrix
        if (indices is None):
            indices = range(toAddTo.shape[0])
        # POST: have the proper indices
        self._safeMatrixAdd(toAddTo,col,indices,finalDat)
        # note: we add an index to the name, so we can keep track of which
        # column. This is helpful for the plots.
        newFeature = Feat(finalDat,name,col,norm,mean,std,
                          categoryLabels,bigramData)
        # XXX TODO: :-(. This is copy pasta.
        nameArr.append(newFeature)
        return col + 1
    def _columnWise(self,data,limit=None):
        if (limit is None):
            nCols = data.shape[1]
        else:
            nCols = min(limit,data.shape[1])
        toRet = []
        for i in range(nCols):
            toRet.append(data[:,i])
        return toRet
    def _addBigram(self,toAddTo,labels,col1,col2,col,logic=lambda x,y: x*y):
        data1 = toAddTo[:,col1].toarray()
        data2 = toAddTo[:,col2].toarray()
        lab1 = labels[col1].label()
        lab2 = labels[col2].label()
        if (data1.size != data2.size):
            print(("Couldn't make feat {:s} [size {:d}] and {:s} [size {:d}]"+
                   " compatible, skipping...").format(data1.size,lab1,
                                                      data2.size,lab2))
            return col
        # XXX not safe! need to make sure the indices match...
        newCol =  logic(data1,data2)
        col = self._addEngr(toAddTo,col,newCol,labels,
                            "Bi:{:d}_and_{:d}_{:s}*{:s}".\
                            format(col1,col2,lab1,lab2),bigramData=(col1,col2))
        return col
    def _addBigramByName(self,toAddTo,labels,name1,name2,col,**kwargs):
        names = [l._name for l in labels]
        col1 = names.index(name1)
        col2 = names.index(name2)
        return self._addBigram(toAddTo,labels,col1,col2,col,**kwargs)
    def _defaultXY(self,data,test=False):
        # XXX add in support for testing data
        if (not test):
#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embark
            xx,yRaw,dClass,dName,dSex,dAge,dSib,dPar,dTicket,dFare,dCab,\
            dEmb = self._columnWise(data)
            trainY = [int(i) for i in yRaw]
        else:
            # there is no ID on the y column...
            xx,dClass,dName,dSex,dAge,dSib,dPar,dTicket,dFare,dCab,\
            dEmb = self._columnWise(data)
            trainY = 0
        nStats = 1800 # make very large, prune later. XXX TODO: better way?  
        nPassengers = data.shape[0]
        trainX = csr_matrix((nPassengers,nStats),dtype=np.float64)
        labels = []
        # add the class (2)
        col = 0
        col = self._add(trainX,col,dClass,labels,'class')
        # add the sex (4)
        col = self._add(trainX,col,dSex,labels,'sex',idxFunc = self._allIdx,
                        txFunc=self._gendTx)
        # add the age(5)
        col = self._add(trainX,col,dAge,labels,'age',idxFunc = self._empIdx,
                        txFunc=self._floatTx,norm=True)
        # add the siblings (6)
        col = self._add(trainX,col,dSib,labels,'nSib',norm=True)
        # add the parents (7)
        col = self._add(trainX,col,dPar,labels,'nParents',norm=True)
        # XXX skip the ticket (8)
        # add the fare (9)
        col = self._add(trainX,col,dFare,labels,'fare',idxFunc=self._empIdx,
                        txFunc=self._floatTx,norm=True) 
        # number of cabins skip the cabin (10) if non empty
        col = self._addEngr(trainX,col,self._hasCabin(dCab),labels,
                            'HasCabin')
        col = self._add(trainX,col,dCab,labels,'cabinNum',idxFunc=self._empIdx,
                        txFunc=self._numCabins,norm=True) 
        col = self._add(trainX,col,dCab,labels,'cabinLevel',
                        idxFunc=self._empIdx,txFunc=self._cabLevel,norm=False)
        # add the port (11) if non empty
        col = self._add(trainX,col,dEmb,labels,'port',txFunc=self._embTx,
                        idxFunc=self._empIdx)
        # XXX engineered stats
        # first stat: name length. 
        col = self._addEngr(trainX,col,[len(s) for s in dName],labels,
                            'nameLen',norm=True) 
        col = self._addEngr(trainX,col,self._hasSecondName(dName),labels,
                            'HasSecondName')
        col = self._add(trainX,col,dName,labels,"SecondNameLen",
                        idxFunc=self._secondNameIdx,txFunc=self._secondNameLen)
        col = self._addEngr(trainX,col,self._hasNickName(dName),labels,
                            'HasNickName')
        col = self._add(trainX,col,dName,labels,"NickNameLen",
                        idxFunc= lambda x: self._toIdx(self._hasNickName(x)),
                        txFunc=self._nickNameLength)
        col = self._addEngr(trainX,col,self._ageEstimated(dAge),labels,'ageEst')
        col = self._addEngr(trainX,col,self._ageKnown(dAge),labels,'ageKnown')
        col = self._addEngr(trainX,col,self._portUnknown(dEmb),labels,'embark')
        col = self._add(trainX,col,dAge,labels,'child',
                        idxFunc=self._empIdx,txFunc=self._isChild,
                        norm=False)
        col = self._add(trainX,col,dAge,labels,'infant',
                        idxFunc=self._empIdx,txFunc=self._isInfant,
                        norm=False)
        col = self._addEngr(trainX,col,self._hasSiblings(dSib),labels,
                            'siblings')
        col = self._addEngr(trainX,col,self._hasParents(dPar),labels,'parents')
        col = self._add(trainX,col,dAge,labels,'elderly',
                        idxFunc=self._empIdx,txFunc=self._isElderly,
                        norm=False)
        col = self._addEngr(trainX,col,self._isFirstClass(dClass),
                            labels,'1stClass')
        col = self._addEngr(trainX,col,self._isSecondClass(dClass),
                            labels,'2ndClass')
        col = self._addEngr(trainX,col,self._isThirdClass(dClass),
                            labels,'3rdClass')
        col = self._addEngr(trainX,col,self._highFare(dFare),labels,'HighFare')
        col = self._addEngr(trainX,col,self._fareUnknown(dFare),labels,'NoFare')
        col = self._addEngr(trainX,col,self._highSiblings(dSib),
                            labels,'>3Siblings')
        col = self._addEngr(trainX,col,self._hasPrefixFeature(dTicket),
                            labels,'hasPrefix')
        col = self._add(trainX,col,dTicket,labels,'TicketPrefix',
                        txFunc=self._getTicketPrefix,idxFunc=self._hasPrefixIdx,
                        categoryLabels=self._getPrefixes())
        col = self._add(trainX,col,dTicket,labels,'PrefixSurv',
                        txFunc=self._prefixInSurv,idxFunc=self._hasPrefixIdx)
        col = self._add(trainX,col,dTicket,labels,'PrefixDec',
                        txFunc=self._prefixInDec,idxFunc=self._hasPrefixIdx)
        col = self._add(trainX,col,dTicket,labels,'PrefixA5',
                        txFunc=self._prefixA5,idxFunc=self._hasPrefixIdx)
        col = self._add(trainX,col,dCab,labels,'cabinHigh',idxFunc=self._empIdx,
                        txFunc=self._highCab,norm=True) 
        col = self._add(trainX,col,dEmb,labels,'portFirst',
                        txFunc=self._embFirst,idxFunc=self._empIdx)
        col = self._addEngr(trainX,col,self._smallNameLen(dName),
                            labels,'smallName')
        col = self._addEngr(trainX,col,self._longNameLen(dName),
                            labels,'longName')
        norGate = lambda x,y : (1.-x) * (1.-y)
        col = self._addBigramByName(trainX,labels,'cabinHigh','1stClass',col,
                                    logic=norGate)
        return trainX,trainY,labels,col
    @abc.abstractmethod
    def _getXandY(self,data,test=False):
        # implement in your specialized class! should return 
        # x matrix : csr_matrix((nPassengers,nStats),dtype=np.float64)
        # y matrix : np.array(nPassengers)
        # labels: array of feature objects (Feat)
        # --- you should use _addEngr / _add methods to make this not horrible
        pass
    def _maskArr(self,array,columns):
        return array[:,columns]
    def mask(self,columns):
        # pass in the [0,1,3,6], takes columns 0,1,3,6
        self._trainX     = self._maskArr(self._trainX,columns)
        self._trainNames = self._maskArr(self._trainNames,columns)
        self._validX     =self._maskArr(self._validX,columns)
        self._validNames = self._maskArr(self._validNames,columns)
    def _createData(self,profileName,trainSize,validSize):
        self._trainX, self._trainY,self._trainObj,maxColIdx = \
                                self._getXandY(self._trainRaw,self._test)
        # prune possible duplicates
        self._trainX = self._trainX[:,:maxColIdx]
        # only create the validation data if it exists...
        if (validSize > 0):
            self._validX, self._validY,self._validObj,maxColIdx = \
                    self._getXandY(self._validRaw,self._test)
            self._validX = self._validX[:,:maxColIdx]
        if (profileName is not None):
            self.profileSelf(profileName,"feature_")
    def _shuffleAndPopulate(self,profileName=None):
        nPassengers = self._allData.shape[0]
        rowIdx = range(nPassengers)
        np.random.shuffle(rowIdx)
        # randomly shuffle the data
        self._allData = self._allData[rowIdx,:]
        self._id = [int(i) for i in self._allData[:,0]]
        # get the train and validation sizes
        validSize = 0
        if (self._valid is not None):
            validSize = int(nPassengers * self._valid)
        trainSize = nPassengers - validSize
        self._trainRaw = self._allData[:trainSize,:]
        self._validRaw = self._allData[trainSize:,:]
        # XXX: make shuffling faster, if we don't need to re-engineer.
        self._createData(profileName,trainSize,validSize)

    def __init__(self,dataInfoDir,data,valid=None,test=False,
                 profileName=None):
        # save the data, how much validation to use (0->1), if this is a test
        # and where to save the profiling information
        self._allData = data
        self._valid = valid
        self._test = test
        self._profileName = profileName
        self._shuffleAndPopulate(profileName)
        
    def _addLabelTicksIfNeeded(self,obj,xRange):
        mLab =obj._labels 
        if (mLab is not None):
            ax = plt.gca()
            ax.set_xticks(range(len(mLab)))
            ax.set_xticklabels(mLab,rotation='vertical')

    def profileSelf(self,outDir,label,train=True):
        # PRE: must have called constructor..
        # used for non-testing, to look at the distribution of data...
        mData = self._trainX     if train else self._validX
        mY    = self._trainY     if train else self._validY
        colObj= self._trainObj if train else self._validObj
        colIds=[c.label() for c in colObj]        
        maxCols = len(colIds)
        data = self._columnWise(mData,limit=maxCols)
        for name,dCol,obj in zip(colIds,data,colObj):
            fig = pPlotUtil.figure()
            # use 1 subplot if we dont have labels, otherwise use both
            subplots = 1 if self._test else 2
            counter = 1
            ax = plt.subplot(subplots,1,counter)
            tmpData =dCol.toarray()
            uniqueElements = np.unique(tmpData)
            minV = min(tmpData)
            maxV = max(tmpData)
            rangeV = maxV-minV
            if (rangeV < 1e-12):
                print("Feature {:s} is messed up...".format(name))
                continue
            minChanges = max(rangeV/50.,
                             np.median(np.diff(np.sort(uniqueElements))))
            bins = np.arange(minV,maxV*1.05,minChanges/2)
            opt=dict(alpha=0.5,log=True,bins=bins,align='left')
            vals, edges, patch = plt.hist(tmpData,label=name,**opt)
            ylimit = [0.5,max(vals)*1.05]
            xlimit = [-maxV*1.05,maxV*1.05]
            plt.ylim(ylimit)
            plt.xlim(xlimit)
            self._addLabelTicksIfNeeded(obj,xlimit)
            plt.ylabel("Occurence of {:s}".format(name))
            plt.legend()
            counter += 1
            if (not self._test):
                ax = plt.subplot(subplots,1,counter)
                surviveIdx = [ i  for i in range(len(mY)) if mY[i] == 1 ]
                notSurvIdx = [ i  for i in range(len(mY)) if mY[i] == 0 ]
                plt.hist(tmpData[surviveIdx],label='survived (y=1)',color='r',
                         **opt)
                plt.hist(tmpData[notSurvIdx],label='deceased (y=0)',color='g',
                         **opt)
                plt.ylim(ylimit)
                plt.xlim(xlimit)
                self._addLabelTicksIfNeeded(obj,xlimit)
                plt.legend()
            plt.xlabel('value of {:s}'.format(name))
            titleStr ="Histgram for Feature: {:s}".format(name)
            if (obj.isNorm()):
                titleStr += "\n" + obj.statStr()
            plt.title(titleStr)
            pPlotUtil.savefig(fig,outDir+label+name)
