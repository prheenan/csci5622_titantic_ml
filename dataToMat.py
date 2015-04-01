# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
# need to add the utilities class. Want 'home' to be platform independent
import sys
sys.path.append("./util/")
# import the patrick-specific utilities
import GenUtilities  as pGenUtil
import PlotUtilities as pPlotUtil
import CheckpointUtilities as pCheckUtil
from scipy.sparse import csr_matrix
import re

class Feat:
    def __init__(self,data,name,isNorm=False,mean=None,std=None):
        if (mean is None or std is None):
            self._mean = np.mean(data)
            self._std  = np.std(data)
        else:
            self._mean = mean
            self._std = std
        self._name = name
        self._norm = isNorm
    def isNorm(self):
        return self._norm
    def statStr(self):
        return "Mean/std: {:.2g}/{:.1g}".format(self._mean,self._std)
    def __str__(self):
        return self._name

class ShipData:
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
        levels = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8}
        toRet = np.zeros(len(cabins))
        for i,c in enumerate(cabins):
            allCabins = c.split(' ')
            levelLetter = [s[0] for s in allCabins]
            meanLevel = np.mean( [levels[s] for s in levelLetter] )
            toRet[i] = meanLevel
        return toRet
    def _gendTx(self,arr):
        # binary male/female
        return self._dictTx(arr,{'male':0, 'female':1 })
    def _embTx(self,arr):
        # embark destination
        return self._dictTx(arr,{'C':0, 'Q':1, 'S':2})
    def _addTicketBools(self,toAddTo,col,arr,labels):
        # transform ticket destinations (?)
        prefixes = []
        toRet = np.zeros(len(arr))
        process = lambda x : x.replace('.','').replace('/','').upper()
        for i,a in enumerate(arr):
            tix = a.split(" ")
            if (len(tix) == 1):
                toRet[i] = 0
            else:
                prefixes.append(process(tix[0]))
        prefixes = sorted(set(prefixes))
        numTix = len(arr)
        tmp = np.zeros(numTix)
        for j,a in enumerate(arr):
            split = a.split(" ")
            if (len(split) == 0 or process(split[0]) not in prefixes):
                tmp[j] = -1
            else:
                tmp[j] = prefixes.index(process(split[0]))
        toAddTo[:,col] = tmp[:,None]
        labels[col] = Feat(tmp,'prefix')
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
    def _hasNickname(self,names):
        has = [ '''"''' in s for s in names ]
        return has
    def _hasMaiden(self,names):
        # match an enclosed name in parenthesis. don't match quotes
        regex = re.compile(r'''\([^\(\"]+\)''',re.VERBOSE)
        toRet = [ len(re.findall(regex,s)) > 0 for s in names]
        return toRet
    def _isInfant(self,ages):
        return [1 if len(a) >0 and float(a) < 2.0 else 0 for a in ages]
    def _isChild(self,ages):
        return [1 if len(a) >0 and float(a) < 10.0 else 0 for a in ages]
    def _hasSiblings(self,count):
        return [1 if int(c) > 0 else 0 for c in count]
    def _hasParents(self,count):
        return [1 if int(c) > 0 else 0 for c in count]
    def _ageUnknown(self,ages):
        return [1 if len(a) == 0  else 0 for a in ages ]
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
        return [len(fare) == 0 for fare in fares ]
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
    def _add(self,mArr,data,arrCol,nameArr,name,idxFunc = None,
             txFunc = None,norm=False):
        # add to the sparse internal matrix at col 'column' using data from
        # col 'column'
        if txFunc is None:
            txFunc = self._intTx
        if idxFunc is None:
            idxFunc = self._allIdx
        goodIndices = idxFunc(data)
        goodData = np.array(txFunc(data[goodIndices]))
        return self._addEngr(mArr,arrCol,goodData,nameArr,name,norm,goodIndices)
    def _addEngr(self,toAddTo,col,data,nameArr,name,norm=False,indices=None):
        toAdd = np.array(data)
        mean = np.mean(toAdd)
        std = np.std(toAdd)
        if (norm):
            toAdd,mean,std = self._safeNorm(toAdd)
        finalDat = np.reshape(toAdd,((toAdd.size),1))
        if (indices is None):
            toAddTo[:,col] = finalDat
        else:
            toAddTo[indices,col] = finalDat
        nameArr[col] = Feat(finalDat,name,norm,mean,std)
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
    def _getXandY(self,data,test=False):
        # XXX add in support for testing data
        nPassengers = data.shape[0]
        dataStats = 9
        nPrefix = 1
        engineeredStats = 17 + nPrefix
        nStats = dataStats+engineeredStats
        trainX = csr_matrix((nPassengers,nStats),dtype=np.float64)
        labels = np.empty((nStats),dtype=np.object)
        # columns go like: 
#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embark
        if (not test):
            xx,yRaw,dClass,dName,dSex,dAge,dSib,dPar,dTicket,dFare,dCab,\
            dEmb = self._columnWise(data)
            trainY = [int(i) for i in yRaw]
        else:
            # there is no ID on the y column...
            xx,dClass,dName,dSex,dAge,dSib,dPar,dTicket,dFare,dCab,\
            dEmb = self._columnWise(data)
            trainY = 0
        # add the class (2)
        col = 0
        col = self._add(trainX,dClass,col,labels,'class')
        # add the sex (4)
        col = self._add(trainX,dSex,col,labels,'sex',idxFunc = self._allIdx,
                        txFunc=self._gendTx)
        # add the age(5)
        col = self._add(trainX,dAge,col,labels,'age',idxFunc = self._empIdx,
                        txFunc=self._floatTx,norm=True)
        # add the siblings (6)
        col = self._add(trainX,dSib,col,labels,'nSib',norm=True)
        # add the parents (7)
        col = self._add(trainX,dPar,col,labels,'nParents',norm=True)
        # XXX skip the ticket (8)
        # add the fare (9)
        col = self._add(trainX,dFare,col,labels,'fare',idxFunc=self._empIdx,
                        txFunc=self._floatTx,norm=True) 
        # number of cabins skip the cabin (10) if non empty
        col = self._add(trainX,dCab,col,labels,'cabinNum',idxFunc=self._empIdx,
                        txFunc=self._numCabins,norm=True) 
        col = self._add(trainX,dCab,col,labels,'noCabin',idxFunc=self._empIdx,
                        txFunc=self._cabLevel,norm=False)
        # add the port (11) if non empty
        col = self._add(trainX,dEmb,col,labels,'port',txFunc=self._embTx,
                        idxFunc=self._empIdx)
        # XXX engineered stats
        # first stat: name length. 
        col = self._addEngr(trainX,col,[len(s) for s in dName],labels,
                            'nameLen',norm=True) 
        col = self._addEngr(trainX,col,self._hasNickname(dName),labels,
                            'nickname')
        col = self._addEngr(trainX,col,self._hasMaiden(dName),labels,'maiden')
        col = self._addEngr(trainX,col,self._ageEstimated(dAge),labels,'ageEst')
        col = self._addEngr(trainX,col,self._ageUnknown(dAge),labels,'ageUnk')
        col = self._addEngr(trainX,col,self._portUnknown(dEmb),labels,'embark')
        col = self._addEngr(trainX,col,self._isChild(dAge),labels,'child')
        col = self._addEngr(trainX,col,self._isInfant(dAge),labels,'infant')
        col = self._addEngr(trainX,col,self._hasSiblings(dSib),labels,
                            'siblings')
        col = self._addEngr(trainX,col,self._hasParents(dPar),labels,'parents')
        col = self._addEngr(trainX,col,self._isElderly(dAge),labels,'elderly')
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
        col = self._addTicketBools(trainX,col,dTicket,labels)
        return trainX,trainY,labels
    def mask(self,columns):
        self._trainX     = self._trainX[:,columns]
        self._trainNames = self._trainNames[columns]
        self._validX     = self._validX[:,columns]
        self._validNames = self._validNames[columns]

    def __init__(self,dataInfoDir,data,valid=None,test=False,
                 profileName=None):
        estAge = lambda x: (x - int(x)) < 1.e-7
        # assume rows are the number of passengers
        nPassengers = data.shape[0]
        # XXX for now, ignore names.
        validSize = 0
        np.random.shuffle(data)
        if (valid is not None):
            validSize = int(nPassengers * valid)
        trainSize = nPassengers - validSize
        self._id = [int(i) for i in data[:,0]]
        self._trainRaw = data[:trainSize,:]
        self._validRaw = data[trainSize:,:]
        self._trainX, self._trainY,self._trainObj = \
                        self._getXandY(self._trainRaw,test)
        # only create the validation data if it exists...
        if (validSize > 0):
            self._validX, self._validY,self._validObj = \
                        self._getXandY(self._validRaw,test)
        self._test = test
        if (profileName is not None):
            self.profileSelf(profileName,"feature_")
    def profileSelf(self,outDir,label,train=True):
        # PRE: must have called constructor..
        # used for non-testing, to look at the distribution of data...
        mData = self._trainX     if train else self._validX
        mY    = self._trainY     if train else self._validY
        colObj= self._trainObj if train else self._validObj
        colIds=[str(c) for c in colObj]        
        maxCols = len(colIds)
        data = self._columnWise(mData,limit=maxCols)
        for name,dCol,obj in zip(colIds,data,colObj):
            fig = pPlotUtil.figure()
            # use 1 subplot if we dont have labels, otherwise use both
            subplots = 1 if self._test else 2
            counter = 1
            ax = plt.subplot(subplots,1,counter)
            tmpData =dCol.toarray() 
            bins = np.linspace(0,max(tmpData),10,endpoint=True)
            opt=dict(alpha=0.5,align='left',bins=bins,log=True)
            vals, edges, patch = plt.hist(tmpData,label=name,**opt)
            ylimit = [0,max(vals)*1.05]
            plt.ylim(ylimit)
            xlimit = [min(tmpData)-0.1*abs(min(tmpData)),max(tmpData)*1.05]
            plt.xlim(xlimit)
            plt.ylabel("Occurence of {:s}".format(name))
            plt.legend()
            counter += 1
            if (not self._test):
                ax = plt.subplot(subplots,1,counter)
                surviveIdx = [ i  for i in range(len(mY)) if mY[i] == 1 ]
                notSurvIdx = [ i  for i in range(len(mY)) if mY[i] == 0 ]
                plt.hist(tmpData[surviveIdx],label='survived (y=1)',color='r',
                         **opt)
                plt.hist(tmpData[notSurvIdx],label='deceased (y=0)',color='b',
                         **opt)
                plt.ylim(ylimit)
                plt.xlim(xlimit)
                plt.legend()
            plt.xlabel('value of {:s}'.format(name))
            titleStr ="Histgram for Feature: {:s}".format(name)
            if (obj.isNorm()):
                titleStr += "\n" + obj.statStr()
            plt.title(titleStr)
            pPlotUtil.savefig(fig,outDir+label+name)
