pipe_fileIdx = 0
pipe_funcIdx = 1

import GenUtilities as pGenUtil
import numpy as np
import pickle
from scipy.sparse import csc_matrix


def getCheckpoint(filePath,orCall,force,*args,**kwargs):
    # use the npz fil format, unpack arguments in the order they
    # are returned by 'orCall'. most 'intuitive', maybe less flexible
    print('Checkpoint: {:s} via {:s}'.format(filePath,str(orCall)))
    return _checkpointGen(filePath,orCall,force,True,False,*args,**kwargs)

def _npyLoad(filePath,unpack):
    data  = np.load(filePath)
    if (unpack == True):
        keys = sorted(data.keys())
        # return either the single element, or the in-save-order list
        if (len(keys)) == 1:
            return data[keys[0]]
        else:
            return tuple(data[key] for key in keys)
    else:
        return data

def _npySave(filePath,dataToSave):
    if (type(dataToSave) is tuple):
        np.savez(filePath,*dataToSave)
    else:
        np.savez(filePath,dataToSave)

def _checkpointGen(filePath,orCall,force,unpack,useNpy,*args,**kwargs):
    # XXX assume pickling now, ends with 'npz'
    # if the file from 'filePath' exists and 'force' is false, loads the file
    # otherwise, calls 'orCall' and saves the result. *args and **kwargs
    # are passed to 'orCall'.
    # 'Unpack' unpacks the array upon a load. This makes it 'look' like a 
    # simple function call (returns the args, or a tuple list of args)
    # use unpack if you aren't dealing with dictionaries or things like that
    if pGenUtil.isfile(filePath) and not force:
        if (useNpy):
            return _npyLoad(filePath,unpack)
        else:
            # assume we pickle in binary
            fh = open(filePath,'rb')
            data = pickle.load(fh)
            fh.close()
            return data
    else:
        # couldn't find the file.
        # make sure it exists
        path = pGenUtil.getBasePath(filePath)
        pGenUtil.ensureDirExists(path)
        # POST: we can put our file here
        dataToSave = orCall(*args,**kwargs)
        # need to figure out if we need to unpack all the arguments..
        if (useNpy):
            _npySave(filePath,dataToSave)
        else:
            # open the file in binary format for writing
            with open(filePath, 'wb') as fh:
                pickle.dump(dataToSave,fh)
        return dataToSave

def _pipeHelper(objectToPipe,force,useNpy,otherArgs = None):
    # sets up all the arguments we need. 
    args = []
    # add all the arguments we need
    if (otherArgs is not None):
        # XXX this might not work so well with multiple arguments
        # passing betwene them.
        if (not (isinstance(otherArgs, (list,tuple)))):
            # if we have some thing which is not a list or a tuple, just add
            args.append(otherArgs)
        else:
             # otherwise, add elements one at a time.
            args.extend(otherArgs)
    customArgs = objectToPipe[pipe_funcIdx+1:]
    if (len(customArgs) > 0):
        for cArg in customArgs:
            args.append(cArg)
    # POST: have all the arguments we need in 'args'
    return _checkpointGen(objectToPipe[pipe_fileIdx],
                          objectToPipe[pipe_funcIdx],
                          force,True,useNpy,
                          *args)

def _pipeListParser(value,default,length):
    if value is None:
        safeList= [default] * length
    elif type(value) is not list:
        safeList = [value] * length
    else:
        # must be a non-None list
        return value
    return safeList 

def pipeline(objects,force=None):
    # objects are a list, each element is : [<file>,<function>,<args>]: 
    # file name,
    # function then the ('extra' args the funcion
    # needs. we assume that each filter in the pipeline takes
    # the previous arguments, plus any others, and returns the next arg
    # the first just takes in whatever it is given, the last can return anything
    # in other words, the signatures are:
    # f1(f1_args), returning f2_chain
    # f2(f2_chain,f2_args), returning f3_chain
    # ...
    # fN(fN_chain,fNargs), returning whatever.

    filesExist = [pGenUtil.isfile(o[pipe_fileIdx]) for o in objects]
    numObjects = len(objects)
    # get a list of forces
    force = _pipeListParser(force,False,numObjects)
    # get a list of how to save.
    numpy = [ not o[pipe_fileIdx].endswith('.pkl') for o in objects] 
    # by default, if no force arguments passed, assume we dont want to force
    # in other words: just load by default

    runIfFalse = [ fExists and (not forceThis)  
                 for fExists,forceThis in zip(filesExist,force)]
    if (False not in runIfFalse):
        # just load the last...
        otherArgs = _pipeHelper(objects[-1],False,numpy[-1])
    else:
        # need to run at least one, go through them all
        otherArgs = None
        firstZero = runIfFalse.index(False)
        # if not at the start, load 'most downstream'
        if (firstZero != 0):
            idx = firstZero-1
            otherArgs = _pipeHelper(objects[idx],
                                    force[idx],numpy[idx],
                                    otherArgs)
        # POST: otherargs is set up, if we need it.
        for i in range(firstZero,numObjects):
            otherArgs = _pipeHelper(objects[i],force[i],numpy[i],otherArgs)
    return otherArgs

