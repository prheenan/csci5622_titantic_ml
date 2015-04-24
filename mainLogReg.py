import base.titanicMain as main
from base.dataToMat_logreg import dataToMat_logreg  as lrClass
from base.analysis import plotErrorAnalysis

label='logreg'
forceRun = True # otherwise, use checkpoint (cached file)
fullOutput = "./work/out/"+label+"/"
mean,std= main.run(main.defaultFitter,main.defaultFitterParams,
                   main.defaultCoeff,lrClass,label=label,valid=0.1,nTrials=10,
                   force=forceRun,plot=True)
plotErrorAnalysis([mean],[std],[main.defaultFitterParams()],[label],fullOutput)
    
