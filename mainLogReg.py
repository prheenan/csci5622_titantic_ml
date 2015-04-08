import base.titanicMain as main
from base.dataToMat_logreg import dataToMat_logreg  as lrClass

main.run(main.defaultFitter,main.defaultFitterParams,main.defaultCoeff,
         lrClass,valid=0.05)
