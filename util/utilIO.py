import numpy as np
import matplotlib.pyplot as plt
# need to add the utilities class. Want 'home' to be platform independent
# import the patrick-specific utilities
import sys
sys.path.append("./base/")
sys.path.append("../base/")

import GenUtilities  as pGenUtil
import PlotUtilities as pPlotUtil
import CheckpointUtilities as pCheckUtil
import csv as csv 
import argparse

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import AdaBoostClassifier     
from base.dataToMat import ShipData       

def getData(saveDir,fileName,valid=0.0,test=False,testSave=None):
# copied from (for IO): https://www.kaggle.com/c/titanic-gettingStarted/details/getting-started-with-python
    # Open up the csv file in to a Python object
    with open(fileName) as fh:
        csv_file_object = csv.reader(fh)
        header = csv_file_object.next()  # The next() command just skips the 
        # first line which is a header
        data=[]                          # Create a variable called 'data'.
        for row in csv_file_object:      # Run through each row in the csv file,
            data.append(row)             # adding each row to the data variable
        data = np.array(data) 
    return ShipData(saveDir,data,valid,test,testSave)

def getDirsFromCmdLine():
    parser = argparse.ArgumentParser(description='Protein Visualizatio args')
    parser.add_argument('--inPath', type=str, default="./data/",
                        help="Folder where formatted .dat file reside")
    parser.add_argument('--cachePath', type=str, default="../work/tmp/",
                        help="Default cache output directory (base)")    
    parser.add_argument('--outPath', type=str, default="../work/out/",
                        help="Default cache output directory (base)")    
    args = parser.parse_args()
    return args.inPath,args.cachePath,args.outPath

