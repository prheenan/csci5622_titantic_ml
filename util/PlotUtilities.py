# import utilities for error repoorting etc
import GenUtilities as util
# use matplotlib for plotting
#http://matplotlib.org/faq/usage_faq.html#what-is-a-backend
import matplotlib.pyplot  as plt
# import numpy for array stuff
import numpy as np
# for color cycling
from itertools import cycle
import sys


def cmap(num,cmap = plt.cm.brg):
    return cmap(np.linspace(0, 1, num))

def useTex():
    # may need to install:
    # tlmgr install dvipng helvetic palatino mathpazo type1cm
    # http://stackoverflow.com/questions/14389892/ipython-notebook-plotting-with-latex
    from matplotlib import rc
    sys.path.append("/usr/texbin/")
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)

# add a second axis to ax.
def secondAxis(ax,label,limits,secondY =True,yColor="Black",scale=None):
    #copies the first axis (ax) and uses it to overlay an axis of limits
    if (scale is None):
        if secondY:
            scale = ax.get_yscale() 
        else:
            scale = ax.get_xscale()
    if(secondY):
        ax2 = ax.twinx()
        ax2.set_yscale(scale, nonposy='clip')
        ax2.set_ylim(limits)
        # set the y axis to the appropriate label
        lab = ax2.set_ylabel(label)
        ticks = ax2.get_yticklabels()
    else:
        ax2 = ax.twiny()
        ax2.set_xscale(scale, nonposy='clip')
        ax2.set_xlim(limits)
        # set the x axis to the appropriate label
        lab = ax2.set_xlabel(label)
        ticks = ax2.get_xticklabels()
    [i.set_color(yColor) for i in ticks]
    lab.set_color(yColor)
    return ax2

def pm(stdOrMinMax,mean=None,fmt=".3g"):
    if (mean ==None):
        mean = np.mean(stdOrMinMax)
    arr = np.array(stdOrMinMax)
    if (len(arr) == 1):
        delta = arr[0]
    else:
        delta = np.mean(np.abs(arr-mean))
    return ("{:"+ fmt + "}+/-{:.2g}").format(mean,delta)

def savefig(figure,fileName,close=True,tight=True):
    # source : where to save the output iunder the output folder
    # filename: what to save the file as. automagically saved as high res pdf
    # override IO: if true, ignore any path infomation in the file name stuff.
    # close: if true, close the figure after saving.
    if (tight):
        plt.tight_layout(True)
    formatStr = "png"
    figure.savefig(fileName + '.' + formatStr,format=formatStr, 
                   dpi=figure.get_dpi())
    if (close):
        plt.close(figure)

def figure(xSize=10,ySize=8,dpi=100):
    return  plt.figure(figsize=(xSize,ySize),dpi=dpi)

def getNStr(n,space = " "):
    return space + "n={:d}".format(n)


# legacy API. plan is now to mimic matplotlib 
def colorCyc(num,cmap = plt.cm.winter):
    cmap(num,cmap)
def pFigure(xSize=10,ySize=8,dpi=100):
    return figure(xSize,ySize,dpi)
def saveFigure(figure,fileName,close=True):
    savefig(figure,fileName,close)
