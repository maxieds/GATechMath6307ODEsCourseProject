#### RosslerGenLyapunovExponentExperiments.py
#### Maxie Dion Schmidt
#### 2021.10.23

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
import sympy
import sys
import math
import cmath
import itertools
from sage.all import *
from PythonODEBaseLibrary import *
from Utils import * 
from sage.plot.plot3d.transform import rotate_arbitrary

RR = RealField(sci_not=0, prec=4, rnd='RNDU')
PP = PolynomialRing(RR, 3, "txy")
T, X, Y = PP.gens()
XVAR, YVAR = var('x y')
AVAR = A = var('za')
XP = -Y
YP = X + A * Y 

DEFAULT_IC=(0, (1, 1))
DEFAULT_XYH=0.075  #0.25 #0.050
DEFAULT_AH=0.075   #0.1  #0.025
DEFAULT_HIST_BINS=63
DEFAULT_V0=(0, -1)
DEFAULT_ATRANGE=(-15.0, 15.0)

PLOT_SUMMARY_TEMP_OUTFILE='../../Images/TempOutputData/RosslerAttractorExpt2-%s.png'
stampOutputFilePathFunc = lambda: PLOT_SUMMARY_TEMP_OUTFILE % Utils.GetTimestamp()

def GetXYSolutionsGrid(icPoint, h):
    (t0, (x0, y0)) = icPoint
    fFunc = lambda tv, xv, yv: XP.subs({ Y : yv, YVAR : yv })
    gFunc = lambda tv, xv, yv: YP.subs({ X : xv, XVAR : xv, Y : yv, YVAR : yv })
    odeSol = np.array(eulers_method_2x2(fFunc, gFunc, t0, x0, y0, h, 1, algorithm="none"))
    return (odeSol[:, 0], list(zip(odeSol[:, 1], odeSol[:, 2])))

def PlotFirstXVarProductDiagram2D(xPoints, ax):
    NN = float(len(xPoints))
    xpointFunc = lambda xpa, aa: xpa if not hasattr(xpa, 'subs') else xpa.subs({ A : aa, AVAR : aa })
    YaFunc = lambda aa: 1 / NN * simplify(sum([ log(Utils.SageMathNorm(xpointFunc(xpa, aa))) for xpa in xPoints ]))
    numAPoints = math.floor((DEFAULT_ATRANGE[1] - DEFAULT_ATRANGE[0]) / DEFAULT_AH) + 1
    AA = np.linspace(DEFAULT_ATRANGE[0], DEFAULT_ATRANGE[1], numAPoints)
    YY = [ YaFunc(avalue) for avalue in AA ]
    ax.plot(AA, YY, 'b--')
    plt.sca(ax)
    plt.xlabel('a')
    plt.ylabel(r'$\frac{1}{N} \times \log\left(\prod_{0 \leq i < N} x_i(a)\right)$ for $N = %d$' % NN, rotation=90)
    plt.title('Experiment 2: Normalized X-point-log-product component')
    return ax

def PlotSecondXYPdfDensityDiagram3D(xyPoints, axArr):
    xyRatioFunc = lambda xx, yy: yy / xx
    thetaRatios = [ xyRatioFunc(xp, yp) for (xp, yp) in xyPoints ]
    tmin, tmax = DEFAULT_ATRANGE[0], DEFAULT_ATRANGE[1]
    numTPoints = DEFAULT_HIST_BINS + 1
    tSpec = np.linspace(tmin, tmax, numTPoints)
    aSpec = tSpec
    AA, TT, PDF = [], [], []
    apointCtr = 0
    for avalue in aSpec:
        try:
            thetaRatiosA = [ tr if isinstance(tr, float) else tr.subs({ A : avalue, AVAR : avalue }) for tr in thetaRatios ]
        except ValueError:
            print("TAKE NOTE OF SINGULARITY: Reached a NaN error for A===%1.3f" % avalue)
            continue
        thetaHistA, histBinEdgesA = np.histogram(thetaRatiosA, bins=tSpec, density=True)
        for (pidx, thPdfPoint) in enumerate(list(thetaHistA)):
            AA += [ avalue ]
            TT += [ tSpec[pidx] ]
            PDF += [ thPdfPoint ]
        apointCtr += 1
    cmapOption = 'YlGnBu_r'
    alpha = 0.5
    rstride, cstride = 2, 2
    numReshapeRows, numReshapeCols = int(apointCtr), DEFAULT_HIST_BINS
    mgAA = np.array(AA).reshape(numReshapeRows, numReshapeCols)
    mgTT = np.array(TT).reshape(numReshapeRows, numReshapeCols)
    mgPDF = np.array(PDF).reshape(numReshapeRows, numReshapeCols)
    for (axIdx, ax) in enumerate(axArr):
        if axIdx == 0:
            ax.plot_trisurf(PDF, TT, AA, cmap='magma', edgecolor='none', alpha=alpha)
            ax.set_zlabel('a')
            ax.set_ylabel('t')
            ax.set_xlabel(r'$\operatorname{pdf}\left(\vartheta_a=\frac{x(a)}{y(a)}; t\right)$')
        elif axIdx == 1: 
            ax.plot_surface(mgPDF, mgTT, mgAA, cmap=cmapOption, edgecolor='none', alpha=alpha, rstride=rstride, cstride=cstride)
            ax.set_zlabel('a')
            ax.set_ylabel('t')
            ax.set_xlabel(r'$\operatorname{pdf}\left(\vartheta_a=\frac{x(a)}{y(a)}; t\right)$')
        elif axIdx == 2:
            ax.plot_trisurf(AA, TT, PDF, cmap='magma', edgecolor='none', alpha=alpha)
            ax.set_xlabel('a')
            ax.set_ylabel('t')
            ax.set_zlabel(r'$\operatorname{pdf}\left(\vartheta_a=\frac{x(a)}{y(a)}; t\right)$', rotation=90)
        elif axIdx == 3:
            ax.plot_surface(mgAA, mgTT, mgPDF, cmap=cmapOption, edgecolor='none', alpha=alpha, rstride=rstride, cstride=cstride)
            ax.set_xlabel('a')
            ax.set_ylabel('t')
            ax.set_zlabel(r'$\operatorname{pdf}\left(\vartheta_a=\frac{x(a)}{y(a)}; t\right)$', rotation=90)
        ax.set_title(r'Verify ansatz on limiting PDF(Y/X)')
        axArr[axIdx] = ax
    return axArr

if __name__ == "__main__":
    (tSpec, xyPoints) = GetXYSolutionsGrid(DEFAULT_IC, DEFAULT_XYH)
    (xPoints, yPoints) = list(zip(*xyPoints))
    fig  = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs   = GridSpec(2, 4, figure=fig)
    ax1  = fig.add_subplot(gs[0, :])
    ax1  = PlotFirstXVarProductDiagram2D(xPoints, ax1)
    ax2i = [ fig.add_subplot(gs[1, colIdx], projection='3d') for colIdx in range(0, 4) ]
    ax2i = PlotSecondXYPdfDensityDiagram3D(xyPoints, ax2i)
    axs  = [ ax1 ] + ax2i
    for ax in axs:
        ax.set_aspect('auto', adjustable='datalim')
    fig.tight_layout(pad=1)
    plt.savefig(stampOutputFilePathFunc())

