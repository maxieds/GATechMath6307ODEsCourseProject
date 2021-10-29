#### RosslerGenLyapunovExponentExperiments.py
#### Maxie Dion Schmidt
#### 2021.10.23

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
import sympy
import sys
import math
import cmath
from sage.all import *
from PythonODEBaseLibrary import *
from Utils import * 
import pandas as pd

RR = RealField(sci_not=0, prec=4, rnd='RNDU')
PP = PolynomialRing(RR, 3, "txy")
T, X, Y = PP.gens()
XVAR, YVAR = var('x y')
AVAR = A = var('za')
XP = -Y
YP = X + A * Y 

DEFAULT_IC=(0, (1, 1))
DEFAULT_XYH=0.02  #0.25 #0.050
DEFAULT_AH=0.1    #0.1  #0.025
DEFAULT_HIST_BINS=63
DEFAULT_HIST_TRANGE=(-1.0, 3.5)
DEFAULT_V0=(0, -1)

PLOT_SUMMARY_TEMP_OUTFILE='../../Images/TempOutputData/RosslerAttractorExpt2-V%d-%s.png'
stampOutputFilePathFunc = lambda atIntervalVersion: PLOT_SUMMARY_TEMP_OUTFILE % (atIntervalVersion, Utils.GetTimestamp())

def GetXYSolutionsGrid(icPoint, tRange, h):
    (t0, (x0, y0)) = icPoint
    (solA, solB) = tRange
    fFunc = lambda tv, xv, yv: XP.subs({ Y : yv, YVAR : yv })
    gFunc = lambda tv, xv, yv: YP.subs({ X : xv, XVAR : xv, Y : yv, YVAR : yv })
    odeSol = np.array(eulers_method_2x2(fFunc, gFunc, t0, x0, y0, h, solB, algorithm="none"))
    initTPoints, initXPoints, initYPoints = odeSol[:, 0], odeSol[:, 1], odeSol[:, 2]
    tPoints, xPoints, yPoints = [], [], []
    for (tidx, tpoint) in enumerate(list(initTPoints)):
        if solA <= tpoint and tpoint <= solB:
            tPoints += [ tpoint ]
            xPoints += [ initXPoints[tidx] ]
            yPoints += [ initYPoints[tidx] ]
    return (tPoints, list(zip(xPoints, yPoints)))

def PlotFirstXVarProductDiagram2D(xPoints, aRange, ax):
    NN = float(len(xPoints))
    xpointFunc = lambda xpa, aa: xpa if not hasattr(xpa, 'subs') else xpa.subs({ A : aa, AVAR : aa })
    YaFunc = lambda aa: 1 / NN * simplify(sum([ log(Utils.SageMathNorm(xpointFunc(xpa, aa))) for xpa in xPoints ]))
    (amin, amax) = aRange
    numAPoints = math.floor((amax - amin) / DEFAULT_AH) + 1
    AA = np.linspace(amin, amax, numAPoints)
    YY = [ YaFunc(avalue) for avalue in AA ]
    ax.plot(AA, YY, 'b--')
    plt.sca(ax)
    plt.xlabel('a')
    plt.ylabel(r'$\frac{1}{N} \times \log\left(\prod_{0 \leq i < N} x_i(a)\right)$ for $N = %d$' % NN, rotation=90)
    plt.title('Experiment 2: Normalized X-point-log-product component')
    return ax

def PlotSecondXYPdfDensityDiagram3D(xyPoints, aRange, axArr):
    xyRatioFunc = lambda xx, yy: yy / xx
    thetaRatios = [ xyRatioFunc(xp, yp) for (xp, yp) in xyPoints ]
    numTPoints = DEFAULT_HIST_BINS + 1
    tSpec = np.linspace(DEFAULT_HIST_TRANGE[0], DEFAULT_HIST_TRANGE[1], numTPoints)
    (amin, amax) = aRange
    numAPoints = math.floor((amax - amin) / DEFAULT_AH) + 1
    aSpec = np.linspace(amin, amax, numAPoints)
    AA, TT, PDF = [], [], []
    apointCtr = 0
    for avalue in aSpec:
        try:
            thetaRatiosA = [ tr if isinstance(tr, float) else tr.subs({ A : avalue, AVAR : avalue }) for tr in thetaRatios ]
        except ValueError:
            print("TAKE NOTE OF SINGULARITY: Reached a NaN error for A===%1.3f" % avalue)
            continue
        thetaHistA, histBinEdgesA = np.histogram(thetaRatiosA, bins=tSpec, density=True)
        print("SORTED-HIST-VALUES: %s" % list(sorted(thetaRatiosA)))
        print("HIST-DATA (a = %g): %s\n" % (avalue, thetaHistA))
        ## Since there are only a finite, sparse number of initially filled bins, use an adaptive method 
        ## where we make the t-grid points finer around those points, with the idea that it should yield 
        ## more resolution in the plots generated below: 
        #nonzeroBinT = [ (bidx, histBinEdgesA[bidx]) for bidx in range(0, len(thetaHistA)) if thetaHistA[bidx] != 0.0 ] 
        #numRefinedIntPoints = math.floor(2.0 * len(tSpec) / len(nonzeroBinT)) + 1
        #tIntervalDelta = math.floor(1.5 * (DEFAULT_HIST_TRANGE[1] - DEFAULT_HIST_TRANGE[0]) / len(tSpec) * len(nonzeroBinT))
        #tSpecLst = list(tSpec)
        #refinedTSpec = [] 
        #for (bidx, (origTIdx, tvalue)) in enumerate(nonzeroBinT):
        #    refinedTSpec += tSpecLst[:nonzeroBinT[bidx][0]]
        #    lastMaxT = refinedTSpec[-1]
        #    refinedTIntMin, refinedTIntMax = max(lastMaxT, tvalue - tIntervalDelta), min(tvalue + tIntervalDelta, DEFAULT_HIST_TRANGE[1])
        #    refinedTSpec += list(np.linspace(refinedTIntMin, refinedTIntMax, numRefinedIntPoints))
        #nextTSpec = np.array(refinedTSpec)
        #print("NEXT-TSPEC: %s\n" % nextTSpec)
        #thetaHistA, histBinEdgesA = np.histogram(thetaRatiosA, bins=tSpec, density=True)
        #print("HIST-DATA-REFINED (a = %g): %s\n" % (avalue, thetaHistA))
        for (pidx, thPdfPoint) in enumerate(list(thetaHistA)):
            AA += [ avalue ]
            TT += [ tSpec[pidx] ]
            PDF += [ thPdfPoint ]
        apointCtr += 1
    cmapOption = 'flag'
    alpha = 0.5
    rstride, cstride = 1, 1
    rgiV = np.zeros((len(AA), len(TT), len(PDF)))
    for xi in range(0, len(AA)):
        for yi in range(0, len(TT)):
            for zi in range(0, len(PDF)):
                rgiV[xi, yi, zi] = 100*AA[xi] + 10*TT[yi] + PDF[zi]
    #dataFrame = pd.DataFrame(dict(x=AA, y=TT, z=PDF))
    #xcol, ycol, zcol = 'x', 'y', 'z'
    #mgAA, mgTT = dataFrame[xcol].unique(), dataFrame[ycol].unique()
    #mgPDF = dataFrame[zcol].values.reshape(len(mgAA), len(mgTT)).T
    #mgAA, mgTT = np.meshgrid(mgAA, mgTT)
    mgAA, mgTT = np.meshgrid(AA, TT)
    plot3dGridInterpFunc = LinearNDInterpolator(list(zip(AA, TT)), PDF)
    mgZZ = plot3dGridInterpFunc(mgXX, mgYY)
    for (axIdx, ax) in enumerate(axArr):
        if axIdx == 0:
            ax.tricontourf(PDF, TT, AA, cmap='magma', edgecolor='none', alpha=alpha, antialiased=False)
            ax.set_zlabel('a')
            ax.set_ylabel('t')
            ax.set_xlabel(r'$\operatorname{pdf}\left(\vartheta_a=\frac{x(a)}{y(a)}; t\right)$')
        if axIdx == 1:
            plt.sca(ax)
            plt.tricontourf(AA, TT, PDF, cmap='magma')
            ax.set_xlabel('a')
            ax.set_ylabel('t')
            ax.set_zlabel(r'$\operatorname{pdf}\left(\vartheta_a=\frac{x(a)}{y(a)}; t\right)$')
        elif axIdx == 2:
            ax.plot_trisurf(AA, TT, PDF, cmap='magma', edgecolor='none', alpha=alpha, antialiased=False)
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
    aRanges = [ (-1.75, -0.25), (0.25, 1.75) ]
    tRanges = [ (0, float(9.0 / 20.0)), (float(9.0 / 20.0), 1.0) ]
    atFigureCtr = 1
    for tGridRange in tRanges:
        (tSpec, xyPoints) = GetXYSolutionsGrid(DEFAULT_IC, tGridRange, DEFAULT_XYH)
        (xPoints, yPoints) = list(zip(*xyPoints))
        for aGridRange in aRanges[1:]:
            fig  = plt.figure(figsize=(16, 8), constrained_layout=True)
            gs   = GridSpec(2, 4, figure=fig)
            ax1  = fig.add_subplot(gs[0, :])
            ax1  = PlotFirstXVarProductDiagram2D(xPoints, aGridRange, ax1)
            ax2i = [ fig.add_subplot(gs[1, colIdx], projection='3d') for colIdx in range(0, 4) ]
            ax2i = PlotSecondXYPdfDensityDiagram3D(xyPoints, aGridRange, ax2i)
            axs  = [ ax1 ] + ax2i
            for ax in axs:
                ax.set_aspect('auto', adjustable='datalim')
            fig.tight_layout(pad=1)
            fig.subplots_adjust(top=0.8, bottom=0.2, right=0.9, left=0.1, hspace=0.5, wspace=0.5)
            summaryTitle = r'Rossler attractor experiment-v2 with ' + "\n" + \
                           r'$\mathbf{h = %1.3f}$, $\mathbf{\Delta a = %1.3f}$, $\mathbf{t \in (%1.3f, %1.3f)}$ and $\mathbf{a \in (%1.3f, %1.3f)}$' % \
                           (DEFAULT_XYH, DEFAULT_AH, tGridRange[0], tGridRange[1], aGridRange[0], aGridRange[1])
            fig.suptitle(summaryTitle, fontsize=18, fontweight='bold')
            plt.savefig(stampOutputFilePathFunc(atFigureCtr))
            atFigureCtr += 1

