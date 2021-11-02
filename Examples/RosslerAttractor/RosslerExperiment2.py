#### RosslerGenLyapunovExponentExperiments.py
#### Maxie Dion Schmidt
#### 2021.10.23

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.tri as tri
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
import sympy
import sys
import math
import cmath
from sage.all import *
from PythonODEBaseLibrary import *
from Utils import * 
import pandas as pd

#RR = RealField(sci_not=0, prec=4, rnd='RNDU')
#PP = PolynomialRing(RR, 3, "xyz")
#X, Y, Z = PP.gens()
XVAR, YVAR, ZVAR, T = var('x y z t')
#AVAR = A = var('za')
#XP = -Y
#YP = X + A * Y 

DEFAULT_IC=(0, (1, 1))
DEFAULT_XYH=0.005    #0.25 #0.050
DEFAULT_AH=0.005     #0.1  #0.025
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

def GetXYSolutionsGridForFixedA(avalue, icPoint, tRange, h):
    (t0, (x0, y0)) = icPoint
    fxyz = [ -YVAR, XVAR + avalue * YVAR ] 
    numT = math.floor((tRange[1] - tRange[0]) / h) + 1
    tSpec = list(np.linspace(tRange[0], tRange[1], numT))
    odeVars = [ XVAR, YVAR ]
    initConds = [ x0, y0 ]
    try:
        xyzPoints = desolve_odeint(fxyz, initConds, tSpec, odeVars)
    except Exception as excpt:
        raise excpt
    xSolPoints, ySolPoints = xyzPoints[:, 0], xyzPoints[:, 1]
    return (tSpec, list(zip(xSolPoints, ySolPoints)))

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

def PlotSecondXYPdfDensityDiagram3D_V2(aRange, tRange, axArr):
    (amin, amax) = aRange
    numA = math.floor((amax - amin) / DEFAULT_AH) + 1
    aSpec = np.linspace(amin, amax, numA)
    AA, TT, PDF = [], [], []
    AAnotdep, RV = [], []
    apointCtr = 0
    for avalue in list(aSpec):
        (rtSpec, xySolPoints) = GetXYSolutionsGridForFixedA(avalue, DEFAULT_IC, tRange, DEFAULT_XYH)
        xyRatioFunc = lambda xx, yy: yy / xx
        xPoints, yPoints = tuple(zip(*xySolPoints))
        thetaRatios = list(np.array([ [ xyRatioFunc(xp, yp) for yp in yPoints ] for xp in xPoints ]).flatten())
        numTPoints = min(math.floor(len(thetaRatios) / 3), DEFAULT_HIST_BINS) + 1
        tSpec = np.linspace(DEFAULT_HIST_TRANGE[0], DEFAULT_HIST_TRANGE[1], numTPoints)
        try:
            thetaRatiosA = [ tr if isinstance(tr, float) else tr.subs({ A : avalue, AVAR : avalue }) for tr in thetaRatios ]
        except ValueError:
            print("TAKE NOTE OF SINGULARITY: Reached a NaN error for A===%1.3f" % avalue)
            continue
        thetaHistA, histBinEdgesA = np.histogram(thetaRatiosA, bins=tSpec, density=True)
        #print("SORTED-HIST-VALUES: %s" % list(sorted(thetaRatiosA)))
        #print("HIST-DATA (a = %g): %s\n" % (avalue, thetaHistA))
        for (pidx, thPdfPoint) in enumerate(list(thetaHistA)):
            AA += [ avalue ]
            TT += [ tSpec[pidx] ]
            PDF += [ thPdfPoint ]
        AAnotdep += [ avalue ] * len(thetaRatios)
        RV += thetaRatios
        apointCtr += 1
    cmapOption = 'flag'
    alpha = 0.5
    rstride, cstride = 1, 1
    mgAA, mgTT = np.meshgrid(AA, TT)
    plot3dGridInterpFunc = LinearNDInterpolator(list(zip(AA, TT)), PDF)
    mgPDF = plot3dGridInterpFunc(mgAA, mgTT)
    for (axIdx, ax) in enumerate(axArr):
        plt.sca(ax)
        if axIdx == 0:
            levels = np.arange(0.0, 1.0, 0.025)
            plt.contourf(mgAA, mgTT, mgPDF, cmap='magma', alpha=0.5)
            plt.contour(mgAA, mgTT, mgPDF, levels=levels,
                        colors=['0.25', '0.5', '0.5', '0.5', '0.5'],
                        linewidths=[1.0, 0.5, 0.5, 0.5, 0.5])
            ax.set_ylabel('t')
            ax.set_xlabel(r'a')
        elif axIdx == 1:
            ax.pcolormesh(mgAA, mgTT, mgPDF, cmap='plasma_r', edgecolor='none', alpha=alpha, antialiased=False)
            ax.set_ylabel('t')
            ax.set_xlabel(r'a')
        elif axIdx == 2:
            triang = tri.Triangulation(AA, TT)
            plt.triplot(triang, lw=0.5, color='white')
            triRefiner = tri.UniformTriRefiner(triang)
            tri_refi, z_test_refi = triRefiner.refine_field(RV, subdiv=3)
            levels = np.arange(0.0, 1.0, 0.025)
            plt.tricontourf(tri_refi, z_test_refi, levels=levels, cmap='terrain')
            plt.tricontour(tri_refi, z_test_refi, levels=levels,
                           colors=['0.25', '0.5', '0.5', '0.5', '0.5'],
                           linewidths=[1.0, 0.5, 0.5, 0.5, 0.5])
        elif axIdx == 3:
            dt = 0.0005
            NFFT = 1024
            Fs = int(1.0 / dt)
            plt.specgram(np.array(np.dot(RV, AAnotdep)), NFFT=NFFT, Fs=Fs, noverlap=900)
        plt.colorbar(pad=0.65, shrink=0.685)
        axArr[axIdx] = ax
    return axArr

if __name__ == "__main__":
    aRanges = [ (-1.75, -0.25), (0.25, 1.75) ]
    tRanges = [ (0, float(9.0 / 20.0)), (float(9.0 / 20.0), 1.0) ]
    atFigureCtr = 1
    for tGridRange in tRanges[1:]:
        for aGridRange in aRanges[1:]:
            fig  = plt.figure(figsize=(16, 8), constrained_layout=True)
            gs = GridSpec(1, 3, figure=fig)
            ax2i = [ fig.add_subplot(gs[0, colIdx]) for colIdx in range(0, 3) ]
            ax2i = PlotSecondXYPdfDensityDiagram3D_V2(aGridRange, tGridRange, ax2i)
            axs = ax2i
            for ax in axs:
                ax.set_aspect('auto', adjustable='datalim')
            fig.tight_layout(pad=1)
            fig.subplots_adjust(top=0.8, bottom=0.2, right=0.9, left=0.1, hspace=0.5, wspace=0.5)
            summaryTitle = (r'Rossler attractor experiment-v2 with ' + "\n" + \
                            r'$\mathbf{h = %1.3f}$, $\mathbf{\Delta a = %1.3f}$, ' + \
                            r'$\mathbf{t \in (%1.3f, %1.3f)}$ and $\mathbf{a \in (%1.3f, %1.3f)}$') % \
                           (DEFAULT_XYH, DEFAULT_AH, tGridRange[0], tGridRange[1], aGridRange[0], aGridRange[1])
            fig.suptitle(summaryTitle, fontsize=18, fontweight='bold')
            plt.savefig(stampOutputFilePathFunc(atFigureCtr))
            atFigureCtr += 1

