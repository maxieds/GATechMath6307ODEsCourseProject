#### RosslerGenLyapunovExponentExperiments.py
#### Maxie Dion Schmidt
#### 2021.10.23

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.tri as tri
import scipy
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
import scipy.stats as spstats
import sympy
import sys
import math
import cmath
from sage.all import *
from PythonODEBaseLibrary import *
from Utils import * 
from sage.repl.display.pretty_print import SagePrettyPrinter
from io import StringIO
import time

sppStream=StringIO()
spp=SagePrettyPrinter(sppStream, 78, '\n')
shortPrint = lambda inputText, maxLen=256: \
             inputText if len(inputText) <= 2 * maxLen else inputText[:maxLen] + '\n< ... >\n' + inputText[-maxLen:]

PLANE_PROJ_TYPE='XY'
#PLANE_PROJ_TYPE='XZ'
#PLANE_PROJ_TYPE='YZ'

ABC_PARAMS = (0.2, 0.2, 5.7)
XVAR, YVAR, ZVAR, T = var('x y z t')
DEFAULT_IC=(0, (1, 1, 1))
DEFAULT_XYH=0.05 #0.25 #0.050
DEFAULT_ODE_TRANGE=(0.0, 1.0)

RHS_RV = spstats.norm() #spstats.gamma(3.0)
RHS_RV_MOMENT_FUNC = lambda M, a, b, pdfNormalizer, rv=RHS_RV: \
                     rv.expect(lambda x: 0 if scipy.exp(x)-1 == 0.0 else np.lib.scimath.power(scipy.exp(x)-1, -float(M)), lb=a, ub=b) / pdfNormalizer
RHS_RV_PDF = lambda t, rv=RHS_RV: (np.exp(t) - 1) * rv.pdf(t)
RHS_RV_PLOT_TITLE = r'$\operatorname{HalfNorm}(t)$'
RHS_RV_PDF_TRANGE = (-1.0, 6.0)
FIGURE_SUMMARY_TITLE = ''.join([
    r'Distribution of $U_1 = U_2$ such that\n' 
    r'$\log\left(1+\frac{%s[U_1]}{%s[U_2]}\right) ' % (PLANE_PROJ_TYPE[0], PLANE_PROJ_TYPE[1]), 
    r'\overset{\mathcal{D}}{=} \operatorname{Normal}(0, 1)$\n', 
    r'with parameters (a, b, c) := (%1.3f, %1.3f, %1.3f)\n' % (ABC_PARAMS[0], ABC_PARAMS[1], ABC_PARAMS[2]),
    r'[TOTAL RUNNING TIME: %%d sec / %%1.2f min / %%1.2f hours]'
])

PLANE_PROJ_GET_XFUNCS = {
    'XY' : lambda xp, yp, zp, idx: xp[idx],
    'XZ' : lambda xp, yp, zp, idx: xp[idx],
    'YZ' : lambda xp, yp, zp, idx: yp[idx],
}
PLANE_PROJ_GET_YFUNCS = {
    'XY' : lambda xp, yp, zp, idx: yp[idx],
    'XZ' : lambda xp, yp, zp, idx: zp[idx],
    'YZ' : lambda xp, yp, zp, idx: zp[idx],
}

PLOT_SUMMARY_TEMP_OUTFILE='../../Images/TempOutputData/RosslerAttractorExpt3-%s.png'
stampOutputFilePathFunc = lambda: PLOT_SUMMARY_TEMP_OUTFILE % (Utils.GetTimestamp())

def GetXYSolutionsGridForFixedParams(abcParams, icPoint, tRange, h):
    (t0, (x0, y0, z0)) = icPoint
    (a, b, c) = abcParams
    fxyz = [ -YVAR-ZVAR, XVAR + a * YVAR, b + ZVAR * (XVAR - c) ] 
    numT = math.floor((tRange[1] - tRange[0]) / h) + 1
    tSpec = list(np.linspace(tRange[0], tRange[1], numT))
    odeVars = [ XVAR, YVAR, ZVAR ]
    initConds = [ x0, y0, z0 ]
    try:
        xyzPoints = desolve_odeint(fxyz, initConds, tSpec, odeVars)
    except Exception as excpt:
        raise excpt
    xSolPoints, ySolPoints, zSolPoints = xyzPoints[:, 0], xyzPoints[:, 1], xyzPoints[:, 2]
    return (tSpec, list(zip(xSolPoints, ySolPoints, zSolPoints)))

def PlotRHSPdfGraph(pdfFunc, tRange, deltaT, ax, plotTitle=''):
    (tA, tB) = tRange
    numT = math.floor((tB - tA) / deltaT) + 1
    tSpec = np.linspace(tA, tB, numT)
    pdfPoints = [ pdfFunc(tv) for tv in tSpec ]
    plt.sca(ax)
    plt.plot(tSpec, pdfPoints, 'g--')
    plt.hist(pdfPoints, bins=100, alpha=0.5, density=True)
    plt.xlabel('t')
    plt.title(plotTitle)
    return ax

def DetermineApproxODESolRVDists(abcParams, odeIC, odeTSpec, h, axs, runSimplify=True, verbose=True):
    [ ax1, ax2 ] = axs
    (odeTPoints, odeXYZSols) = GetXYSolutionsGridForFixedParams(abcParams, odeIC, odeTSpec, h)
    xPoints, yPoints, zPoints = tuple(zip(*odeXYZSols))
    (tA, tB) = odeTSpec
    pdfNormalizer = 1.0 #RHS_RV.expect(lambda x: 1, lb=tA, ub=tB)
    numN = len(odeTPoints)
    getXiFunc = lambda i: PLANE_PROJ_GET_XFUNCS[PLANE_PROJ_TYPE](xPoints, yPoints, zPoints, i)
    getYjFunc = lambda j: PLANE_PROJ_GET_YFUNCS[PLANE_PROJ_TYPE](xPoints, yPoints, zPoints, j)
    p1Vars = list(var(' '.join([ 'pOne%d' % p1VarIdx for p1VarIdx in range(0, numN) ])))
    p2Vars = p1Vars
    p12Vars = p1Vars
    onesVector = np.array([ 1 ] * (2 * numN))
    onesZerosVector = np.array([ 1 ] * numN + [ 0 ] * numN)
    zerosOnesVector = np.array([ 0 ] * numN + [ 1 ] * numN)
    xyDiagMatrixLstTop, xyDiagMatrixLstBottom = [], []
    p1DiagMarixLstTop, p2DiagMarixLstBottom = [], []
    for rowIdx in range(0, numN):
        zeroPadding = [ 0 ] * numN
        upperYRow = [ [ 1 / getYjFunc(j) if j == rowIdx else 0 for j in range(0, numN) ] + zeroPadding ]
        lowerXRow = [ zeroPadding + [ getXiFunc(i) if i == rowIdx else 0 for i in range(0, numN) ] ]
        xyDiagMatrixLstTop += upperYRow
        xyDiagMatrixLstBottom += lowerXRow
        p1DiagMarixLstTop += [ [ p1Vars[i] if i == rowIdx else 0 for i in range(0, numN) ] + zeroPadding ]
        p2DiagMarixLstBottom += [ zeroPadding + [ p2Vars[j] if j == rowIdx else 0 for j in range(0, numN) ] ]
    xyDiagMatrix = np.array(xyDiagMatrixLstTop + xyDiagMatrixLstBottom)
    p12VarsDiagMatrix = np.array(p1DiagMarixLstTop + p2DiagMarixLstBottom)
    momentEqns = []
    lastMatrixPow = xyDiagMatrix
    for M in range(1, numN + 1):
        rhsMoment = RHS_RV_MOMENT_FUNC(M, tA, tB, pdfNormalizer)
        mthMatrix = xyDiagMatrix * lastMatrixPow 
        matrixPartialProd = np.dot(np.matmul(mthMatrix, p12VarsDiagMatrix), onesVector)
        lhsVarsEqn = np.dot(onesZerosVector.T, matrixPartialProd) * np.dot(zerosOnesVector.T, matrixPartialProd)
        if runSimplify:
            lhsVarsEqn = expand(simplify(lhsVarsEqn))
        momentEqns += [ lhsVarsEqn - rhsMoment ]
        lastMatrixPow = mthMatrix
    if verbose:
        spp.pretty(momentEqns)
        print("MOMENT-EQNS:\n", shortPrint(sppStream.getvalue()), "\n")
    p12VarsSols = sympy.solve(momentEqns, p12Vars)
    if verbose:
        spp.pretty(p12VarsSols)
        print("FULL-INDET-SOL-SET:\n", sppStream.getvalue(), "\n")
    indetParamsSubsts = {}
    p1SolVarsSet = None
    for solSubsetIdx in range(0, len(p12VarsSols)):
        if all((solEq in RR) and (solEq >= 0.0) for solEq in p12VarsSols[solSubsetIdx]):
            p1SolVarsSet = list(p12VarsSols[solSubsetIdx])
            break
    if p1SolVarsSet == None:
        raise RuntimeError("No solutions found for PDF values!!!")
    p1SolVars = [ eqnSol for eqnSol in p1SolVarsSet[:numN] ]
    p1VarValues = [ p1v.subs(indetParamsSubsts) for p1v in p1SolVars ] 
    if verbose:
        spp.pretty(p1VarValues)
        print("P1-PDF-VALUES:\n", shortPrint(sppStream.getvalue(), 4096), "\n")
    plt.sca(ax1)
    rvu1HistData = p1VarValues 
    rvu1HistBins = odeTPoints
    plt.hist(rvu1HistData, bins=rvu1HistBins, color='#0733bb')
    plt.title(r'$U_1$ random variable distribution')
    plt.sca(ax2)
    rvu2HistData = [ SPARE_VAR_DIST_PDF(t) for t in odeTPoints ]
    rvu2HistBins = odeTPoints
    plt.plot(rvu2HistData, rvu2HistBins, color='#0504aa')
    plt.title(r'$U_2$ random variable distribution')
    return [ ax1, ax2 ]

if __name__ == "__main__":
    startTime = time.time()
    tRange = DEFAULT_ODE_TRANGE
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    figRows, figCols = 1, 3
    gs = GridSpec(figRows, figCols, figure=fig)
    axs = [ fig.add_subplot(gs[0, colIdx]) for colIdx in range(0, figCols) ]
    axs[0] = PlotRHSPdfGraph(RHS_RV_PDF, RHS_RV_PDF_TRANGE, DEFAULT_XYH, axs[0], RHS_RV_PLOT_TITLE)
    axs[1:] = DetermineApproxODESolRVDists(ABC_PARAMS, DEFAULT_IC, DEFAULT_ODE_TRANGE, DEFAULT_XYH, axs[1:])
    for ax in axs:
        ax.set_aspect('auto', adjustable='datalim')
    fig.tight_layout(pad=1)
    fig.subplots_adjust(top=0.8, bottom=0.2, right=0.9, left=0.1, hspace=0.5, wspace=0.5)
    endTime = time.time()
    diffTime = endTime - startTime
    summaryTitle = FIGURE_SUMMARY_TITLE % (diffTime, diffTime / 60.0, diffTime / 3600.0)
    fig.suptitle(summaryTitle, fontsize=15, fontweight='bold')
    plt.savefig(stampOutputFilePathFunc())

