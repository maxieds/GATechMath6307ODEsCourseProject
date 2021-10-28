#### RosslerGenLyapunovExponentExperiments.py
#### Maxie Dion Schmidt
#### 2021.10.23

import numpy as np
import matplotlib.pyplot as plt
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
PP = PolynomialRing(RR, 4, "txyz")
T, X, Y, Z = PP.gens()
A, B, C = var('za zb zc')
AVAR, BVAR, CVAR = A, B, C
XVAR, YVAR, ZVAR = var('x y z')
XP = -(Y + Z)
YP = X + A * Y 
ZP = B + Z * (X - C)

DEFAULT_IC=(0, (1, 1, 1))
extractICs = lambda ics, ndim: ics[0] if ndim == 0 else ics[1][ndim - 1] 

DEFAULT_ABCPARAMS='classic'
DEFAULT_H=0.25
DEFAULT_AXIS_PLOT_INTERVAL=(-6.0, 6.0)

PLANE_PROJ_TYPE='XY'
#PLANE_PROJ_TYPE='XZ'
#PLANE_PROJ_TYPE='YZ'
PLANE_PROJ_TYPE_SYSTEMS = {
    'XY' : [ XP.subs({ Z : 0 }), YP.subs({ Z : 0 }) ], 
    'XZ' : [ XP.subs({ Y : 0 }), ZP.subs({ Y : 0 }) ],
    'YZ' : [ YP.subs({ X : 0 }), ZP.subs({ X : 0 }) ],
}
PLANE_PROJ_TYPE_SYSTEM_VARS = {
    'XY' : [ (X, XVAR), (Y, YVAR) ], 
    'XZ' : [ (X, XVAR), (Z, ZVAR) ],
    'YZ' : [ (Y, YVAR), (Z, ZVAR) ],
}

PLANE_PROJ_TYPE_ICS = {
    'XY' : (extractICs(DEFAULT_IC, 1), extractICs(DEFAULT_IC, 2)), 
    'XZ' : (extractICs(DEFAULT_IC, 1), extractICs(DEFAULT_IC, 3)),
    'YZ' : (extractICs(DEFAULT_IC, 2), extractICs(DEFAULT_IC, 3)),
}
PLANE_PROJ_TYPE_OMITXYZ = {
    'XY' : Z, 
    'XZ' : Y, 
    'YZ' : X,

}

#DEFAULT_V0_VECS=[ (1, -3) ]
#DEFAULT_V0_VECS=[ (-4, 7) ]
#DEFAULT_V0_VECS=[ (1/sqrt(2), 0.5) ]
#DEFAULT_V0_VECS=[ (sqrt(2)*6, -sqrt(2)*2.5) ]
#DEFAULT_V0_VECS=[ (24.0, -32.0) ]
DEFAULT_V0_VECS=[ (-1, 1) ]
#DEFAULT_V0_VECS=[ (A**2-B**2, -(A**2-B**2)) ]

#DEFAULT_YTRANS_FUNC=lambda ap, bp, cp: 0 if cp == 0 else Utils.SageMathNorm(n(cp / ap + ap * ap / cp))
#DEFAULT_YTRANS_FUNC_LABEL=r'$\frac{c}{a} + \frac{a^2}{c}$'
#DEFAULT_YTRANS_FUNC=lambda ap, bp, cp: 0 if cp == 0 else Utils.SageMathNorm(n(ap * (1-cp) / sqrt(3) + cp / pi))
#DEFAULT_YTRANS_FUNC_LABEL=r'$a(1-c) / \sqrt{3} + c / \pi$'
#DEFAULT_YTRANS_FUNC=lambda ap, bp, cp: ap + bp + cp
#DEFAULT_YTRANS_FUNC_LABEL=r'$a+b+c$'
#DEFAULT_YTRANS_FUNC=lambda ap, bp, cp: ap * bp * cp
#DEFAULT_YTRANS_FUNC_LABEL=r'$abc$'
#DEFAULT_YTRANS_FUNC=lambda ap, bp, cp: arctanh(cp / ap / bp) * ap * bp * cp / (1 + (ap * bp * cp)**2)
#DEFAULT_YTRANS_FUNC_LABEL=r'$\frac{abc}{1+(abc)^2} \times \tanh^{-1}\left(\frac{c}{ab}\right)$'
#DEFAULT_YTRANS_FUNC=lambda ap, bp, cp: (ap + bp) * (ap * bp / (1 + (ap * bp)**2) + ap**2 - bp**3)
#DEFAULT_YTRANS_FUNC_LABEL=r'$(a + b)\left(\frac{ab}{1+(abc)^2} + a^2 - b^3\right)$'
#DEFAULT_YTRANS_FUNC=lambda ap, bp, cp: cp * (sin(ap)**2) + (cp**2 + 1) * cosh(bp**2)
#DEFAULT_YTRANS_FUNC_LABEL=r'$c \sin^2(a) + (1 + c^2) \cosh(b^2)$'
#DEFAULT_YTRANS_FUNC=lambda ap, bp, cp: (cp * ap + bp)**(cp * ap - bp)
#DEFAULT_YTRANS_FUNC_LABEL=r'$(ac+b)^(ac-b)$'
DEFAULT_YTRANS_FUNC=lambda ap, bp, cp: 13 * cos(ap + bp + cp) - 5 * cos(2 * (ap + bp + cp)) - \
                                       2 * cos(3 * (ap + bp + cp)) - cos(4 * (ap + bp + cp))
DEFAULT_YTRANS_FUNC_LABEL=r'$13 \cos(T) - 5\cos(2T) - 2\cos(3T) - \cos(4T) + 20$ for $T \equiv a+b+c$'
#XT_PARAM=lambda ap, bp, cp: 16 * (sin(ap + bp + cp)**3)
#YT_PARAM=lambda ap, bp, cp: 13 * cos(ap + bp + cp) - 5 * cos(2 * (ap + bp + cp)) - \
#                            2 * cos(3 * (ap + bp + cp)) - cos(4 * (ap + bp + cp))
#DEFAULT_YTRANS_FUNC=lambda ap, bp, cp: math.fmod(Utils.SageMathNorm(sqrt(XT_PARAM(ap, bp, cp)**2 + YT_PARAM(ap, bp, cp)**2) * \
#                                       max((XT_PARAM(ap, bp, cp)**(1/3.0)), YT_PARAM(ap, bp, cp) / sqrt(3)) / \
#                                       (1 + min((XT_PARAM(ap, bp, cp)**(1/5.0)), YT_PARAM(ap, bp, cp) / sqrt(5)))), (1 + sqrt(5)) / 2.0)
#DEFAULT_YTRANS_FUNC_LABEL=r'$\log\left(\frac{\max\left(X(T)^{\frac{1}{3}}, \frac{Y(T)}{\sqrt{3}})}{1 + \min\left(X(T)^{\frac{1}{5}}, \frac{Y(T)}{\sqrt{5}})} \times \sqrt(X(T)^2+Y(T)^2) \pmod{\frac{1 + \sqrt{5}}{2}}\right)$ for $T \equiv a+b+c$'

#DEFAULT_ABCPARAMS_RELATION=[ A**2+B**2+C**2 == 4 ]
#DEFAULT_ABCPARAMS_RELATION=[ (A-0.2)**4-(B-0.2)**3+2*(C-5.7)**2 == 2 ]
#DEFAULT_ABCPARAMS_RELATION=[ A**4-B**3+2*(C**2) == 2 ]
#DEFAULT_ABCPARAMS_RELATION=[ (A+B+C)**2 == 3.0 ]
#DEFAULT_ABCPARAMS_RELATION=[ (A+B+C)**2 - (A + B + C) - 1 == 0 ]
#DEFAULT_ABCPARAMS_RELATION=[ A**5 + B**5 == 2 * (C**5) ]
DEFAULT_ABCPARAMS_RELATION=[ (A**2 + B**2 + C * A)**2 == (C**2) * (A**2 + B**2) ]

## NOTE: The solver cannot handle this in SageMath, though parametric solutions exist, so an extension of the 
##       existing code is needed to plot (a, b, c) along this "heart" shaped surface: 
#DEFAULT_ABCPARAMS_RELATION=[ (2 * (A**2) + 2 * (B**2) + C**2 - 1)**3 - 1 / 10.0 * (A**2) * (C**3) - (B**2) * (C**3) == 0 ]
#DEFAULT_ABCPARAMS_RELATION=[ 
#    ((2 * (A**2) + 2 * (B**2) + C**2 - 1)**3 - 1 / 10.0 * (A**2) * (C**3) - (B**2) * (C**3)).subs({ C : 1.5 }) == 0
#]

PERFORM_3DROTATION_YZTPOINTS=False
SOLUTION_3DROTATION_MATRIX = rotate_arbitrary((-0.125, 2, 1.1), n(2 * pi / 3))
USE_LOGARITHMIC_SCALE=False

IMAGE_OUTFILE_EXTRA_DESC_BASE="-Variant%s" % Utils.GetShortObjectHash( [ DEFAULT_V0_VECS, DEFAULT_YTRANS_FUNC_LABEL, DEFAULT_ABCPARAMS_RELATION ], 6)
IMAGE_OUTFILE_EXTRA_DESC="%s-%s" % (IMAGE_OUTFILE_EXTRA_DESC_BASE, 'logscale' if USE_LOGARITHMIC_SCALE else 'linearscale')
PLOT_SUMMARY_TEMP_OUTFILE='../../Images/TempOutputData/RosslerAttractorExpt1%s-Type%s-%s.png'
stampOutputFilePathFunc = lambda filePathTempl: str(filePathTempl) % (IMAGE_OUTFILE_EXTRA_DESC, PLANE_PROJ_TYPE, Utils.GetTimestamp())

RUNTIME_OPTIONS_KWARGS_DICT = {
     'h'                 : DEFAULT_H, 
     'ics'               : PLANE_PROJ_TYPE_ICS[PLANE_PROJ_TYPE], 
     't0'                : extractICs(DEFAULT_IC, 0), 
     'pltAB'             : DEFAULT_AXIS_PLOT_INTERVAL, 
     'plotOutPath'       : PLOT_SUMMARY_TEMP_OUTFILE, 
     'omitXYZ'           : PLANE_PROJ_TYPE_OMITXYZ[PLANE_PROJ_TYPE], 
     'omitXYZSystem'     : PLANE_PROJ_TYPE_SYSTEMS[PLANE_PROJ_TYPE], 
     'omitXYZSystemVars' : PLANE_PROJ_TYPE_SYSTEM_VARS[PLANE_PROJ_TYPE], 
     'omitICs'           : PLANE_PROJ_TYPE_ICS[PLANE_PROJ_TYPE],
     'v0Vectors'         : DEFAULT_V0_VECS, 
     'yTransFunc'        : DEFAULT_YTRANS_FUNC, 
     'yTransFuncLabel'   : DEFAULT_YTRANS_FUNC_LABEL, 
     'abcParamsRelation' : DEFAULT_ABCPARAMS_RELATION,
}


def GetXYZSolutionsGrid(icPoint, solInterval, h, fgFuncs=None):
    (t0, (x0, y0)) = icPoint
    odeSol = np.array(eulers_method_2x2(fgFuncs[0], fgFuncs[1], t0, x0, y0, h, 1, algorithm="none"))
    return (odeSol[:, 0], list(zip(odeSol[:, 1], odeSol[:, 2])))

def GetSolutionRHSVariables(eqnSol, initVarsList):
    rhsSol = [ solComp.rhs() for (cidx, solComp) in enumerate(eqnSol) ]
    solActiveVars = []
    for solComp in rhsSol:
        for freeVar in solComp.variables():
            if freeVar not in solActiveVars:
                solActiveVars += [ freeVar ]
    solActiveVars.sort(reverse=False, key=lambda vname: str(vname))
    if len(solActiveVars) > len(initVarsList):
        raise RuntimeError("Invalid variables: %s" % solActiveVars)
    newVarNameDictMap = dict([ (newVarName, initVarsList[vidx]) for (vidx, newVarName) in enumerate(solActiveVars) ])
    rhsSol = list(map(lambda lstElem: lstElem.subs(newVarNameDictMap), rhsSol))
    solActiveVars = initVarsList[:len(solActiveVars)]
    return ( tuple(rhsSol), solActiveVars )

def GetConstrainedABCGrid(zeroEqns, abcVars, meshH, varApproxInt=(-5.0, 5.0), enforceVarLen=2):
    eq               = zeroEqns
    (vaIntA, vaIntB) = varApproxInt
    h                = meshH
    eqnSols = solve(eq, abcVars)
    activeVars = []
    for (eidx, eqSol) in enumerate(eqnSols): 
        (rhsSol, localVars) = GetSolutionRHSVariables(eqSol, abcVars)
        eqnSols[eidx] = rhsSol
        for nextFreeVar in localVars:
            if nextFreeVar not in activeVars:
                activeVars += [ nextFreeVar ]
    if enforceVarLen > 0 and len(activeVars) > enforceVarLen:
        raise RuntimeError("Too many variables (> %d): %s" % (enforceVarLen, activeVars))
    num1DGridPoints = math.floor((vaIntB - vaIntA) / float(h)) + 1
    abcVarGrid = np.linspace(vaIntA, vaIntB, num1DGridPoints)
    varProductGrid = itertools.product(abcVarGrid, repeat=len(activeVars))
    abcVarGridPoints = []
    for gridPoint in varProductGrid:
        for eqSol in eqnSols:
            abcGridPoint = []
            allRealComps = True
            for tupleElem in list(eqSol):
                varToGridPointDictMap = dict([ (var, gridPoint[vidx]) for (vidx, var) in enumerate(activeVars) ])
                # Only take the abc parameters that are real-valued: 
                try:
                     tupleElem = tupleElem.subs(varToGridPointDictMap)
                except ValueError:
                    allRealComps = False
                    continue
                abcGridPoint += [ tupleElem ]
            if allRealComps:
                abcVarGridPoints += [ tuple(abcGridPoint) ]
    return abcVarGridPoints

def ComputeLambdaABC(xyzGrid, v0):
    (yGrid, zGrid) = tuple(zip(*xyzGrid))
    nUpper = len(yGrid)
    v0 = vector(list(v0))
    rDtFunc = vector([ A * YP, B - ZP * C ])
    lambdaSum = 0
    for n in range(0, nUpper):
        yp, zp = simplify(yGrid[n]), simplify(zGrid[n])
        logArgFunc = simplify(rDtFunc.dot_product(v0))
        logArgFunc = simplify(logArgFunc.substitute(dict([ (XVAR, 0), (YVAR, yp), (ZVAR, zp) ])))
        lambdaSum += log(logArgFunc) / float(nUpper)
    lambdaSum = simplify(lambdaSum)
    return lambda ap, bp, cp: lambdaSum.substitute(dict([ (AVAR, ap), (BVAR, bp), (CVAR, cp) ]))

def ComputeABCDensityPlot(abcGrid, xyTransFuncs, showPlot=True, axPlt=None, plotTitle=None, xyLabels=(None, None), bins=None):
     (xAxisTransFunc, yAxisTransFunc) = xyTransFuncs
     xPoints, yPoints = [], []
     for abcPoint in abcGrid:
          abcPoint = tuple(map(lambda cpoint: Utils.SageMathNorm(cpoint), abcPoint))
          (ap, bp, cp) = abcPoint
          xPoints += [ xAxisTransFunc(ap, bp, cp) ] 
          yPoints += [ yAxisTransFunc(ap, bp, cp) ]
     xmin, xmax = min(xPoints), max(xPoints)
     ymin, ymax = min(yPoints), max(yPoints)
     if axPlt == None:
         pltFig = plt.figure(1)
     else:
         pltFig = axPlt
     plt.hexbin(xPoints, yPoints, gridsize=24, cmap='magma', bins=bins)
     plt.xlabel('' if xyLabels[0] == None else xyLabels[0])
     plt.ylabel('' if xyLabels[1] == None else xyLabels[1])
     plt.title('' if plotTitle == None else plotTitle)
     if axPlt != None:
         plt.colorbar()
     if showPlot:
         plt.draw()

if __name__ == "__main__":     
    v0Vecs = RUNTIME_OPTIONS_KWARGS_DICT['v0Vectors']
    meshH = RUNTIME_OPTIONS_KWARGS_DICT['h']
    xyzICPoint =  RUNTIME_OPTIONS_KWARGS_DICT['ics']
    t0 = RUNTIME_OPTIONS_KWARGS_DICT['t0']
    xyzSolInt = (xyzIntA, xyzIntB) = DEFAULT_AXIS_PLOT_INTERVAL
    fgFuncsInit = RUNTIME_OPTIONS_KWARGS_DICT['omitXYZSystem']
    modSystemVars = RUNTIME_OPTIONS_KWARGS_DICT['omitXYZSystemVars']
    getFGFunc = lambda fgIdx: lambda tv, xv, yv: \
            fgFuncsInit[fgIdx].subs({ modSystemVars[0][0] : xv, modSystemVars[0][1] : xv, modSystemVars[1][0] : yv, modSystemVars[1][1] : yv })
    fgFuncs = (getFGFunc(0), getFGFunc(1))
    (tGridPoints, xyzGridSolPoints) = GetXYZSolutionsGrid((t0, xyzICPoint), xyzSolInt, meshH, fgFuncs)
    if PERFORM_3DROTATION_YZTPOINTS:
        (yCompPoints, zCompPoints) = zip(*xyzGridSolPoints)
        tyzPoints = list(zip(tGridPoints, yCompPoints, zCompPoints))
        tyzPoints = list(map(lambda lstElem: vector(lstElem), tyzPoints))
        rotatedTYZPoints = list(map(lambda tyzVec: tuple(list(SOLUTION_3DROTATION_MATRIX * tyzVec)), tyzPoints))
        (rtComp, ryComp, rzComp) = zip(*rotatedTYZPoints)
        xyzGridSolPoints = list(zip(ryComp, rzComp)) 
    

    xTransFunc = lambda ap, bp, cp: Utils.SageMathNorm(n(ComputeLambdaABC(xyzGridSolPoints, v0)(ap, bp, cp)))
    yTransFunc = RUNTIME_OPTIONS_KWARGS_DICT['yTransFunc']
    xyTransFuncs = (xTransFunc, yTransFunc)
    abcGrid = GetConstrainedABCGrid( RUNTIME_OPTIONS_KWARGS_DICT['abcParamsRelation'], [ A, B, C ], meshH, varApproxInt=(xyzIntA, xyzIntB))
    figCols = 1
    pltFig, pltAx = plt.subplots(nrows=(len(v0Vecs)%figCols)+1,ncols=figCols, sharex=False, sharey=False, figsize=(8, 4), squeeze=False)
    pltFig.subplots_adjust(hspace=0.5, left=0.08, right=0.92)
    useLogScale = 'log' if USE_LOGARITHMIC_SCALE else None
    for (vidx, v0) in enumerate(v0Vecs):
        ax = pltAx.flatten()[vidx]
        ax.set_aspect('auto', adjustable='datalim')
        try:
            plotTitle = r'Density plot corresponding to $v_0=%s$' % \
                        sympy.latex(str(v0).replace("\\text{", "{")).replace("\\text{", "{")
        except ValueError:
            try:
                plotTitle = r'Density plot corresponding to v_0=%s' % str(v0)
            except ValueError:
                plotTitle = r'Density plot (for unprintable symbolic $v_0$)'
        plotTitle = plotTitle.replace('za', 'a').replace('zb', 'b').replace('zc', 'c')
        xLabel = RUNTIME_OPTIONS_KWARGS_DICT['yTransFuncLabel']
        yLabel = r'$|\lambda(a, b, c)|$'
        ComputeABCDensityPlot(abcGrid, xyTransFuncs, showPlot=False, axPlt=ax, 
                              plotTitle=plotTitle, xyLabels=(xLabel, yLabel), bins=useLogScale)
    plt.savefig(stampOutputFilePathFunc(RUNTIME_OPTIONS_KWARGS_DICT['plotOutPath']))
    #plt.show(block=True)

