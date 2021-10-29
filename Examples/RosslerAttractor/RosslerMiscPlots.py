#### RosslerMiscPlots.py
#### Maxie Dion Schmidt
#### 2021.10.24

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
from MatplotlibBase import *
from scipy.integrate import odeint
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.abc import r

RR = RealField(sci_not=0, prec=4, rnd='RNDU')
PP = PolynomialRing(RR, 4, "txyz")
T, X, Y, Z = PP.gens()
A, B, C = var('za zb zc')
AVAR, BVAR, CVAR = A, B, C
XVAR, YVAR, ZVAR = var('x y z')
ZERO_XYZ_COMPONENT = X
XP = -1.0 * Y - 1.0 * Z
YP = X + A * Y 
ZP = B + Z * (X - C)
[EFUNC, FFUNC, GFUNC] = [ XP, YP, ZP ]

DEFAULT_IC=(0, (1, 1, 1))
extractICs = lambda ics, ndim: ics[0] if ndim == 0 else ics[1][ndim - 1] 

COMMON_ABCPARAMS_LOOKUP_BYNAME = dict([
    ('classic', (0.2, 0.2, 5.7)), 
    ('common1', (0.2, 0.2, 14.0)), 
    ('common2', (0.1, 0.1, 14.0)), 
    ('a24v1',   (-1.0, 2.0, 4.0)), 
    ('a24v2',   (0.1, 2.0, 4.0)), 
    ('a24v3',   (0.2, 2.0, 4.0)), 
    ('a24v4',   (0.3, 2.0, 4.0)), 
    ('a24v5',   (0.35, 2.0, 4.0)), 
    ('a24v6',   (0.38, 2.0, 4.0)), 
    ('stcv1',   (0.1, 0.1, 4.0)), 
    ('stcv2',   (0.1, 0.1, 6.0)), 
    ('stcv3',   (0.1, 0.1, 8.5)), 
    ('stcv4',   (0.1, 0.1, 8.7)), 
    ('stcv5',   (0.1, 0.1, 9.0)), 
    ('stcv6',   (0.1, 0.1, 12.0)), 
    ('stcv7',   (0.1, 0.1, 12.6)), 
    ('stcv8',   (0.1, 0.1, 13.0)), 
    ('stcv9',   (0.1, 0.1, 18.0)),
])

DEFAULT_ABCPARAMS='classic'
DEFAULT_H=0.01
DEFAULT_AXIS_PLOT_INTERVAL=(-5.0, 5.0)

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
PLANE_PROJ_TYPE_1DPOINTMAP_COMP1_FUNC = {
    'XY' : lambda pA, pB, pC, xx, yy: -1.0 * yy, 
    'XZ' : lambda pA, pB, pC, xx, zz: -1.0 * zz, 
    'YZ' : lambda pA, pB, pC, yy, zz: pA * yy,
}

PLANE_PROJ_TYPE_1DPOINTMAP_COMP2_FUNC = {
    'XY' : lambda pA, pB, pC, xx, yy: xx + pA * yy, 
    'XZ' : lambda pA, pB, pC, xx, zz: pB + zz * (xx - pC), 
    'YZ' : lambda pA, pB, pC, yy, zz: pB - pC * zz,
}

PLOT1D_SUMMARY_TEMP_OUTFILE='../../Images/TempOutputData/RosslerAttractorSummary-Type%s-Plot1D-A%1.3fB%1.3fC%1.3f-%s.png'
PLOT2D_SUMMARY_TEMP_OUTFILE='../../Images/TempOutputData/RosslerAttractorSummary-Type%s-Plot2D-A%1.3fB%1.3fC%1.3f-%s.png'
PLOT3D_SUMMARY_TEMP_OUTFILE='../../Images/TempOutputData/RosslerAttractorSummary-Type%s-Plot3D-A%1.3fB%1.3fC%1.3f-%s.png'
stampOutputFilePathFunc = lambda filePathTempl, abc: str(filePathTempl) % (PLANE_PROJ_TYPE, abc[0], abc[1], abc[2], Utils.GetTimestamp())

DEFAULT_RUNTIME_OPTIONS_KWARGS_DICT = {
     'ABC'               : COMMON_ABCPARAMS_LOOKUP_BYNAME[DEFAULT_ABCPARAMS], 
     'h'                 : DEFAULT_H, 
     'ics'               : DEFAULT_IC, 
     't0'                : extractICs(DEFAULT_IC, 0), 
     'pltAB'             : DEFAULT_AXIS_PLOT_INTERVAL, 
     'doPlot1D'          : True,
     'plot1DOutPath'     : PLOT1D_SUMMARY_TEMP_OUTFILE, 
     'doPlot2D'          : True, 
     'plot2DOutPath'     : PLOT2D_SUMMARY_TEMP_OUTFILE,
     'doPlot3D'          : True, 
     'plot3DOutPath'     : PLOT3D_SUMMARY_TEMP_OUTFILE,
     'omitXYZ'           : PLANE_PROJ_TYPE_OMITXYZ[PLANE_PROJ_TYPE], 
     'omitXYZSystem'     : PLANE_PROJ_TYPE_SYSTEMS[PLANE_PROJ_TYPE], 
     'omitXYZSystemVars' : PLANE_PROJ_TYPE_SYSTEM_VARS[PLANE_PROJ_TYPE], 
     'omitICs'           : PLANE_PROJ_TYPE_ICS[PLANE_PROJ_TYPE],
     'vecMapComp1Func'   : PLANE_PROJ_TYPE_1DPOINTMAP_COMP1_FUNC[PLANE_PROJ_TYPE],
     'vecMapComp2Func'   : PLANE_PROJ_TYPE_1DPOINTMAP_COMP2_FUNC[PLANE_PROJ_TYPE],
}

def RosserAttractorSolutionSummary(runOptionsKWDict=DEFAULT_RUNTIME_OPTIONS_KWARGS_DICT, 
                                   extraPlotOptionsKWDict=DEFAULT_PLOT_OPTIONS_DICT):     
    if runOptionsKWDict == None or extraPlotOptionsKWDict == None:
        raise ValueError
    
    (pA, pB, pC)  = runOptionsKWDict['ABC']
    h             = runOptionsKWDict['h']
    t0            = runOptionsKWDict['t0']
    (intA, intB)  = runOptionsKWDict['pltAB'] 
    modXYZSystem  = runOptionsKWDict['omitXYZSystem']
    ics           = runOptionsKWDict['omitICs']
    modXYZSystem  = [ 
        modXYZSystem[0].subs({ AVAR : pA, BVAR : pB, CVAR : pC }), 
        modXYZSystem[1].subs({ AVAR : pA, BVAR : pB, CVAR : pC }) 
    ]
    modSystemVars = runOptionsKWDict['omitXYZSystemVars']
    numGridPoints = math.floor((intB - intA) / h) + 1
    timeSpecT     = np.linspace(intA, intB, numGridPoints)
    FxyzFunc       = lambda s, t: (modXYZSystem[0].subs({ \
                                    modSystemVars[0][0] : s[0], modSystemVars[0][1] : s[0], modSystemVars[1][0] : s[1], modSystemVars[1][1] : s[1] }), \
                                   modXYZSystem[1].subs({ \
                                    modSystemVars[0][0] : s[0], modSystemVars[0][1] : s[0], modSystemVars[1][0] : s[1], modSystemVars[1][1] : s[1] }))

    odeSol        = odeint(FxyzFunc, list(ics), timeSpecT)
    yDataPoints, zDataPoints = odeSol[:, 0], odeSol[:, 1]
    yzDataPoints  = list(zip(list(yDataPoints), list(zDataPoints)))

    plot1DVectorMapComp1Func = lambda v1, v2: DEFAULT_RUNTIME_OPTIONS_KWARGS_DICT['vecMapComp1Func'](pA, pB, pC, v1, v2)
    plot1DVectorMapComp2Func = lambda v1, v2: DEFAULT_RUNTIME_OPTIONS_KWARGS_DICT['vecMapComp2Func'](pA, pB, pC, v1, v2)
    plot1DPointMapFunc = lambda v1, v2: plot1DVectorMapComp1Func(v1, v2) * plot1DVectorMapComp1Func(v1, v2) + \
                                        plot1DVectorMapComp2Func(v1, v2) * plot1DVectorMapComp2Func(v1, v2)
    plot1DMapFuncs = (plot1DPointMapFunc, (plot1DVectorMapComp1Func, plot1DVectorMapComp2Func))

    doPlot1, doPlot2, doPlot3 = runOptionsKWDict['doPlot1D'], runOptionsKWDict['doPlot2D'], \
                                runOptionsKWDict['doPlot3D']
    p1OutPath, p2OutPath, p3OutPath = runOptionsKWDict['plot1DOutPath'], runOptionsKWDict['plot2DOutPath'], \
                                      runOptionsKWDict['plot3DOutPath']
    plotSummaryRuntimeSpec = [ 
        (doPlot1, 1, p1OutPath, \
                lambda: GetData1DComparisonPlots(timeSpecT, yDataPoints, zDataPoints, plot1DMapFuncs, extraPlotOptionsKWDict)), 
        #(doPlot2, 2, p2OutPath, \
        #        lambda: GetData2DComparisonPlots(timeSpecT, yDataPoints, zDataPoints, plot1DPointMapFunc, extraPlotOptionsKWDict)), 
        (doPlot3, 3, p3OutPath, \
                lambda: GetData3DComparisonPlots(timeSpecT, yDataPoints, zDataPoints, plot1DPointMapFunc, extraPlotOptionsKWDict)), 
    ]
    for (fidx, (execTruthOpt, nD, imgOutPathFmt, runnerFunc)) in enumerate(plotSummaryRuntimeSpec):
        if not execTruthOpt:
            continue
        imgOutPath = stampOutputFilePathFunc(imgOutPathFmt, (pA, pB, pC))
        print("  >> Running the Plot-%dD diagram summary routines with (a, b, c) = (%1.2f, %1.2f, %1.2f) ... " % (nD, pA, pB, pC))
        rfig = runnerFunc()
        print("  >> DONE!")
        plt.savefig(imgOutPath)
        print("  ++ SAVING temporary image to \'%s\' ..." % imgOutPath)
    print("")
    #plt.show(block=False)
    return None

ONLY_ONE_ABC_PARAM=True

if __name__ == "__main__":
    """
    ## NOTE: Color schemes (plot option 'cmap') are easily changed to user taste for appearances. 
    ##       A shortlist of other available built-in plot themes is found in the 
    ##       source file "MatplotlibBase.py" (Python library folder in the top-level repo directory tree). 
    ##       E.g., if you prefer the look of the named colormap 'cubehelix', then add the following 
    ##       uncommented line to the inner loop below:
    ##       ``plotOptionsBaseDict['cmap'] = 'Pastel1'``
    """
    runOptionsBaseDict = DEFAULT_RUNTIME_OPTIONS_KWARGS_DICT
    plotOptionsBaseDict = DEFAULT_PLOT_OPTIONS_DICT 
    plotOptionsBaseDict['cmap'] = 'cubehelix'
    for namedABCParams in COMMON_ABCPARAMS_LOOKUP_BYNAME.keys():
        abcParams =  COMMON_ABCPARAMS_LOOKUP_BYNAME[namedABCParams]
        runOptionsBaseDict['ABC'] = abcParams
        RosserAttractorSolutionSummary(runOptionsKWDict=runOptionsBaseDict, extraPlotOptionsKWDict=plotOptionsBaseDict)
        if ONLY_ONE_ABC_PARAM:
            break
    sys.exit(0)

