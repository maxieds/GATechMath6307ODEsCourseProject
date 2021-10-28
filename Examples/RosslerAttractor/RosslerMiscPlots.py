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
    """ 
    NB: I (MDS) have opinions about open source software and have no intention of politely dancing around 
    a problem working with numerical methods in Python that SHOULD HAVE BEEN, or could have been 
    if only we had tried harder, been fixed by now. I will remain unimpressed by the hoardes 
    of students that plug a large dataset into a Pandas spreadsheet and conclude victory in the 
    name of a successful model for machine learning. 
    Symbolic mathematics with OSS CAS is a huge open problem that limits 
    widespread adoption of the platform, and has kept me using Mathematica with a EULA for 20+ years. 
    Understand that my pronounced b*!chiness on the subject is a symptom, not a problem with me as 
    the end user of large software libraries in the otherwise mature de factor "kitchen sink" for 
    fast prototyping and scripting that we call Python(3). Having to make changes to delicate 
    numerical methods in C++ is tedious and is a prohibitively expensive, time-consuming 
    alternative in these use cases.

    This (agreed) diatribe is also the bulk of the documentation I will write down for the project within 
    the source code in place of Javadocs and/or Doxygen markup. 
    Surprisingly, the programatic nuances of gluing together slices of linearly organized 
    solution data become routine and 
    familiar within an afternoon of sitting with numpy, scipy, SageMath, and their kin (this is the easy part). 
    The hardest thing to grasp about the source code I have written here is somewhat more 
    philosophical. It is also not a popular expression of grief with many developers of the component OSS libraries. 
    It is non-trivial to fix from the established top-down development cycle and is clearly 
    a lot of work to even start organizing a clear plans to work around for the longterm. 
    Experimental mathematics in Python where we work with symbolic parameters is NOT YET there 
    for mass consumption. I am saddened that the cases that we programmers and experimentalists 
    are able to handle well in routine fashion is dumbed down to the level of having to 
    truncate (i.e., to IEEE-standard-codified floating point precision) the solution data. 
    It feels worse to me disappointed today as 3.14, or 22/7, is a mathematical layperson's misnomer for what we 
    mean by the famous centrally transcendental inner workings of PI (thank you Euclid). Similarly, 
    0.577 is unsatisfying to express Euler's gamma constant, 1.64493 would have still intellectually irritated 
    Bernoulli in his Basel problem days, and saying 1.03693 === 337/325 wins no number theorists 
    a Fields medal next year either.  

    ## NOTE: And caveat emptor to newbies to Python and numerical analysis 
    ## implementations in Python where the underlying problem involves symbolic data 
    ## (e.g., parameters). In particular, as of 10/2021, there are still oodles of 
    ## problems with type conversions and operator compatibility amongst the biggest of 
    ## the scientific Python packages that is a SUBSTANTIAL barrier to interoperability. 
    ## SageMath is the best of the bunch, but it still suffers often from self-termed 
    ## "coersion" exceptions when interfacing with common objects from numpy (for example) 
    ## whereby its algorithms are unable to place objects with certain external features 
    ## in a superclass type that is a successor to its functionality as an algebraic object in mathematics. 
    ## Leaving it mildly for now, I think that doing this when operating with variables that are 
    ## contained within the RR (reals), CC (complexes), or even a polynomial extension ring PP[[x]]
    ## is the wrong approach as it completely nullifies the good abstraction of being able 
    ## to run a generic numerical ODE solver type algorithm on non-standardized floating point aware 
    ## data. For example, so long as assumptions about convergence and numerical satibility 
    ## to the correct solution have at least a heuristical basis for being correct, 
    ## or the user is aware of complications, 
    ## we should just consider working with the abstract types and leaving them unevaluated in the 
    ## otherwise traditional numerical data that gets generated stepping through the stock famous algorithms. 
    ## N.b., scipy, sympy, maxima, even FriCAS and others reject processing any intermediate 
    ## data that does not valuate to a real decimal approximation in numerical ODE functionality. 
    ## This is, it happens, also why there ARE NO EXCELLENT (nor even noteworthy) Python implementations of 3D systems of 
    ## first-order ODEs (like this Rosser problem) when there are indeterminate independent parameters in the system. 
    ## Often, we want to vary the indeterminates over a fine-grained mesh themselves without 
    ## needing to invoke another fresh round of the expensive ODE solver to see a close asymptotic 
    ## approximation to the true behavior along the parameter set! 
    ## This is an unfortunate example where beloved freer form traditions in open source software only 
    ## confound matters and make progress slow if not practically impossible. 
    ## THERE NEEDS TO BE A STANDARDIZED SPECIFICATION FOR THIS TYPE OF INTEROPERABILITY BETWEEN THE BIGGEST  
    ## NUMERICAL LIBRARIES, IT SEEMS TO ME, BEFORE CONDITIONS WILL IMPROVE (SO TO SPEAK). 
    ## SIGH... :(
    ##
    ## 
    ## TECHNICAL NOTES ABOUT THE NEXT IMPLEMENTATION DETAIL: 
    ## We consider the transform function to be the Euclidean norm (2-norm) squared of the 
    ## LHS vector field, (YP, ZP), as defined in the code above. 
    ## However, since we have to be able to input objects of type np.matrix, it is 
    ## not possible to simply use our stock Utils.SageMathNorm function as is done in the 
    ## other exploration examples to convert between numerical types from the different 
    ## scientific Python libraries. We can though take powers of a np.matrix object so that 
    ## it works to just define an auxiliary handler function for this task given the fixed 
    ## parameter tuple, (a, b, c), which besides the dependent components, (y, z), 
    ## is the only remaining symbolic data we would have to keep track of here. 
    ## Notice that another nuerical subtlty that arises is raising objects to fractional, 
    ## or even square powers. Therefore, we are careful to square the norm inputs directly 
    ## using repeated multiplication of the relevant computer-object-typed monomials below. 
    """
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

