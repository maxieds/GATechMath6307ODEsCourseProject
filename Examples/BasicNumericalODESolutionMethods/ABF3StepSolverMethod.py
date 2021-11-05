#### ImplicitStepSolverMethod.py
#### Author: Maxie Dion Schmidt
#### Created: 2021.10.23

import numpy as np
import matplotlib.pyplot as plt
import math
from PythonODEBaseLibrary import * 
from MatplotlibBase import *
import sympy
from sympy import factorial
from sympy.abc import t as tvar
from functools import reduce
import operator

def LagrangePolynomialInterpolation(ftyFunc, numStepsS, prevYPoints, t0, h, n):
    f             = ftyFunc
    s             = numStepsS 
    yPoints       = prevYPoints
    tDiffProdFunc = lambda t, j: reduce(operator.mul, [ t - (t0 + h * (n + i)) if i != j else 1 for i in range(0, s) ])
    ptFunc        = lambda t: sum([ (-1)**(s-j-1) * f(t0 + h * (n + j), yPoints[n+j]) / factorial(j) / factorial(s-j-1) / \
                                    (h**(s-1)) * tDiffProdFunc(t, j) for j in range(0, n + len(yPoints)) ])
    nextYPoints   = []
    lastYPoint    = prevYPoints[-1]
    for sidx in range(0, s):
        tnpim1 = t0 + h * (n + sidx - 1)
        tnpi = t0 + h * (n + sidx)
        ynpi = lastYPoint + sympy.integrate(ptFunc(tvar), (tvar, tnpim1, tnpi))
        yPoints += [ ynpi ]
        nextYPoints += [ ynpi ]
        lastYPoint = ynpi
    return nextYPoints

def AdamsBashforthABF3(ftyFunc, icPoint, solInterval, h, plotInterval=None, showPlot=True, drawStyle=None, legendLabel=None, figureNum=1):
    f             = ftyFunc
    s             = 3
    (t0, y0)      = icPoint
    (solA, solB)  = solInterval
    numGridPoints = math.floor(float((solB - solA) / h))
    tPoints       = np.linspace(solA, solB, numGridPoints + 1)
    yPoints       = [ y0 ] + LagrangePolynomialInterpolation(f, s-1, [ y0 ], t0, h, n=0)
    curYn         = y0
    for n in range(0, numGridPoints + 1 - s):
        tn2, tn1, tn = tPoints[n+2], tPoints[n+1], tPoints[n]
        yn2, yn1, yn = yPoints[n+2], yPoints[n+1], yPoints[n]
        fn2, fn1, fn = f(tn2, yn2), f(tn1, yn1), f(tn, yn)
        nextYn = yn2 + h * (23.0 / 12.0 * fn2 - 4.0 / 3.0 * fn1 + 5.0 / 12.0 * fn)
        yPoints += [ nextYn ]
    if plotInterval != None:
        yPoints = [ y for (tidx, y) in enumerate(yPoints) if tPoints[tidx] >= plotInterval[0] and tPoints[tidx] <= plotInterval[1] ]
        tPoints = [ t for t in tPoints if t >= plotInterval[0] and t <= plotInterval[1] ]
    pltDrawStyle = GetDistinctDrawStyle(0) if drawStyle == None else drawStyle
    pltLegendLabel = '' if legendLabel == None else legendLabel
    pltFig = plt.figure(figureNum)
    plt.plot(tPoints, yPoints, pltDrawStyle, label=pltLegendLabel, linewidth=1)
    plt.xlabel("Time (t) -- %d points equispaced at difference %1.2f" % (numGridPoints, h))
    plt.ylabel('Solution y(t)')
    plt.title(r'ABF-3 method to solve the ODE $y^{\prime}(t) = f(t, y(t))$')
    if showPlot:
        plt.draw()
    return pltFig

def yPowerODEFunc(kpow):
    return lambda t, y: (t - y**kpow) * (3 - y * t - 2 * (y**2))

if __name__ == "__main__":
    kPowParams   = [ 1.0, 1.5, 2.5, 4.0, 6.2 ]
    drawStyles   = [ GetDistinctDrawStyle(n) for n in range(0, len(kPowParams)) ]
    gridSpacingH = 0.0025
    solInterval  = (0, 2)
    icPoint      = (0, 1)
    for (kidx, kpow) in enumerate(kPowParams):
        ftyFunc = yPowerODEFunc(kpow)
        kthlbl = "k = %1.2f" % kpow
        axFig = AdamsBashforthABF3(ftyFunc, icPoint, solInterval, gridSpacingH, 
                                   showPlot=False, drawStyle=drawStyles[kidx], legendLabel=kthlbl)
    axFig.legend(loc='center right')
    plt.show(block=False)

    hStepParams  = [ 0.001, 0.005, 0.01, 0.05, 0.1 ] 
    drawStyles   = [ GetDistinctDrawStyle(n) for n in range(0, len(hStepParams)) ]
    solInterval  = (0, 2)
    plotInterval = (0, 0.5)
    icPoint      = (0, 1)
    for (hidx, hstep) in enumerate(hStepParams):
        gridSpacingH = hstep
        ftyFunc = lambda t, y: -15 * y
        hthlbl = r'$\Delta t = %g$' % hstep
        axFig = AdamsBashforthABF3(ftyFunc, icPoint, solInterval, gridSpacingH, 
                                   plotInterval=plotInterval, 
                                   showPlot=False, drawStyle=drawStyles[hidx], 
                                   legendLabel=hthlbl, figureNum=2)
    defaultGridSpacingH = 0.0025
    tPoints = np.linspace(plotInterval[0], plotInterval[1], math.floor((plotInterval[1] - plotInterval[0]) / defaultGridSpacingH) + 1)
    exactSolYPoints = [ float(np.exp(-15 * tval)) for tval in tPoints ]
    plt.plot(tPoints, exactSolYPoints, 'b', label=r'$y(t) = e^{-15t}$', linewidth=2.5, color='limegreen', alpha=0.4)
    axFig.legend(loc='center right')
    plt.show(block=False)
