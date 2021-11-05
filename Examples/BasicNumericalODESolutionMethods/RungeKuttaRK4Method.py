#### ImplicitRungeKuttaMethod.py
#### Author: Maxie Dion Schmidt
#### Created: 2021.10.23

import numpy as np
import matplotlib.pyplot as plt
import math
from PythonODEBaseLibrary import * 
from MatplotlibBase import *

def RungeKuttaRK4(ftyFunc, icPoint, solInterval, h, plotInterval=None, showPlot=True, drawStyle=None, legendLabel=None, figureNum=1):
    f = ftyFunc
    (t0, y0) = icPoint
    (solA, solB) = solInterval
    numGridPoints = math.floor(float((solB - solA) / h))
    tPoints = np.linspace(solA, solB, numGridPoints + 1)
    yPoints = [ y0 ]
    curYn = y0
    for n in range(0, numGridPoints):
        tn = tPoints[n]
        k1 = f(tn, curYn)
        k2 = f(tn + h / 2.0, curYn + k1 * h / 2.0)
        k3 = f(tn + h / 2.0, curYn + k2 * h / 2.0)
        k4 = f(tn + h, curYn + k3 * h)
        nextYn = curYn + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        yPoints += [ nextYn ]
        curYn = nextYn
    if plotInterval != None:
        yPoints = [ y for (tidx, y) in enumerate(yPoints) if tPoints[tidx] >= plotInterval[0] and tPoints[tidx] <= plotInterval[1] ]
        tPoints = [ t for t in tPoints if t >= plotInterval[0] and t <= plotInterval[1] ]
    pltDrawStyle = GetDistinctDrawStyle(0) if drawStyle == None else drawStyle
    pltLegendLabel = '' if legendLabel == None else legendLabel
    pltFig = plt.figure(figureNum)
    plt.plot(tPoints, yPoints, pltDrawStyle, label=pltLegendLabel, linewidth=1)
    plt.xlabel("Time (t) -- %d points equispaced at difference %1.2f" % (numGridPoints, h))
    plt.ylabel('Solution y(t)')
    plt.title(r'Runge-Kutta (RK4) method to solve the ODE $y^{\prime}(t) = f(t, y(t))$')
    if showPlot:
        plt.show()
    return pltFig

def yPowerODEFunc(kpow):
    return lambda t, y: (t - y**kpow) * (3 - y * t - 2 * (y**2))

if __name__ == "__main__":
    kPowParams   = [ 1.0, 1.5, 2.5, 4.0, 6.2 ]
    drawStyles   = [ GetDistinctDrawStyle(n) for n in range(0, len(kPowParams)) ]
    gridSpacingH = 0.0025
    solInterval  = (0, 8)
    icPoint      = (0, 1)
    for (kidx, kpow) in enumerate(kPowParams):
        ftyFunc = yPowerODEFunc(kpow)
        kthlbl = "k = %1.2f" % kpow
        axFig = RungeKuttaRK4(ftyFunc, icPoint, solInterval, gridSpacingH, 
                              showPlot=False, drawStyle=drawStyles[kidx], legendLabel=kthlbl)
    axFig.legend(loc='center right')
    plt.show(block=False)

    hStepParams  = [ 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1 ] 
    drawStyles   = [ GetDistinctDrawStyle(n) for n in range(0, len(hStepParams)) ]
    solInterval  = (0, 8)
    plotInterval = (0, 0.5)
    icPoint      = (0, 1)
    for (hidx, hstep) in enumerate(hStepParams):
        gridSpacingH = hstep
        ftyFunc = lambda t, y: -15 * y
        hthlbl = r'$\Delta t = %g$' % hstep
        axFig = RungeKuttaRK4(ftyFunc, icPoint, solInterval, gridSpacingH, 
                              plotInterval=plotInterval, 
                              showPlot=False, drawStyle=drawStyles[hidx], 
                              legendLabel=hthlbl, figureNum=2)
    defaultGridSpacingH = 0.0025
    tPoints = np.linspace(plotInterval[0], plotInterval[1], math.floor((plotInterval[1] - plotInterval[0]) / defaultGridSpacingH) + 1)
    exactSolYPoints = [ float(np.exp(-15 * tval)) for tval in tPoints ]
    plt.plot(tPoints, exactSolYPoints, 'b', label=r'$y(t) = e^{-15t}$', linewidth=2.5, color='limegreen', alpha=0.4)
    axFig.legend(loc='center right')
    plt.show(block=False)
