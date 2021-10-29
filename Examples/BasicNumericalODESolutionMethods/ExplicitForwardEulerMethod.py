#### ExplicitForwardEulerMethod.py
#### Author: Maxie Dion Schmidt
#### Created: 2021.10.23

import numpy as np
import matplotlib.pyplot as plt
import math
from PythonODEBaseLibrary import * 
from MatplotlibBase import *

def ExplicitForwardEuler(ftyFunc, icPoint, solInterval, h, showPlot=True, drawStyle=None, legendLabel=None, figureNum=1):
    f = ftyFunc
    (t0, y0) = icPoint
    (solA, solB) = solInterval
    numGridPoints = math.floor(float((solB - solA) / h))
    tPoints = np.linspace(solA, solB, numGridPoints + 1)
    yPoints = [ y0 ]
    curYn = y0
    for n in range(0, numGridPoints):
        tn = tPoints[n]
        nextYn = curYn + f(tn, curYn) * h
        yPoints += [ nextYn ]
        curYn = nextYn
    pltDrawStyle = GetDistinctDrawStyle(0) if drawStyle == None else drawStyle
    pltLegendLabel = '' if legendLabel == None else legendLabel
    pltFig = plt.figure(figureNum)
    plt.plot(tPoints, yPoints, pltDrawStyle, label=pltLegendLabel, linewidth=1)
    plt.xlabel("Time (t)")
    plt.ylabel('Solution y(t)')
    plt.title(r'Explicit forward Euler method to solve the ODE $y^{\prime}(t) = f(t, y(t))$')
    if showPlot:
        plt.draw()
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
        axFig = ExplicitForwardEuler(ftyFunc, icPoint, solInterval, gridSpacingH, 
                                     showPlot=False, drawStyle=drawStyles[kidx], legendLabel=kthlbl)
    axFig.legend(loc='center right')
    plt.show(block=False)

    hStepParams  = [ 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1 ] 
    drawStyles   = [ GetDistinctDrawStyle(n) for n in range(0, len(hStepParams)) ]
    solInterval  = (0.25, 1.25)
    icPoint      = (0, 1.65)
    for (hidx, hstep) in enumerate(hStepParams):
        gridSpacingH = hstep
        ftyFunc = lambda t, y: -15 * y
        hthlbl = r'$\Delta t = %g$' % hstep
        axFig = ExplicitForwardEuler(ftyFunc, icPoint, solInterval, gridSpacingH, 
                                     showPlot=False, drawStyle=drawStyles[hidx], 
                                     legendLabel=hthlbl, figureNum=2)
    defaultGridSpacingH = 0.0025
    tPoints = np.linspace(solInterval[0], solInterval[1], math.floor((solInterval[1] - solInterval[0]) / defaultGridSpacingH) + 1)
    exactSolYPoints = [ float(np.exp(-15 * tval)) for tval in tPoints ]
    plt.plot(tPoints, exactSolYPoints, 'b', label=r'$y(t) = e^{-15t}$', linewidth=2.5, color='limegreen', alpha=0.4)
    axFig.legend(loc='center right')
    plt.show(block=False)

