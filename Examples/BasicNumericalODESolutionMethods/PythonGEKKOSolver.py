#### PythonGEKKOSolver.py
#### Author: Maxie Dion Schmidt (@github/maxieds)
#### Created: 2021.10.07

import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
from PythonODEBaseLibrary import MatplotlibBase
from MatplotlibBase import *

gk = GEKKO(remote=False)
VERBOSE = True

def yPowerODEFunc(kpow):
    return lambda t, y: (t - y**kpow) * (3 - y * t - 2 * (y**2))

if __name__ == "__main__":
    kPowParams    = [ 1.0, 1.5, 2.5 ]
    drawStyles    = [ GetDistinctDrawStyle(n) for n in range(0, len(kPowParams)) ]
    gridSpacingH  = 0.001
    solInterval   = (0, 6.0)
    (solA, solB)  = solInterval
    icPoint       = (0, 1)
    (t0, y0)      = icPoint
    numGridPoints = math.floor(float((solB - solA) / gridSpacingH))
    gk.options.IMODE = 4
    gk.options.TIME_SHIFT = 0
    gk.options.SOLVER = 1
    axFig = plt.figure(1)
    for (kidx, kpow) in enumerate(kPowParams):
        k             = gk.Param()
        y             = gk.Var(value=y0)
        gk.time       = np.linspace(solA, solB, numGridPoints + 1)
        t             = gk.Param(value=gk.time)
        k.value       = kpow
        ftyFunc       = yPowerODEFunc(k)(t, y)
        gk.Equation(y.dt() == ftyFunc)
        gk.options.MAX_ITER = 250 * math.floor(kpow) 
        gk.solve(disp=VERBOSE)
        pltDrawStyle = drawStyles[kidx]
        pltLegendLabel = "k = %1.2f" % kpow
        plt.plot(gk.time, y, pltDrawStyle, label=pltLegendLabel, linewidth=1)
    plt.xlabel("Time (t) -- %d points equispaced at difference %1.2f" % (numGridPoints, gridSpacingH))
    plt.ylabel('Solution y(t)')
    plt.title(r'Explicit forward Euler method to solve the ODE $y^{\prime}(t) = f(t, y(t))$')
    axFig.legend(loc='center right')
    plt.show()

