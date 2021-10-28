#### ExploringVectorFieldsAndODESystems.py
#### Maxie Dion Schmidt
#### 2021.10.23

import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy.abc import x, y, t
from sympy import solve, Function
from scipy.integrate import odeint
import math
from PythonODEBaseLibrary import * 
from MatplotlibBase import *

def ComputeEquilibriumPoints(FxyFunc):
    Fxy = lambda x, y: list(FxyFunc(x, y))
    (xyVars, equilibPts) = solve(Fxy(x, y), set=True)
    eqPoints = []
    for xyEqPoint in iter(equilibPts):
        (xe, ye) = xyEqPoint
        eqPoints += [ (xe, ye) ]
    return eqPoints

def PlotVectorField(FxyFunc, xRange=np.linspace(-10, 10, 50), yRange=None, showPlot=True):
    Fxy = lambda x, y: np.array(list(FxyFunc(x, y)))
    if yRange == None:
        xGridPoints, yGridPoints = np.meshgrid(xRange, xRange)
    else:
        xGridPoints, yGridPoints = np.meshgrid(xRange, yRange)
    xv, yv = sympy.var('x y')
    (uQuiver, vQuiver) = Fxy(xGridPoints, yGridPoints)
    xmin, xmax = min(xGridPoints.flatten()), max(xGridPoints.flatten())
    ymin, ymax = min(yGridPoints.flatten()), max(yGridPoints.flatten())
    axFig = plt.figure(2)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.quiver(xGridPoints, yGridPoints, uQuiver, vQuiver)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.streamplot(xGridPoints, yGridPoints, uQuiver, vQuiver)
    plt.title(r'Vector field orientation/flow of $F(x, y)$ in $\dot{x} = F(x)$')
    if showPlot:
        plt.draw() #show()
    return axFig

def SolveODE2DSystemWithVectorField(FxyFunc, icPoint, solInterval, h, showPlot=True, drawStyle=None, legendLabel=None):
    Fxy = lambda s, time: FxyFunc(s[0], s[1])
    (t0, (x0, y0)) = icPoint
    (solA, solB) = solInterval
    numGridPoints = math.floor(float((solB - solA) / h))
    timeSpecT = np.linspace(solA, solB, numGridPoints + 1)
    odeIntSol = odeint(Fxy, [ x0, y0 ], timeSpecT)
    xtSolPoints = odeIntSol[:, 0] 
    ytSolPoints = odeIntSol[:, 1]
    axFig = plt.figure(1) 
    plt.xlabel(r'Time (t)')
    plt.plot(timeSpecT, xtSolPoints, GetDistinctDrawStyle(2), label=r'$x(t)$')
    plt.plot(timeSpecT, ytSolPoints, GetDistinctDrawStyle(6), label=r'$y(t)$')
    plt.title(r'Solutions $(x(t), y(t))$ to the ODE system $(x^{\prime}, y^{\prime}) = F(x, y)$')
    if showPlot:
        plt.draw() #show()
    return axFig

def FxyVectorFieldFromMidtermExam():
    return lambda x, y: (x * (1-x**2-y**2) - y, y * (1-x**2-y**2) - x)

if __name__ == "__main__":
    gridSpacingH = 0.0025
    solInterval  = (0, 8)
    icPoint      = (0, (1, 1))
    Fxy          = FxyVectorFieldFromMidtermExam()
    eqPoints     = ComputeEquilibriumPoints(Fxy)
    print(r'Equilibrium points of $F(x, y)=(0, 0)$ are: %s' % eqPoints)
    PlotVectorField(Fxy, np.linspace(solInterval[0], solInterval[1], 50), showPlot=True)
    print(r'Plotting the underlying vector field $F(x, y)$ ... ')
    SolveODE2DSystemWithVectorField(Fxy, icPoint, solInterval, gridSpacingH, showPlot=True)
    print(r'Plotting the numerical solutions to the system ... ')
    plt.show()

