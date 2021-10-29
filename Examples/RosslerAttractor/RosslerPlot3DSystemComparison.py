#### RosslerPlot3DSystemComparison.py
#### Maxie Dion Schmidt
#### 2021.10.29

import numpy as np
import matplotlib.pyplot as plt
import sympy
import sys
import math
import cmath
from sage.all import *
from PythonODEBaseLibrary import *
from Utils import * 
from MatplotlibBase import *
from scipy.interpolate import LinearNDInterpolator

PLOT_3DXYZ_AXES=True
X, Y, Z, T = var('x y z t')

DEFAULT_TIME_INT=[ -20.0, 20.0 ]
DEFAULT_NUM_POINTS=1250
DEFAULT_IC=(0, (1, 1, 1))

PLOT_SUMMARY_TEMP_OUTFILE='../../Images/TempOutputData/RosslerAttractorSage3DSolvers-%s.png'
stampOutputFilePathFunc = lambda: PLOT_SUMMARY_TEMP_OUTFILE % (Utils.GetTimestamp())

SAGE_NUMERICAL_ODE_METHOD_DESC = dict([
    ('desolve_odeint',     'Uses scipi.integrate.odeint -- adaptive solver'), 
    ('ode_solver_rkf45',   'Sage ode_solver() method with \nalgorithm=\'rkf45\' (RK-Fehlberg)'), 
    ('ode_solver_rk2',     'Sage ode_solver() method with \nalgorithm=\'rk2\' (embedded RK)'), 
    ('ode_solver_rk4',     'Sage ode_solver() method with \nalgorithm=\'rk4\' (classic RK4)'), 
    ('ode_solver_rk8pd',   'Sage ode_solver() method with \nalgorithm=\'rk8pd\' (RK Prince-Dormand)'), 
    ('ode_solver_rk2imp',  'Sage ode_solver() method with algorithm=\'rk2imp\' \n(implicit second order RK as Gaussian points)'), 
    ('ode_solver_rk4imp',  'Sage ode_solver() method with algorithm=\'rk4imp\' \n(implicit fourth order RK as Gaussian points)'), 
    #('ode_solver_bsimp',   'Sage ode_solver() method with \nalgorithm=\'bsimp\' (implicit Burlisch-Stoer)'), 
    ('ode_solver_gear1',   'Sage ode_solver() method with \nalgorithm=\'gear1\' (M=1 implicit gear)'), 
    ('ode_solver_gear2',   'Sage ode_solver() method with \nalgorithm=\'gear2\' (M=2 implicit gear)'), 
])

SAGE_ODE_SOLVER_KEYS = [ 
    'ode_solver_rkf45', 
    'ode_solver_rk2',
    'ode_solver_rk4',
    'ode_solver_rk8pd', 
    'ode_solver_rk2imp', 
    'ode_solver_rk4imp', 
    'ode_solver_bsimp', 
    'ode_solver_gear1', 
    'ode_solver_gear2',
]

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


def Plot3DRosslerSolution(ics, timeIntAB, abcParams, ax, sageODEMethod=None, displayColorbar=True):
    if sageODEMethod == None:
        raise ValueError
    (a, b, c) = abcParams
    t0 = ics[0]
    (x0, y0, z0) = ics[1]
    (solA, solB) = timeIntAB
    jacobianJ = lambda t, xyz, abc: [ [ 0.0, -1.0, -1.0 ], [ 1.0, abc[0], 0.0 ], [ xyz[2], 0.0, xyz[0] - abc[2] ] ]
    functionF = lambda t, xyz, abc: [ -(xyz[1] + xyz[2]), xyz[0] + abc[0] * xyz[1], abc[1] + xyz[2] * (xyz[0] - abc[2]) ]
    initConds = [ x0, y0, z0 ]
    odeMethodAlg = sageODEMethod.replace('ode_solver_', '') if sageODEMethod.find('ode_solver_') >= 0 else 'rk4'
    T = ode_solver()
    T.algorithm = odeMethodAlg
    T.function = functionF
    T.jacobian = jacobianJ
    T.y_0 = initConds
    if sageODEMethod == 'desolve_odeint':
        fxyz = [ -(Y + Z), X + a * Y, b + Z * (X - c) ]
        tSpec = list(np.linspace(solA, solB, DEFAULT_NUM_POINTS))
        odeVars = [ X, Y, Z ]
        xyzPoints = desolve_odeint(fxyz, initConds, tSpec, odeVars)
        xSolPoints, ySolPoints, zSolPoints = xyzPoints[:, 0], xyzPoints[:, 1], xyzPoints[:, 2]
    elif sageODEMethod in SAGE_ODE_SOLVER_KEYS:
        T.ode_solve(y_0=initConds, t_span=DEFAULT_TIME_INT, params=[ a, b, c ], num_points=DEFAULT_NUM_POINTS)
        xyzPoints = np.array([ ynTuple for (tk, ynTuple) in T.solution ])
        xSolPoints, ySolPoints, zSolPoints = xyzPoints[:, 0], xyzPoints[:, 1], xyzPoints[:, 2]
    else:
        raise ValueError(sageODEMethod)
    mgXX, mgYY = np.meshgrid(np.array(xSolPoints), np.array(ySolPoints))
    plot3dGridInterpFunc = LinearNDInterpolator(list(zip(xSolPoints, ySolPoints)), zSolPoints)
    mgZZ = plot3dGridInterpFunc(mgXX, mgYY)
    plt.sca(ax)
    if PLOT_3DXYZ_AXES:
        ax.plot_surface(mgXX, mgYY, mgZZ, rstride=1, cstride=1, cmap='coolwarm',
                        linewidth=0, antialiased=False, vmin=-1, vmax=1)
        #plt.tricontourf(xSolPoints, ySolPoints, zSolPoints, cmap='cubehelix', edgecolor='none')
        ax.set_zlabel(r'$z(t)$')
    else:
        plt.contourf(mgXX, mgYY, mgZZ, cmap='magma')
    if displayColorbar and not PLOT_3DXYZ_AXES:
        plt.colorbar(pad=0.65, shrink=0.685)
    ax.set_xlabel(r'$x(t)$')
    ax.set_ylabel(r'$y(t)$')
    ax.set_aspect('auto', adjustable='datalim')
    axTitle = '\n'.join([
        SAGE_NUMERICAL_ODE_METHOD_DESC[sageODEMethod],
        r'$\mathbf{(a, b, c) = (%1.3f, %1.3f, %1.3f)}$' % (a, b, c)
    ])
    ax.set_title(axTitle, fontweight='bold', fontsize=10)
    return ax

if __name__ == "__main__":
     fig = plt.figure(figsize=(18.0, 12.0), constrained_layout=True)
     figRows, figCols = 3, 3
     namedABCParams = list(COMMON_ABCPARAMS_LOOKUP_BYNAME.keys())
     for (pltIdx, algKey) in enumerate(list(SAGE_NUMERICAL_ODE_METHOD_DESC.keys())):
         if PLOT_3DXYZ_AXES:
             ax = fig.add_subplot(figRows, figCols, pltIdx + 1, projection='3d')
         else:
             ax = fig.add_subplot(figRows, figCols, pltIdx + 1)
         displayColorbarQ = True #(pltIdx + 1) % figRows == 0
         abcParams = COMMON_ABCPARAMS_LOOKUP_BYNAME[ namedABCParams[pltIdx] ]
         ax = Plot3DRosslerSolution(DEFAULT_IC, DEFAULT_TIME_INT, abcParams, ax, sageODEMethod=algKey, displayColorbar=displayColorbarQ)
     fig.tight_layout(pad=1)
     fig.subplots_adjust(top=0.85, bottom=0.15, right=0.9, left=0.1, hspace=0.6, wspace=0.6)
     summaryTitle = '\n'.join([ 
        r'Rossler attractor solutions for $\mathbf{t \in [%1.3f, %1.3f]}$' % (DEFAULT_TIME_INT[0], DEFAULT_TIME_INT[1]), 
        r'Comparison of numerical methods in SageMath'
     ])
     fig.suptitle(summaryTitle, fontsize=20, fontweight='bold')
     plt.savefig(stampOutputFilePathFunc(), dpi=150, tight=True)

