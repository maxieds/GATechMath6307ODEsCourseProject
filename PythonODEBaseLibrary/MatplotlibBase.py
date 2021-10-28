#### MatplotlibBase.py 
#### Author: Maxie Dion Schmidt (@github/maxieds)
#### Created: 2021.10.07

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from Utils import * 
from scipy.ndimage.filters import gaussian_filter
from sage.all import *

## TODO: There is a plt.errorbar plot type that can be used to visualize differences in data. 
##       Can we (should we) use it to analyze what happens by slightly perturbing the ICs to 
##       the ODE system for any fixed (a, b, c) parameter configuration ??? 

MPL_LINE_STYLES     = [ '-', '--', '-.', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 
                        's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_' ]
MPL_NAMED_COLORS    = [ 'b', 'g', 'r', 'c', 'm', 'y', 'k' ]

def GetDistinctDrawStyle(nthStyle):
    if nthStyle < 0 or nthStyle > len(MPL_LINE_STYLES) * len(MPL_NAMED_COLORS):
        raise RuntimeError("Number of requested draw styles (%d) is invalid" % nthStyle)
    numLineStyles = math.ceil(nthStyle / len(MPL_NAMED_COLORS))
    lineStyle = MPL_LINE_STYLES[numLineStyles]
    colorSpec = MPL_NAMED_COLORS[nthStyle % len(MPL_NAMED_COLORS)]
    fullDrawStyle = "%s%s" % (colorSpec, lineStyle)
    return fullDrawStyle

"""
See also color schemes: 
https://matplotlib.org/stable/tutorials/colors/colormaps.html
"""
MPL_NAMED_COLORMAPS = [ 'magma', 'nipy_spectral', 'tab20c', 'brg', 'spring', 'flag', 
                        'prism', 'ocean', 'Set2', 'Pastel1', 'cubehelix', 'CMRmap', 
                        'terrain', 'RdYlGn', 'PiYG', 'BrBG', 'RdYlBu', 'viridis', 'hsv' ] 

DEFAULT_PLOT_OPTIONS_DICT = {
     'bins'              : 'log', 
     'cmap'              : 'PiYG',
     'C'                 :  None,
     'histbins'          :  42, 
     'filledcontourplot' :  True, 
     'edgecolor'         :  None, 
     'alpha'             :  0.5, 
     'histtype'          : 'stepfilled', 
     'normed'            :  True, 
     'colors'            :  'lightblue', 
     'tickstyle'         :  'logspaced', 
     'ticklocator'       :  
     {
         'none'          : plt.NullLocator(), 
         'evenlyspaced'  : plt.LinearLocator(), 
         'logspaced'     : plt.LogLocator(), 
         'best4'         : plt.MaxNLocator(4),
         'best8'         : plt.MaxNLocator(8),
         'best16'        : plt.MaxNLocator(16),
         'best50'        : plt.MaxNLocator(50),
         'best100'       : plt.MaxNLocator(100),
     }, 
     'legendframe'       :  True,
     'sharex'            :  True, 
     'sharey'            :  True,
     'hspace'            :  32, 
     'wspace'            :  32, 
     # Other values for x/yscale: 'linear', 'symlog', 'symlogx', 'symlogy', 'semilogy', 'semilogx', 
     #                            'logit', 'function', 'functionlog'
     'xscale'            :  'linear', #'log',
     'yscale'            :  'linear',  
     'scalebase'         :  2, 
     'textpadleft'       :  0.05, 
     'textpadright'      :  0.95, 
     'fontsize'          :  15, 
     'fontweight'        :  'bolditalic',
     'ha'                :  'center', 
     'va'                :  'top',
}

def TrimAxs(fullAxs, fig, N):
    for (eidx, ax) in enumerate(fullAxs.flat):
        if eidx <= N:
            continue
        fig.delaxes(ax)
    return fig, fullAxs

def ApplyGlobalFigureDiagramOptions(fig, axs, xyAxesLims=None, plotOptionsDict=None):
    if plotOptionsDict == None:
        raise ValueError
    getPlotOptionFunc = lambda kwd, default: default if kwd not in plotOptionsDict.keys() else plotOptionsDict[kwd]
    hsp, wsp = getPlotOptionFunc('hspace', 0), getPlotOptionFunc('wspace', 0)
    xsc, ysc = getPlotOptionFunc('xscale', None), getPlotOptionFunc('yscale', None)
    tickstyle = getPlotOptionFunc('tickstyle', None)
    tickFormatterFunc = plotOptionsDict['ticklocator'][tickstyle]
    txtPadLeft, txtPadRight = getPlotOptionFunc('textpadleft', 0.05), getPlotOptionFunc('textpadright', 0.95)
    fontSize, fontWeight = getPlotOptionFunc('fontsize', 16), getPlotOptionFunc('fontweight', 'heavy')
    hap, vap = getPlotOptionFunc('ha', 'center'), getPlotOptionFunc('va', 'top')
    for axsIdx in range(0, len(axs)):
        ax = axs.flatten()[axsIdx]
        ax.set_aspect('auto', adjustable='datalim')
        plt.sca(ax)
        plt.gca().set_xscale(xsc)
        plt.gca().set_yscale(ysc)
        #if xyAxesLims != None:
        #    (xmin, xmax, ymin, ymax) = xyAxesLims
        #    plt.xlim(xmin, xmax)
        #    plt.ylim(ymin, ymax)
    fig.subplots_adjust(hspace=hsp, wspace=wsp)
    #fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98)
    fig.tight_layout(pad=1)
    #plt.axes.set_major_formatter(tickFormatterFunc)
    #plt.axes.set_major_locator(tickFormatterFunc)
    return axs

def VectorFieldVisualizationPlotsDiagram(xDataPoints, yDataPoints, tfMapFunc=None, ax=None, vectorMapFuncs=None, 
                                         plotOptionsDict=DEFAULT_PLOT_OPTIONS_DICT): 
    """
    Idea for the alpha-opacity based postprocessing from: 
    https://stackoverflow.com/questions/54457141/is-there-anything-in-matplotlib-that-behaves-like-alpha-but-reversed
    """
    if ax == None or tfMapFunc == None or vectorMapFuncs == None or plotOptionsDict == None:
        raise ValueError
    baseXX, baseYY = np.meshgrid(
        np.linspace(np.amin(xDataPoints), np.amax(xDataPoints), len(xDataPoints)),
        np.linspace(np.amin(yDataPoints), np.amax(yDataPoints), len(yDataPoints))
    )
    XX, YY = np.meshgrid(xDataPoints, yDataPoints)
    u = vectorMapFuncs[0](XX, YY)
    v = vectorMapFuncs[1](XX, YY)
    ZZ = tfMapFunc(XX, YY)
    cmapSetting = 'RdGy' if 'cmap' not in plotOptionsDict.keys() else plotOptionsDict['cmap']
    #ax.patch.set_facecolor('none')
    #ax.patch.set_alpha(0.92)
    ax.set_title('Vector field plots')
    plt.sca(ax)
    plt.streamplot(baseXX, baseYY, u, v, cmap=cmapSetting)
    plt.quiver(baseXX, baseYY, u, v, cmap=cmapSetting)
    plt.imshow(ZZ, extent=(np.amin(XX), np.amax(XX), np.amin(YY), np.amax(YY)), origin='lower', cmap=cmapSetting, alpha=0.5)
    plt.colorbar()
    plt.title(r'Solution flow for $F(y, z) = (ay, b-cz)^T$')
    plt.xlim(min(xDataPoints), max(xDataPoints))
    plt.ylim(min(yDataPoints), max(yDataPoints))
    return ax

def BifurcationDiagram(t, xPoints, yPoints, n, ax=None, pointMapFunc=lambda pt: pt, plotOptionsDict=DEFAULT_PLOT_OPTIONS_DICT): 
    if ax == None or pointMapFunc == None or plotOptionsDict == None:
        raise ValueError
    dataPoints = list(zip(xPoints, yPoints))
    tfDataPoints = list(map(lambda ptSpec: pointMapFunc(ptSpec[0], ptSpec[1]), dataPoints))
    plt.sca(ax)
    plt.plot(t, tfDataPoints, 'k', lw=2)
    plt.plot([ 0, 1 ], [ 0, 1 ], 'k', lw=2)
    xsc, ysc = plotOptionsDict['xscale'], plotOptionsDict['yscale']
    plt.xscale(xsc)
    plt.yscale(ysc)
    tv = t[0]
    numIterUpper = min(n, len(xPoints), len(yPoints))
    for midx in range(0, numIterUpper):
        yv = tfDataPoints[midx]
        ax.plot([ tv, tv], [ tv, yv ], 'k', lw=1)
        ax.plot([ tv, yv], [ yv, yv ], 'k', lw=1)
        ax.plot([ tv ], [ yv ], 'ok', ms=10, alpha=(midx + 1) / numIterUpper)
        tv = yv
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(np.amin(tfDataPoints), np.amax(tfDataPoints))
    ax.set_title('Bifurcation diagram after transformation')
    return ax

def ContourPlot3DDiagram(xPoints, yPoints, tfMapFunc=None, ax=None, plotOptionsDict=DEFAULT_PLOT_OPTIONS_DICT):
    if ax == None or plotOptionsDict == None or tfMapFunc == None:
        raise ValueError
    XX, YY = np.meshgrid(xPoints, yPoints)
    ZZ = tfMapFunc(XX, YY)
    cmapPlotSettings = 'YlGnBu_r' if 'cmap' not in plotOptionsDict.keys() else plotOptionsDict['cmap']
    ax.set_title('Contour plot of 3D points')
    plt.sca(ax)
    if 'filledcontourplot' not in plotOptionsDict.keys() or not plotOptionsDict['filledcontourplot']:
        plt.contour(XX, YY, ZZ, cmap=cmapPlotSettings, antialiased=True)
    else:
        plt.contourf(XX, YY, ZZ, 20, cmap=cmapPlotSettings, antialiased=True)
    plt.colorbar()
    return ax

def Histogram1DComparisonDiagram(xPoints, ax=None, plotOptionsDict=DEFAULT_PLOT_OPTIONS_DICT):
    if ax == None or plotOptionsDict == None:
        raise ValueError
    kwargsDict = {
        'histtype' : 'stepfilled', 
        'alpha'    : 0.42, 
        'density'  : True, 
        'bins'     : 72
    }
    ax.set_title('1D Histograms of X/Y points')
    ax.hist(xPoints, label=r'y/z(t) points', color='blue', **kwargsDict)
    return ax

def Barchart3DDiagram(xPoints, yPoints, ax=None, plotOptionsDict=DEFAULT_PLOT_OPTIONS_DICT):
    """
    Fancier example of a histogram type 3D bar chart plot is adapted from: 
    https://stackoverflow.com/a/51624315/10661959
    """
    if ax == None or plotOptionsDict == None:
        raise ValueError
    xPoints, yPoints = np.array(xPoints), np.array(yPoints)
    numBins2d = (50, 50)
    xyHist, xEdges, yEdges = np.histogram2d(xPoints, yPoints, bins=numBins2d)
    XX, YY = np.meshgrid(xEdges[:-1] + xEdges[1:], yEdges[:-1] + yEdges[1:])
    XX, YY = XX.flatten() / 2.0, YY.flatten() / 2.0
    ZZ = np.zeros_like(XX)
    dx, dy, dz = xEdges[1] - xEdges[0], yEdges[1] - yEdges[0], xyHist.flatten()
    minHeight, maxHeight = np.amin(dz), np.amax(dz)
    cmapOption = 'YlGnBu_r' if 'cmap' not in plotOptionsDict.keys() else plotOptionsDict['cmap']
    cmap = cm.get_cmap(cmapOption)
    rgba = [ cmap( (k - minHeight) / maxHeight ) for k in dz ]
    ax.bar3d(XX, YY, ZZ, dx, dy, dz, color=rgba, zsort='average')
    plt.sca(ax)
    plt.xlabel('Y ODE solution points')
    plt.ylabel('Z ODE solution points')
    plt.title('3D histogram (normalized density plot)')
    return ax

def ScatterPlot2DDiagram(dataPoints, ax=None, plotOptionsDict=DEFAULT_PLOT_OPTIONS_DICT):
    """ 
    Idea for heatmap color like smoothing procedure of coarse data points from: 
    https://stackoverflow.com/a/47350350/10661959
    """
    if ax == None or plotOptionsDict == None:
        raise ValueError
    (xDataPoints, yDataPoints) = zip(*dataPoints)
    alpha = 1.0 if 'alpha' not in plotOptionsDict.keys() else plotOptionsDict['alpha']
    cmap = 'cubehelix' if 'cmap' not in plotOptionsDict.keys() else plotOptionsDict['cmap']
    binSize = math.ceil(min(np.amax(xDataPoints)-np.amin(xDataPoints), np.amax(yDataPoints) - np.amin(yDataPoints)) / 42.0)
    getCoordAvgFunc = lambda coordPts: float(np.dot(np.histogram(coordPts, bins=binSize)[0], \
                                              np.histogram(coordPts, bins=binSize)[1][1:]) / float(len(coordPts)))
    xAve, yAve = getCoordAvgFunc(xDataPoints), getCoordAvgFunc(yDataPoints)
    xAve = np.array([ xAve for count in range(0, len(xDataPoints)) ] )
    yAve = np.array([ yAve for count in range(0, len(yDataPoints)) ] )
    colorsOption = np.sqrt((xDataPoints - xAve)**2 + (yDataPoints - yAve)**2)
    ax.scatter(xDataPoints, yDataPoints, c=colorsOption, alpha=alpha, cmap=cmap)
    heatmap, xedges, yedges = np.histogram2d(xDataPoints, yDataPoints, bins=1000)
    heatmap = gaussian_filter(heatmap, sigma=18)
    extent = [ xedges[0], xedges[-1], yedges[0], yedges[-1] ]
    ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap)
    ax.set_title('Smoothed and highlighted scatter plot in the plane')
    return ax

def ScatterPlot3DDiagram(xPoints, yPoints, tfMapFunc=None, ax=None, plotOptionsDict=DEFAULT_PLOT_OPTIONS_DICT):
    if ax == None or tfMapFunc == None or plotOptionsDict == None:
        raise ValueError
    XX, YY = np.meshgrid(xPoints, yPoints)
    ZZ = tfMapFunc(XX, YY)
    lineC = None if 'C' not in plotOptionsDict.keys() else plotOptionsDict['C']
    cmapPlotOption = 'hsv' if 'cmap' not in plotOptionsDict.keys() else plotOptionsDict['cmap']
    plt.sca(ax)
    plt.scatter(XX, YY, ZZ, c=lineC, cmap=cmapPlotOption)
    plt.title('3D scatter plot with z-component transformed')
    return ax

def TrisurfacePlot3DDiagram(xPoints, yPoints, tfMapFunc=None, ax=None, plotOptionsDict=DEFAULT_PLOT_OPTIONS_DICT):
    if ax == None or tfMapFunc == None or plotOptionsDict == None:
        raise ValueError
    zPoints = tfMapFunc(xPoints, yPoints)
    cmapPlotOption = 'hsv' if 'cmap' not in plotOptionsDict else plotOptionsDict['cmap']
    ax.plot_trisurf(xPoints, yPoints, zPoints, cmap=cmapPlotOption, edgecolor='none')
    ax.set_title('Surface plot of the 3D data')
    return ax

def SurfacePlot3DDiagram(xPoints, yPoints, tfMapFunc=None, ax=None, plotOptionsDict=DEFAULT_PLOT_OPTIONS_DICT):
    if ax == None or tfMapFunc == None or plotOptionsDict == None:
        raise ValueError
    XX, YY = np.meshgrid(xPoints, yPoints)
    ZZ = tfMapFunc(XX, YY)
    cmapPlotOption = 'hsv' if 'cmap' not in plotOptionsDict else plotOptionsDict['cmap']
    ecPlotOption = 'none' if 'edgecolor' not in plotOptionsDict else plotOptionsDict['edgecolor']
    ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap=cmapPlotOption, edgecolor=ecPlotOption)
    ax.set_title('Surface plot of the 3D data')
    return ax

def WireframePlot3DDiagram(xPoints, yPoints, tfMapFunc=None, ax=None, plotOptionsDict=DEFAULT_PLOT_OPTIONS_DICT):
    if ax == None or tfMapFunc == None or plotOptionsDict == None:
        raise ValueError
    XX, YY = np.meshgrid(xPoints, yPoints)
    ZZ = tfMapFunc(XX, YY)
    cmapPlotOption = 'hsv' if 'cmap' not in plotOptionsDict else plotOptionsDict['cmap']
    ecPlotOption = 'none' if 'edgecolor' not in plotOptionsDict else plotOptionsDict['edgecolor']
    ax.plot_wireframe(XX, YY, ZZ, cmap=cmapPlotOption, edgecolor=ecPlotOption)
    ax.set_title('Wireframed outline of the 3D data')
    return ax

def GetData1DComparisonPlots(t, xPoints, yPoints, dataMapFuncs=None, plotOptionsDict=None, showPlot=True):
    if dataMapFuncs == None or plotOptionsDict == None:
        raise ValueError
    figRows, figCols, numPlots = 2, 2, 4
    figSize = (12, 12)
    fig = plt.figure(figsize=figSize, dpi=150)
    axs = np.array([ fig.add_subplot(figRows, figCols, ctr+1) for ctr in range(0, figRows * figCols) ])
    oneDPlotSpecs = [ 
        lambda ax: Histogram1DComparisonDiagram(xPoints, ax, plotOptionsDict), 
        lambda ax: Histogram1DComparisonDiagram(yPoints, ax, plotOptionsDict),
        lambda ax: BifurcationDiagram(t, xPoints, yPoints, 75, ax, dataMapFuncs[0], plotOptionsDict),
        lambda ax: VectorFieldVisualizationPlotsDiagram(xPoints, yPoints, dataMapFuncs[0], ax, dataMapFuncs[1], plotOptionsDict), 
    ]
    for (aidx, axBlank) in enumerate(list(axs.flatten())):
        if aidx >= numPlots:
            break
        axFilled = oneDPlotSpecs[aidx](axBlank)
    TrimAxs(axs, fig, numPlots)  
    xyLims = (np.amin(xPoints), np.amax(xPoints), np.amin(yPoints), np.amax(yPoints))
    return ApplyGlobalFigureDiagramOptions(fig, axs, xyLims, plotOptionsDict)

def GetData2DComparisonPlots(t, xPoints, yPoints, tfMapFunc=None, plotOptionsDict=None, showPlot=True):
    if plotOptionsDict == None or tfMapFunc == None:
        raise ValueError
    figRows, figCols, numPlots = 1, 1, 1
    figSize = (1, 1)
    fig = plt.figure(figsize=figSize)
    axs = fig.subplots(figRows, figCols, squeeze=False)
    dataPoints = zip(xPoints, yPoints)
    twoDPlotSpecs = [ 
        lambda ax: ScatterPlot2DDiagram(dataPoints, ax, plotOptionsDict),
    ]
    for (aidx, axBlank) in enumerate(list(axs.flatten())):
        if aidx >= numPlots:
            break
        axFilled = twoDPlotSpecs[aidx](axBlank)
    TrimAxs(axs, fig, numPlots)
    xyLims = (np.amin(xPoints), np.amax(xPoints), np.amin(yPoints), np.amax(yPoints))
    return ApplyGlobalFigureDiagramOptions(fig, axs, xyLims, plotOptionsDict)

def GetData3DComparisonPlots(t, xPoints, yPoints, tfMapFunc=None, plotOptionsDict=None, showPlot=True):
    if plotOptionsDict == None or tfMapFunc == None:
        raise ValueError
    figRows, figCols = 2, 3
    figSize = (18, 8)
    fig = plt.figure(figsize=figSize)
    axs = [ fig.add_subplot(figRows, figCols, ctr+1, projection='3d') for ctr in range(0, figRows * figCols) ]
    threeDPlotSpecs = [ 
        lambda ax: ContourPlot3DDiagram(xPoints, yPoints, tfMapFunc, ax, plotOptionsDict), 
        lambda ax: Barchart3DDiagram(xPoints, yPoints, ax, plotOptionsDict),
        lambda ax: ScatterPlot3DDiagram(xPoints, yPoints, tfMapFunc, ax, plotOptionsDict), 
        lambda ax: TrisurfacePlot3DDiagram(xPoints, yPoints, tfMapFunc, ax, plotOptionsDict),
        lambda ax: SurfacePlot3DDiagram(xPoints, yPoints, tfMapFunc, ax, plotOptionsDict), 
        lambda ax: WireframePlot3DDiagram(xPoints, yPoints, tfMapFunc, ax, plotOptionsDict),
    ]
    for (aidx, axBlank) in enumerate(axs):
        axFilled = threeDPlotSpecs[aidx](axBlank)
    xyLims = (np.amin(xPoints), np.amax(xPoints), np.amin(yPoints), np.amax(yPoints))
    return ApplyGlobalFigureDiagramOptions(fig, np.array(axs), xyLims, plotOptionsDict)

