"""
Some useful snippets
--------------------
### Legends - boxes instead of lines
leg = ax[-1].legend(
    ncol=1, columnspacing=0.5, frameon=False,
    handletextpad=0.5, handlelength=0.8,
    loc='upper left', bbox_to_anchor=(1.02,1),
)
for legobj in leg.legendHandles:
        legobj.set_linewidth(8)
        legobj.set_solid_capstyle('butt')
leg.set_title('title', prop={'size':'large'})

### Legends - two legends

### Legend - reverse order
handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(
    handles=handles[::-1], labels=labels[::-1],
    ncol=1, columnspacing=0.5, frameon=False,
    # handletextpad=0.5, handlelength=0.8,
    handletextpad=0.3, handlelength=0.7,
    loc='center left', bbox_to_anchor=(1.02, 0.4),
)

"""
import pandas as pd
import numpy as np
import os, copy, shapely
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    AutoMinorLocator, MultipleLocator, AutoLocator, PercentFormatter)

###########################
### IMPORT PROJECT PATH ###
import zephyr.settings
projpath = zephyr.settings.projpath
datapath = zephyr.settings.datapath

###################
### Plot formatting

def plotparams():
    """
    Format plots
    """
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams['mathtext.it'] = 'Arial:italic'
    plt.rcParams['mathtext.bf'] = 'Arial:bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 'x-large'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.labelsize'] = 'large'
    plt.rcParams['ytick.labelsize'] = 'large'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    # plt.rcParams['figure.figsize'] = 6.4, 4.8 # 1.33, matplotlib default
    # plt.rcParams['figure.figsize'] = 4.792, 3.458 # 1.38577, old default
    # plt.rcParams['figure.figsize'] = 5.0, 4.0 # 1.25
    plt.rcParams['figure.figsize'] = 5.0, 3.75 # 1.33, fits 4x in ppt slide
    plt.rcParams['xtick.major.size'] = 4 # default 3.5
    plt.rcParams['ytick.major.size'] = 4 # default 3.5
    plt.rcParams['xtick.minor.size'] = 2.5 # default 2
    plt.rcParams['ytick.minor.size'] = 2.5 # default 2

### Percentages to hex values:
percent2hex = {
    100: 'FF', 95: 'F2', 90: 'E6', 85: 'D9', 80: 'CC', 75: 'BF', 70: 'B3',
    65:  'A6', 60: '99', 55: '8C', 50: '80', 45: '73', 40: '66', 35: '59',
    30:  '4D', 25: '40', 20: '33', 15: '26', 10: '1A',  5: '0D',  0: '00',
}

### Sequential color dict
def rainbowmapper(iterable, colormap=None, explicitcolors=False):
    if colormap is not None:
        if type(colormap) is list:
            colors=[colormap[i] for i in range(len(iterable))]
        else:
            colors=[colormap(i) for i in np.linspace(0,1,len(iterable))]
    elif len(iterable) == 1: colors=['C3']
    elif len(iterable) == 2: colors=['C3','C0']
    elif len(iterable) == 3: colors=['C3','C2','C0']
    elif len(iterable) == 4: colors=['C3','C1','C2','C0']
    elif len(iterable) == 5: 
        colors=['C3','C1','C2','C0','C4']
    elif len(iterable) == 6:
        colors=['C5','C3','C1','C2','C0','C4']
    elif len(iterable) == 7:
        colors=['C5','C3','C1','C8','C2','C0','C4']
    elif len(iterable) == 8:
        colors=['C5','C3','C1','C8','C2','C0','C4','k']
    elif len(iterable) == 9:
        colors=['C5','C3','C1','C8','C2','C9','C0','C4','k']
    elif len(iterable) == 10:
        colors=['C5','C3','C1','C6','C8','C2','C9','C0','C4','k']
    else:
        colors=[plt.cm.viridis(i) for i in np.linspace(0,1,len(iterable))]
    out = dict(zip(iterable, colors))
    if explicitcolors:
        explicit = {
            'C0': '#1f77b4', 'C1': '#ff7f0e', 'C2': '#2ca02c', 'C3': '#d62728',
            'C4': '#9467bd', 'C5': '#8c564b', 'C6': '#e377c2', 'C7': '#7f7f7f',
            'C8': '#bcbd22', 'C9': '#17becf',
        }
        if len(iterable) <= 10:
            out = {c: explicit[out[c]] for c in iterable}
    return out

############
### Plotting

def addcolorbarhist(
    f, ax0, data, 
    title=None,
    cmap=plt.cm.viridis, 
    bins=None,
    nbins=201, 
    vmin='default',
    vmax='default',
    cbarleft=1.05,
    cbarwidth=0.025,
    cbarheight=0.5,
    cbarbottom=None,
    cbarhoffset=0,
    histpad=0.005,
    histratio=3,
    labelpad=0.03,
    title_fontsize='large',
    title_alignment=None,
    title_weight='bold',
    log=False,
    histcolor='0.5',
    extend=False,
    orientation='vertical'):
    """
    Notes
    -----
    * All dimensions are in fraction of major axis size
    * cmap must a colormap object (e.g. plt.cm.viridis)
    * data should be of type np.array
    """
    
    ########################
    ### Imports and warnings
    import matplotlib as mpl
    from warnings import warn
    if extend:
        warn('extend=True currently does not maintain colorbar and hist alignment')
    ### Warnings
    #############
    ### Procedure

    ### Get bounds and make bins
    if vmin == 'default':
        vmin = data.min()
    if vmax == 'default':
        vmax = data.max()
        
    if bins is None:
        bins = np.linspace(vmin, vmax, nbins)
    elif type(bins) is np.ndarray:
        pass
    else:
        print(type(bins), bins)
        print(type(nbins, nbins))
        raise Exception('Specify bins as np.ndarray or nbins as int')
    ax0x0, ax0y0, ax0width, ax0height = ax0.get_position().bounds
    
    ### Defaults for colorbar position
    if (cbarbottom is None) and (orientation == 'horizontal'):
        cbarbottom = 1.05
    elif (cbarbottom is None) and (orientation == 'vertical'):
        cbarbottom = (1 - cbarheight) / 2

    ### Set extension
    if extend:
        if (vmin > data.min() and vmax < data.max()):
            extend = 'both'
        elif vmin > data.min(): extend = 'min'
        elif vmax < data.max(): extend = 'max'
        else: extend = 'neither'
    else: extend = 'neither'

    ######### Add colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(bins)

    ### Horizontal orientation
    if orientation in ['horizontal', 'h']:
        # ax1left = ax0x0 + ax0width * (1 - cbarheight) / 2
        ax1left = ax0x0 + ax0width * (1 - cbarheight) / 2 * (1 + cbarhoffset)
        ax1bottom = ax0y0 + ax0height * cbarbottom
        ax1width = cbarheight * ax0width
        ax1height = cbarwidth * ax0height
        
        ax1 = f.add_axes([ax1left, ax1bottom, ax1width, ax1height])

        cb1 = mpl.colorbar.ColorbarBase(
            ax1, cmap=cmap, norm=norm, orientation='horizontal',
            extend=extend)
        ax1.xaxis.set_ticks_position('bottom')

        ##### Add histogram
        ax2left = ax1left# + (cbarwidth + histpad) * ax0width
        ax2bottom = ax1bottom + (cbarwidth + histpad) * ax0height
        ax2width = ax1width #histratio * cbarwidth * ax0width
        ax2height = histratio * cbarwidth * ax0height

        ax2 = f.add_axes([ax2left, ax2bottom, ax2width, ax2height])

        ax2.hist(data, bins=bins, color=histcolor, 
                 log=log, orientation='vertical')
        ax2.set_xlim(vmin, vmax)
        ax2.axis('off')

        if title is not None:
            if title_alignment is None:
                title_alignment = 'bottom center'

            xy = {
                'bottom': (0.5, -labelpad),
                'top': (0.5, 1+labelpad+histpad+histratio),
                'gap right': (1+labelpad, 1 + histpad/2),
                'gap left': (-labelpad, 1 + histpad/2),
                'both right': (1+labelpad, (1+histpad+histratio)/2),
                'both left': (-labelpad, (1+histpad+histratio)/2),
            }
            xy['bottom center'], xy['top center'] = xy['bottom'], xy['top']
            xy['center bottom'], xy['center top'] = xy['bottom'], xy['top']
            
            va = {
                'bottom': 'top',
                'top': 'bottom',
                'gap right': 'center',
                'gap left': 'center',
                'both right': 'center',
                'both left': 'center',
            }
            va['bottom center'], va['top center'] = va['bottom'], va['top']
            va['center bottom'], va['center top'] = va['bottom'], va['top']
            
            ha = {
                'bottom': 'center',
                'top': 'center',
                'gap right': 'left',
                'gap left': 'right',
                'both right': 'left',
                'both left': 'right'
            }
            ha['bottom center'], ha['top center'] = ha['bottom'], ha['top']
            ha['center bottom'], ha['center top'] = ha['bottom'], ha['top']

            ax1.annotate(
                title, fontsize=title_fontsize, weight=title_weight,
                xycoords='axes fraction',
                va=va[title_alignment], xy=xy[title_alignment],
                ha=ha[title_alignment])
    
    ### Vertical orientation
    elif orientation in ['vertical', 'vert', 'v', None]:
        ax1left = ax0width + ax0x0 + (ax0width * (cbarleft - 1))
        ax1bottom = (1 - cbarheight * ax0height) / 2
        ax1width = cbarwidth * ax0width
        ax1height = cbarheight * ax0height

        ax1 = f.add_axes([ax1left, ax1bottom, ax1width, ax1height])

        cb1 = mpl.colorbar.ColorbarBase(
            ax1, cmap=cmap, norm=norm, orientation='vertical',
            extend=extend)
        ax1.yaxis.set_ticks_position('left')

        ##### Add histogram
        ax2left = ax1left + (cbarwidth + histpad) * ax0width
        ax2bottom = ax1bottom
        ax2width = histratio * cbarwidth * ax0width
        ax2height = ax1height

        ax2 = f.add_axes([ax2left, ax2bottom, ax2width, ax2height])

        ax2.hist(data, bins=bins, color=histcolor, 
                 log=log, orientation='horizontal')
        ax2.set_ylim(vmin, vmax)
        ax2.axis('off')

        if title is not None:
            ## 'both center': align to center of cbar + hist
            ## 'cbar center': align to center of cbar
            ## 'cbar left': align to left of cbar
            if title_alignment is None:
                title_alignment = 'gap center'

            xy = {
                'both center': ((1 + histpad + histratio)/2, 1+labelpad),
                'cbar center': (0.5, 1+labelpad),
                'cbar left': (0,1+labelpad),
                'hist right': (histratio + histpad + 1, 1+labelpad),
                'gap center': (1 + histpad/2, 1+labelpad),
            }
            horizontalalignment = {
                'both center': 'center',
                'cbar center': 'center',
                'cbar left': 'left',
                'hist right': 'right',
                'gap center': 'center',
            }

            ax1.annotate(
                title, fontsize=title_fontsize, weight=title_weight,
                xycoords='axes fraction',
                verticalalignment='bottom', xy=xy[title_alignment],
                horizontalalignment=horizontalalignment[title_alignment])

    ### Return axes
    return ax1, ax2


def plot2dhistarray(xdata, ydata, logcolor=True, bins=None,
    figsize=(5,7), gridspec_kw=None, sidehistcolor='0.5'):
    """
    Inputs
    ------
    bins: int (to use np.linstpace(datamin, datamax, 101) for both)
        or ((array of x bins), (array of y bins))
        or ((number of x bins), (number of y bins))

    Returns
    -------
    (f, ax): tuple of figure object and axes object
        * ax[(0,0)]: Upper x-axis histogram
        * ax[(1,0)]: Main 2d histogram
        * ax[(1,1)]: Right y-axis histogram
    """
    ### Format inputs
    if type(bins) == int:
        bins = [np.linspace(min(xdata), max(xdata), bins), 
                np.linspace(min(ydata), max(ydata), bins)]
    elif type(bins) == tuple:
        if (type(bins[0]) == int) and (type(bins[1]) == int):
            bins = [np.linspace(min(xdata), max(xdata), bins[0]), 
                    np.linspace(min(ydata), max(ydata), bins[1])]
        elif (type(bins[0] == np.ndarray) and (type(bins[1]) == np.ndarray)):
            pass
    elif type(bins) == np.ndarray:
        bins = [bins, bins]
    elif bins == None:
        bins = [np.linspace(min(xdata), max(xdata), 101), 
                np.linspace(min(ydata), max(ydata), 101)]

    if gridspec_kw == None:
        gridspec_kw = {'height_ratios': [1,6], 'width_ratios': [6,1], 
                       'hspace':0.02, 'wspace': 0.02}
    ### Procedure
    f, ax = plt.subplots(
        2, 2, figsize=figsize, sharex='col', sharey='row', #dpi=300,
        gridspec_kw=gridspec_kw)

    ### Add horizontal histogram to top
    ax[(0,0)].hist(xdata, bins=bins[0], color=sidehistcolor)
    ax[(0,0)].axis('off')

    _,_,_,im = ax[(1,0)].hist2d(
        xdata, ydata, bins=bins, norm=mpl.colors.LogNorm())

    ### Add vertical histogram to right
    ax[(1,1)].hist(ydata, bins=bins[1], color=sidehistcolor, orientation='horizontal')
    ax[(1,1)].axis('off')

    ### Turn off the spare upper left axis
    ax[(0,1)].axis('off')

    ### Add bottom colorbar
    cb = plt.colorbar(im, ax=ax[(1,0)], orientation='horizontal', fraction=0.15)
    cb.set_label('Counts', fontsize='x-large')
    ### Shrink the right histogram in a dumb way
    cb = plt.colorbar(im, ax=ax[(1,1)], orientation='horizontal', fraction=0.15)
    cb.remove()

    return f, ax


def plotquarthist(
    ax, dfplot,
    histpad=0.1, quartpad=-0.1, pad=None,
    squeeze=0.7, 
    plothist=True,
    number_of_bins=101,
    alpha=0.6,
    bootstrap=5000,
    direction='right',
    x_locations=None,
    hist_range=None,
    format_axes=True,
    flierprops=None,
    whiskerprops=None,
    medianfacecolor='none',
    medianedgecolor='k',
    medianmarker='o',
    mediansize=15,
    cicolor='0.75',
    ciwidth=3, 
    histcolor=None,
    density=False,
    spines=['left', 'bottom'],
    xticklabelrotation=0,
    ):
    """
    Inputs
    ------

    Notes
    -----
    * x-axis values should be the column labels
    * pad, if not None, overwrites histpad and quartpad
    """
    ### Interpret inputs
    if flierprops == None:
        flierprops={'markersize': 2, 'markerfacecolor': 'none',
                    'markeredgewidth': 0.25, 'markeredgecolor': '0.5'}

    if hist_range == None:
        hist_range = (dfplot.min().min(), dfplot.max().max())
    else:
        assert (len(hist_range)==2 or type(hist_range) in [float, int]) 

    labels = list(dfplot.columns)
    data_sets = [dfplot[label].dropna().values for label in labels]

    if x_locations == None:
        x_locations = dfplot.columns.values
        if any([type(col) == str for col in x_locations]):
            x_locations = range(len(x_locations))

    if (pad != None) and (direction == 'right'):
        histpad, quartpad = pad, -pad
    elif (pad != None) and (direction == 'left'):
        histpad, quartpad = -pad, pad

    ###### Some shared quantities between quarts and hists
    ### Minimum distance between x points in units of x scale
    xscale = min(np.diff(x_locations))

    ###### Histograms - data
    if plothist:
        ### Compute histograms
        binned_data_sets = [
            np.histogram(d, range=hist_range, bins=number_of_bins)[0]
            for d in data_sets]

        ###### WITHOUT normalization
        if density in [False, None]:
            ### Number of counts in most-filled bin --> sets x spacing of hists
            spacing = np.max(binned_data_sets)

            ### Scaled bins to xscale
            scaled_data_sets = [
                binned_data / spacing * squeeze
                for binned_data in binned_data_sets]
        ###### WITH normalization
        else:
            # ### Normalization factor
            # normfactor = [binned_data.sum() for binned_data in binned_data_sets]

            # ### Normalize bin filling to sum to 1
            # normed_data_sets = [
            #     binned_data_sets[i] / normfactor[i]
            #     for i in range(len(binned_data_sets))]

            # ### Scale them back up to fill the space

            ### Number of counts in most-filled bin in each col --> x spacing
            spacing = [binned_data.max() for binned_data in binned_data_sets]

            ### Scaled bins to xscale
            scaled_data_sets = [
                binned_data_sets[i] / spacing[i] * squeeze
                for i in range(len(binned_data_sets))]

        ### Set bin edges
        bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
        # centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]   ### WRONG!!!
        centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[1:]
        heights = np.diff(bin_edges)

    ###### Quartlines
    ### Calculate medians and confidence interals
    medians, cilos, cihis = [], [], []
    for i in range(len(data_sets)):
        medianstats = mpl.cbook.boxplot_stats(data_sets[i], bootstrap=bootstrap)[0]
        medians.append(medianstats['med'])
        cilos.append(medianstats['cilo'])
        cihis.append(medianstats['cihi'])

    ###### Combo plot
    if plothist:
        ### Histograms
        assert len(x_locations) == len(scaled_data_sets), "mismatched axes"
        for i in range(len(data_sets)):
            ### Set bar color
            if type(histcolor) == list: 
                c = histcolor[i]
            elif type(histcolor) == dict:
                c = histcolor[labels[i]]
            else: 
                c = histcolor
            ### Plot bars
            if direction == 'right':
                ## Bar z ordering
                zorders = list(range(len(data_sets)))[::-1] # Reversed for correct overlap
                lefts = x_locations[i] + histpad * xscale
                ax.barh(y=centers, width=scaled_data_sets[i], height=heights,
                        left=lefts, alpha=alpha, zorder=zorders[i], color=c)
            elif direction == 'left':
                zorders = list(range(len(data_sets)))
                lefts = x_locations[i] - scaled_data_sets[i] + histpad * xscale
                ax.barh(y=centers, width=scaled_data_sets[i], height=heights,
                        left=lefts, alpha=alpha, zorder=zorders[i], color=c)

    ### Quartlines
    ## Whiskers and outliers
    ax.boxplot(
        data_sets, 
        positions=(x_locations + quartpad * xscale),
        showcaps=False, showbox=False,
        medianprops={'color': 'none', 'linewidth': 0}, 
        whiskerprops=whiskerprops,
        flierprops=flierprops,
        zorder=4998
    )
    ## Medians
    ax.scatter(x_locations + quartpad * xscale, 
               medians,
               zorder=5000, marker=medianmarker, s=mediansize,
               c=medianfacecolor, edgecolor=medianedgecolor, 
    )
    ## Botstrapped 95% confidence intervals for median
    for i in range(len(data_sets)):
        ### Set median range color
        if type(cicolor) == list: c = cicolor[i]
        else: c = cicolor
        ### Plot median bars
        ax.plot(np.array([x_locations[i], x_locations[i]]) + (quartpad * xscale), 
                [cilos[i], cihis[i]],
                zorder=4999, solid_capstyle='butt',
                lw=ciwidth, c=c, 
    )

    ### Axis formatting
    if format_axes:
        ax.set_xticks(x_locations)
        # ax.set_xticklabels(labels, rotation=60) # old version
        if (xticklabelrotation == 0) or (xticklabelrotation >= 60):
            ax.set_xticklabels(labels, rotation=xticklabelrotation)
        else:
            ax.set_xticklabels(labels, rotation=xticklabelrotation, 
                ha='right', rotation_mode='anchor')
        if plothist and (direction == 'right'):
            ax.set_xlim(x_locations[0] - 5 * abs(quartpad) * xscale, 
                        (x_locations[-1] + max(scaled_data_sets[-1])
                         + 1.5 * abs(histpad) * xscale)
                        )
        elif plothist and (direction == 'left'):
            ## Provide some extra padding on the left
            ax.set_xlim((x_locations[0] - max(scaled_data_sets[0]) 
                         - 1.5 * abs(histpad) * xscale), 
                        x_locations[-1] + 3 * abs(quartpad) * xscale
                        )
        ax.tick_params(axis='both', direction='out', top=False, right=False)
        for which in ['left', 'right', 'bottom', 'top']:
            ax.spines[which].set_visible(False)
        if len(spines) >= 1:
            for which in spines:
                ax.spines[which].set_visible(True)


def plotquartiles(
    dfframe=None, ax=None, 
    innerband=['25%','75%'],
    outerband=['2.5%','97.5%'],
    inner=True, outer=True, minmax=True, median=True,
    color='C0', alpha=0.2, labels=True, dfdescribe=None,
    lines=None, lws=None, linestyles=None):
    """
    """
    ### Input formatting
    if lines is None:
        lines = []
    percentiles = innerband + ['50%'] + outerband + lines
    percentiles = sorted(list(set(percentiles)))
    if lws is None:
        lws = dict(zip(lines, [1.5]*len(lines)))
    if linestyles is None:
        linestyles = dict(zip(lines, ['-']*len(lines)))
    fractions = [float(pct[:pct.find('%')])*.01 for pct in percentiles]
    if dfdescribe is None:
        dfdescribe = dfframe.describe(percentiles=fractions)
    if labels is True:
        labels = {
            'outer': 'central {:.0f}%'.format((fractions[-1]-fractions[-0])*100),
            'inner': 'central {:.0f}%'.format((fractions[-2]-fractions[1])*100),
            'median': 'median',
            'min': 'min/max',
            'max': '_nolabel_',
            'minmax': 'all nodes',
        }
    elif labels is False:
        labels = dict(zip(['outer','inner','median','min','max','minmax'],
                          ['_nolabel_']*6))
    else:
        labels = labels

    ### Inner band
    if inner is True:
        ax.fill_between(
            dfdescribe.columns.values,
            dfdescribe.loc[innerband[0]].values,
            dfdescribe.loc[innerband[1]].values,
            color=color, lw=0, alpha=alpha*2, 
            label=labels.get('inner','_nolabel_'),
        )
    ### Outer band
    if outer is True:
        ax.fill_between(
            dfdescribe.columns.values,
            dfdescribe.loc[outerband[0]].values,
            dfdescribe.loc[outerband[1]].values,
            color=color, lw=0, alpha=alpha, 
            label=labels.get('outer','_nolabel_'),
        )
    ### Median
    if median is True:
        ax.plot(
            dfdescribe.columns.values, dfdescribe.loc['50%'].values,
            color=color, label=labels.get('median','_nolabel_'))

    ### Min/Max band
    if minmax is True:
        ax.fill_between(
            dfdescribe.columns.values,
            dfdescribe.loc['min'].values,
            dfdescribe.loc['max'].values,
            color=color, lw=0, alpha=alpha/2, 
            label=labels.get('minmax','_nolabel_'),
        )

    ### Extra lines
    for line in lines:
        ax.plot(
            dfdescribe.columns.values, dfdescribe.loc[line].values,
            color=color, label=labels.get(line,'_nolabel_'), 
            lw=lws[line], ls=linestyles[line])


###### fill_between
def subplotpercentiles(ax, dfplot, datacolumn, tracecolumn, subplotcolumn=None,
    colordict=None, traceorder=None, ylimits=None, xdivs=2, 
    plottype='line', fillto=0, ascending=True, **kwargs
    ):
    """
    """
    ### Get categorical variables
    if traceorder is None:
        tracevals = dfplot[tracecolumn].unique()
        tracevals.sort()
    else:
        tracevals = traceorder
        
    ### Get colors
    if colordict is None:
        colors = ['C{}'.format(i%10) for i in range(len(tracevals))]
        colordict = dict(zip(tracevals, colors))
    elif type(colordict) == list:
        colordict = dict(zip(tracevals, colordict))
    elif colordict in ['order', 'ordered', 'rainbow', 'sort']:
        colordict = rainbowmapper(tracevals)

    ### Cycle through tracevals
    for i, traceval in enumerate(tracevals):
        ### Get the data
        dfpanel = dfplot.loc[
            (dfplot[tracecolumn] == traceval)].copy()
        x = np.linspace(0,100,len(dfpanel))
        y = dfpanel[datacolumn].sort_values(ascending=ascending)

        ### Plot it
        if plottype in ['line', 'trace', 'plot']:
            ax.plot(x, y, label=traceval,
                    c=colordict[traceval], **kwargs)
        elif plottype in ['area', 'fillbetween', 'fill_between']:
            if i == 0:
                ax.fill_between(x, y, fillto, label=traceval,
                        color=colordict[traceval], **kwargs)
            else:
                y2 = dfplot.loc[
                    (dfplot[tracecolumn]==tracevals[i-1]), datacolumn
                ].sort_values(ascending=ascending)
                ax.fill_between(x, y, y2, label=traceval,
                        color=colordict[traceval], **kwargs)
                
        ax.set_xlim(0,100)
        ax.xaxis.set_minor_locator(AutoMinorLocator(xdivs))
        ax.tick_params(axis='both', which='major', direction='in', 
                       top=True, right=True, length=4)
        ax.tick_params(axis='x', which='minor', direction='in', 
                       top=True, length=3, color='k')
    
    [line.set_zorder(10) for line in ax.lines] # lines on top of axes
    ax.set_ylabel(datacolumn, fontsize='x-large', weight='bold')

    ### Set y limits based on ymin, ymax
    if ylimits is not None:
        assert len(ylimits) == 2, 'len(ylimits) must be 2 but is {}'.format(len(ylimits))
        assert type(ylimits[0]) == type(ylimits[1])
        if type(ylimits[0]) == str:
            ymin = float(ylimits[0].replace('%','')) * 0.01
            ymax = float(ylimits[1].replace('%','')) * 0.01
            describe = dfplot[datacolumn].describe(percentiles=[ymin, ymax])
            ax.set_ylim(describe[ylimits[0]], describe[ylimits[1]])
        else:
            ax.set_ylim(ylimits[0], ylimits[1])


def plotpercentiles(dfplot, datacolumn, tracecolumn, subplotcolumn=None,
    colordict=None, figsize=None, unsubplot=False,
    traceorder=None, subplotorder=None, ylimits=None, 
    xdivs=2, dpi=None, 
    ):
    """
    """
    ### Get categorical variables
    if traceorder is None:
        tracevals = dfplot[tracecolumn].unique()
        tracevals.sort()
    else:
        tracevals = traceorder

    if subplotcolumn is None:
        ncols = 1
        subplotvals = None
    elif subplotorder is None:
        subplotvals = dfplot[subplotcolumn].unique()
        subplotvals.sort()
        ncols = len(subplotvals)
    else:
        ncols = len(subplotorder) + int(unsubplot)
        subplotvals = subplotorder
        
    if figsize is None:
        figsize = (ncols*2, 3)

    ### Check for overload
    if ncols >= 50:
        raise Exception('Whoooah too many columns: {}'.format(ncols))
    ### Get colors
    if colordict is None:
        colors = ['C{}'.format(i%10) for i in range(len(tracevals))]
        colordict = dict(zip(tracevals, colors))
    elif type(colordict) == list:
        colordict = dict(zip(tracevals, colordict))
    elif colordict in ['order', 'ordered', 'rainbow', 'sort']:
        colordict = rainbowmapper(tracevals)

    ### Plot it
    f, ax = plt.subplots(ncols=ncols, sharey=True, figsize=figsize, dpi=dpi)
    for i in range(len(tracevals)):
        if ncols > 1:
            for j in range(len(subplotvals)):
                dfpanel = dfplot.loc[
                    (dfplot[subplotcolumn] == subplotvals[j])
                    & (dfplot[tracecolumn] == tracevals[i])].copy()
                x = np.linspace(0,100,len(dfpanel))
                y = dfpanel[datacolumn].sort_values()
                ax[j].plot(x, y, label=tracevals[i],
                           c=colordict[tracevals[i]])
                ax[j].set_xlim(0,100)
                ax[j].xaxis.set_minor_locator(AutoMinorLocator(xdivs))
                ax[j].set_title(subplotvals[j], weight='bold', fontname='Arial')
                ax[j].tick_params(
                    axis='both', which='major', direction='in', 
                    top=True, right=True, length=4)
                ax[j].tick_params(
                    axis='x', which='minor', direction='in', 
                    top=True, length=3, color='k')
                [line.set_zorder(10) for line in ax[j].lines] # lines on top of axes
            
            ### Plot the all-data subplot
            if unsubplot:
                dfpanel = dfplot.loc[
                    (dfplot[tracecolumn] == tracevals[i])].copy()
                x = np.linspace(0,100,len(dfpanel))
                y = dfpanel[datacolumn].sort_values()
                ax[-1].plot(x, y, label=tracevals[i],
                           c=colordict[tracevals[i]])
                ax[-1].set_xlim(0,100)
                ax[-1].xaxis.set_minor_locator(AutoMinorLocator(xdivs))
                ax[-1].set_title('All', weight='bold', fontname='Arial')
                ax[-1].tick_params(
                    axis='both', which='major', direction='in', 
                    top=True, right=True, length=4)
                ax[-1].tick_params(
                    axis='x', which='minor', direction='in', 
                    top=True, length=3, color='k')
                [line.set_zorder(10) for line in ax[-1].lines] # lines on top of axes
                
            ### Label the leftmost axis
            ax[0].set_ylabel(datacolumn, fontsize='x-large', weight='bold')
        else:
            dfpanel = dfplot.loc[
                (dfplot[tracecolumn] == tracevals[i])].copy()
            x = np.linspace(0,100,len(dfpanel))
            y = dfpanel[datacolumn].sort_values()
            ax.plot(x, y, label=tracevals[i],
                       c=colordict[tracevals[i]])
            ax.set_xlim(0,100)
            ax.xaxis.set_minor_locator(AutoMinorLocator(xdivs))
            ax.tick_params(
                axis='both', which='major', direction='in', 
                top=True, right=True, length=4)
            ax.tick_params(
                axis='x', which='minor', direction='in', 
                top=True, length=3, color='k')
            [line.set_zorder(10) for line in ax.lines] # lines on top of axes
            ax.set_ylabel(datacolumn, fontsize='x-large', weight='bold')

    ### Set y limits based on ymin, ymax
    if ylimits is not None:
        assert len(ylimits) == 2, 'len(ylimits) must be 2 but is {}'.format(len(ylimits))
        assert type(ylimits[0]) == type(ylimits[1])
        if type(ylimits[0]) == str:
            ymin = float(ylimits[0].replace('%','')) * 0.01
            ymax = float(ylimits[1].replace('%','')) * 0.01
            describe = dfplot[datacolumn].describe(percentiles=[ymin, ymax])
            plt.ylim(describe[ylimits[0]], describe[ylimits[1]])
        else:
            plt.ylim(ylimits[0], ylimits[1])

    # if xlabel:
    #     ### add big axis, hide frame, ticks, and labels
    #     f.add_subplot(111, frameon=False)
    #     plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    #     plt.xlabel('Percentile [%]', fontsize='x-large', weight='bold', fontname='Arial')

    return f, ax


def get_aea_bounds(dfplot, buffer=1.05, lat_1=29.5, lat_2=45.5):
    """
    Purpose: Get bounds for use in plotusascattermap
    """
    import pyproj
    latlabel, lonlabel = zephyr.toolbox.get_latlonlabels(dfplot)
    
    
    bound = [
        dfplot[lonlabel].min(), dfplot[latlabel].min(),
        dfplot[lonlabel].max(), dfplot[latlabel].max()
    ]
    coords = [(bound[0],bound[1]), (bound[2],bound[1]),
              (bound[2],bound[3]), (bound[0],bound[3]),]

    ### Undo defaults if desired
    if lat_1 is None:
        lat1 = bound[1]
    else:
        lat1 = lat_1
    if lat_2 is None:
        lat2 = bound[3]
    else:
        lat2 = lat_2
    ### Get the other params from the center of the supplied points
    lat_0 = (bound[1]+bound[3])/2
    lon_0 = (bound[0]+bound[2])/2

    pa = pyproj.Proj(
        "+proj=aea +lat_1={} +lat_2={} +lat_0={} +lon_0={}".format(
            lat1, lat2, lat_0, lon_0))
    lon,lat = zip(*coords)
    x,y = pa(lon,lat)
    # cop = {'type':'Polygon','coordinates':[zip(x,y)]}
    width = 2 * max([abs(i) for i in x]) * buffer
    height = 2 * max([abs(i) for i in y]) * buffer
    
    bounds = {
        'lat_1': lat1, 'lat_2': lat2,
        'lon_0': lon_0, 'lat_0': lat_0,
        'width': width, 'height': height,
    }
    return bounds


def draw_screen_poly(poly, m, 
    cvalue=0.5, cmap=plt.cm.viridis, 
    facecolor=None, edgecolor='none',
    alpha=1, ax=None,
    linewidth=None,
    ):
    """
    Notes
    -----
    * 0.15 is a good linewidth to avoid gaps between polygons
    Sources
    -------
    https://stackoverflow.com/questions/12251189/how-to-draw-rectangles-on-a-basemap
    https://basemaptutorial.readthedocs.io/en/latest/shapefile.html
    """
    if facecolor is None:
        facecolor = cmap(cvalue)
    if edgecolor is None:
        edgecolor = facecolor
    if linewidth is None:
        linewidth = 0
    if type(poly) is shapely.geometry.multipolygon.MultiPolygon:
        for subpoly in poly:
            x,y = m(subpoly.exterior.coords.xy[0], subpoly.exterior.coords.xy[1])
            xy = list(zip(x,y))
            drawpoly = mpl.patches.Polygon(
                xy, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
            if ax is None:
                ax = plt.gca()
            ax.add_patch(drawpoly)
    else:
        x,y = m(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1])
        xy = list(zip(x,y))
        drawpoly = mpl.patches.Polygon(
            xy, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
        if ax is None:
            ax = plt.gca()
        ax.add_patch(drawpoly)


def plotusascattermap(
    dfplot, colorcolumn=None, sizecolumn=None, filterdict=None, sort=True, 
    m=None, cmap=None, background='light', latlonlabels=None,
    markersize=None, marker='o', figsize=(10,7.5), dpi=None,
    zrange=None, colors=None, maptype='scatter', contourlevels=100,
    colorbarhist=True, colorbarkwargs={}, colorbartitle=None,
    statelinewidths=0.25, bounds='usa', downloadmap=False,
    markeredgecolor=None, markerlw=None, f=None, ax=None,
    maskwater=False, masklakes=False, maskcanmex=False, buffer=1.05, 
    alpha=1, markerfacecolor='C3', **plotkwargs):
    """
    Inputs
    ------
    * bounds: 'usa', 'CA', or dict with keys in 
    ['lat_1','lat_2','lon_0','lat_0','width','height']

    References
    ----------
    * Map: https://hifld-geoplatform.opendata.arcgis.com/datasets/political-boundaries-area
    * Canada/Mexico maps: https://gadm.org/download_country_v3.html
    """
    from mpl_toolkits.basemap import Basemap

    ### Set the plot parameters based on background
    if cmap == None:
        cmapdict = {'light': plt.cm.coolwarm, 'dark': plt.cm.viridis, 
                    'none': plt.cm.coolwarm,}
        cmap = cmapdict[background]
    edgecolors = {'light': 'k', 'dark': 'white', 'none': 'none'}
    facecolor = {'light': 'white', 'dark': 'k', 'none': 'none'}
    if background not in ['light','dark','none']:
        facecolor[background] = background
        edgecolors[background] = 'white'

    ### Set the map bounds based on input
    if type(bounds) is dict:
        dictbounds = bounds
    elif type(bounds) is list:
        dictbounds = dict(zip(
            ['lat_1','lat_2','lon_0','lat_0','width','height'],
            bounds
        ))
    elif bounds in ['usa', 'USA', 'all', 'default', None]:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -96.5, 'lat_0': 38.5,
                      'width': 4650000, 'height': 2900000}
    elif bounds.lower() in ['ca', 'california', 'caiso','camx','egrid_camx']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -119, 'lat_0': 36,
                      'width': 950000, 'height': 1400000}
    elif bounds.lower() in ['tx', 'texas', 'ercot','erct','egrid_erct']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -100, 'lat_0': 31,
                      'width': 1300000, 'height':1300000}
    elif bounds.lower() in ['pjm']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -81.5, 'lat_0': 39,
                      'width': 1300000, 'height':900000}
    elif bounds.lower() in ['isone', 'new england','newe','egrid_newe']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -70.5, 'lat_0': 44,
                      'width': 600000, 'height':800000}
    elif bounds.lower() in ['nyiso', 'ny', 'new york','nyup','egrid_nyup']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -76, 'lat_0': 43,
                      'width': 700000, 'height':600000}
    elif bounds.lower() in ['miso', 'midwest']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -91, 'lat_0': 39,
                      'width': 2200000, 'height':2400000}
    elif bounds.lower() in ['wecc']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -114, 'lat_0': 40,
                      'width': 1800000, 'height':2100000}
    elif bounds.lower() in ['northeast']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -73.5, 'lat_0': 44,
                      'width': 1100000, 'height':800000}

    elif bounds.lower() in ['aznm','egrid_aznm','southwest']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -110, 'lat_0': 34.5,
                       'width': 1400000, 'height':900000,}
    elif bounds.lower() in ['florida','fl','frcc','egrid_frcc']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -84, 'lat_0': 28,
                      'width': 900000, 'height':800000}
    elif bounds.lower() in ['mroe','egrid_mroe','wi','wisconsin']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -89, 'lat_0': 45,
                      'width': 650000, 'height':650000}
    elif bounds.lower() in ['mrow','egrid_mrow']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -98, 'lat_0': 44.5,
                      'width': 1500000, 'height':1100000}
    elif bounds.lower() in ['long island','longisland','nyli','egrid_nyli',
                            'nycw','egrid_nycw']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -73, 'lat_0': 40.9,
                      'width': 220000, 'height':130000}
    elif bounds.lower() in ['nwpp','egrid_nwpp','northwest']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -114, 'lat_0': 42,
                      'width': 1800000, 'height':1800000,}
    elif bounds.lower() in ['rfce','egrid_rfce']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -77., 'lat_0': 39.75,
                      'width': 700000, 'height':650000}
    elif bounds.lower() in ['rfcm','egrid_rfcm','michigan','mi']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -84.5, 'lat_0': 43.7,
                      'width': 400000, 'height':550000}
    elif bounds.lower() in ['rfcw','egrid_rfcw']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -83.8, 'lat_0': 42,
                      'width': 1200000, 'height':1300000}
    elif bounds.lower() in ['rmpa','egrid_rmpa']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -106, 'lat_0': 41,
                      'width': 900000, 'height':1000000}
    elif bounds.lower() in['spno','egrid_spno','nebraska','ne']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -97.5, 'lat_0': 38.5,
                      'width': 900000, 'height':550000}
    elif bounds.lower() in ['spso','egrid_spso']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -97.3, 'lat_0': 33.5,
                      'width': 1550000, 'height':1000000}
    elif bounds.lower() in ['srmv','egrid_srmv']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -93, 'lat_0': 33,
                      'width': 900000, 'height':1000000}
    elif bounds.lower() in ['srmw','egrid_srmw']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -91, 'lat_0': 39.,
                      'width': 850000, 'height':800000}
    elif bounds.lower() in ['srso','egrid_srso']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -86.5, 'lat_0': 32.5,
                      'width': 1200000, 'height':700000}
    elif bounds.lower() in ['srtv','egrid_srtv']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -86.3, 'lat_0': 35.5,
                      'width': 1000000, 'height':900000}
    elif bounds.lower() in ['srvc','egrid_srvc']:
        dictbounds = {'lat_1': 29.5, 'lat_2': 45.5,
                      'lon_0': -79.5, 'lat_0': 35.8,
                      'width': 800000, 'height':950000}
    elif bounds in ['data','bounds','zoom','buffer']:
        dictbounds = get_aea_bounds(dfplot, buffer=buffer)
        
    lat_1, lat_2 = dictbounds['lat_1'], dictbounds['lat_2']
    lon_0, lat_0 = dictbounds['lon_0'], dictbounds['lat_0']
    width, height = dictbounds['width'], dictbounds['height']
    
    ### Load basemap if necessary
    if m is None:
        m_in = Basemap(resolution=None,
            projection='aea',
            lat_1=lat_1, lat_2=lat_2, lon_0=lon_0, lat_0=lat_0,
            width=width, height=height)
        ### Set the datapath
        mappath = os.path.join(
            datapath, 'Maps/HIFLD/Political_Boundaries_Area/Political_Boundaries_Area')

        ###### Download the map file if necessary
        if (not os.path.exists(mappath+'.shp')) and (downloadmap == False):
            raise Exception("No file at {}; try setting downloadmap=True".format(mappath))
        if (not os.path.exists(mappath+'.shp')) and (downloadmap == True):
            import urllib.request, zipfile
            ### Download it
            url = ('https://opendata.arcgis.com/datasets/bee7adfd918e4393995f64e155a1bbdf_0.zip?'
                   'outSR=%7B%22wkid%22%3A102100%2C%22latestWkid%22%3A3857%7D')
            os.makedirs(datapath+'Maps/HIFLD/Political_Boundaries_Area/', exist_ok=True)
            urllib.request.urlretrieve(url, datapath+'Maps/HIFLD/Political_Boundaries_Area.zip')
            ### Unzip it
            zip_ref = zipfile.ZipFile(datapath+'Maps/HIFLD/Political_Boundaries_Area.zip', 'r')
            zip_ref.extractall(datapath+'Maps/HIFLD/Political_Boundaries_Area/')
            zip_ref.close()

        ### Read the state outlines
        m_in.readshapefile(mappath, 'States', drawbounds=False)

    else:
        m_in = copy.copy(m)

    ### Get the state metadata
    dfbasemap = pd.DataFrame(m_in.States_info)
    statenames = list(dfbasemap[dfbasemap.COUNTRY == 'USA']['NAME'].unique())
    for i in ["water/agua/d'eau", "Puerto Rico", "United States Virgin Islands",
              "Navassa Island"]:
        statenames.remove(i)
    
    ### Use passed axis, or draw new plot and axis
    if (f is None) or (ax is None):
        ### Draw the plot
        plt.close()
        f, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.axis('off')
    else:
        pass

    ### Plot the US states
    patches = []
    for info, shape in zip(m_in.States_info, m_in.States):
        if (info['COUNTRY'] == 'USA') and (info['NAME'] in statenames):
            patches.append(mpl.patches.Polygon(np.array(shape)))

    # ax.add_collection(mpl.collections.PatchCollection(
    #     patches, linewidths=statelinewidths, 
    #     edgecolors=edgecolors[background], facecolor=facecolor[background]))
    ### Plot separately, to keep edges on top
    if background != 'light':
        ax.add_collection(mpl.collections.PatchCollection(
            patches, linewidths=statelinewidths, 
            edgecolors='none', facecolor=facecolor[background], zorder=0))
    ax.add_collection(mpl.collections.PatchCollection(
        patches, linewidths=statelinewidths, 
        edgecolors=edgecolors[background], facecolor='none', zorder=200000))

    ###### Mask Canada and Mexico if requested
    if maskcanmex is True:
        infiles = {'canada': datapath+'Maps/GADM/gadm36_CAN_shp/gadm36_CAN_0',
                   'mexico': datapath+'Maps/GADM/gadm36_MEX_shp/gadm36_MEX_0'}
        urls = {'canada': 'https://biogeo.ucdavis.edu/data/gadm3.6/shp/gadm36_CAN_shp.zip',
                'mexico': 'https://biogeo.ucdavis.edu/data/gadm3.6/shp/gadm36_MEX_shp.zip'}
        patches = []
        for country in infiles:
            ### Make sure they exist, download if not
            if (not os.path.exists(infiles[country]) and (downloadmap == True)):
                import urllib.request, zipfile
                name = 'gadm36_{}_shp'.format(country[:3].upper())
                os.makedirs(datapath+'Maps/GADM/{}/'.format(name), exist_ok=True)
                urllib.request.urlretrieve(urls[country], datapath+'Maps/GADM/{}.zip'.format(name))
                ### Unzip it
                zip_ref = zipfile.ZipFile(datapath+'Maps/GADM/{}.zip'.format(name), 'r')
                zip_ref.extractall(datapath+'Maps/GADM/{}/'.format(name))
                zip_ref.close()
            ### Read them in
            m_in.readshapefile(infiles[country], country, drawbounds=False)
            ### Plot them
            for shape in getattr(m_in, country):
                patches.append(mpl.patches.Polygon(np.array(shape)))
            ax.add_collection(mpl.collections.PatchCollection(
                patches, linewidths=statelinewidths, zorder=10000,
                edgecolors=facecolor[background], facecolor=facecolor[background]))

    ### Filter data to plot
    dfmap = dfplot.copy()
    if filterdict is not None:
        for key in filterdict:
            dfmap = dfmap.loc[dfmap[key] == filterdict[key]]

    ### Order points and get zrange
    if colorcolumn is not None:
        if sort in [True,'ascending','ascend','positive','normal']:
            dfmap = dfmap.sort_values(colorcolumn).copy()
        elif sort in ['descending','reverse','descend','negative']:
            dfmap = dfmap.sort_values(colorcolumn, ascending=False).copy()
        else:
            pass

        datamin, datamax = dfmap[colorcolumn].min(), dfmap[colorcolumn].max()
        if zrange is None:
            zrangeplot = [datamin, datamax]
        else:
            assert len(zrange) == 2, "len(zrange) must be 2 but is {}".format(len(zrange))
            assert type(zrange[0]) == type(zrange[1])
            if type(zrange[0]) == str:
                assert (zrange[0].endswith('%') and zrange[1].endswith('%')), "zrange != %"
                zmin = float(zrange[0].replace('%','')) * 0.01
                zmax = float(zrange[1].replace('%','')) * 0.01
                describe = dfmap[colorcolumn].describe(percentiles=[zmin, zmax])
                zrangeplot = [describe[zrange[0]], describe[zrange[1]]]
            else:
                zrangeplot = [zrange[0], zrange[1]]

    ### Get lat, lon labels
    latlabel, lonlabel = zephyr.toolbox.get_latlonlabels(dfmap, latlonlabels=latlonlabels)
    lons = dfmap[lonlabel].values
    lats = dfmap[latlabel].values

    ### Set colors and sizes
    if colorcolumn is not None:
        if colors is None:
            colordata = dfmap[colorcolumn].values
        elif type(colors) == dict:
            colordata = dfmap[colorcolumn].map(lambda x: colors[x]).values
    elif colorcolumn is None:
        if colors is None:
            colordata = markerfacecolor
        else:
            colordata = colors

    if (sizecolumn is None) and (markersize is None):
        size = 2
    elif markersize is None:
        size = dfmap[sizecolumn].values
    elif sizecolumn is None:
        size = markersize
    else:
        size = dfmap[sizecolumn].values * markersize

    x, y = m_in(lons, lats)

    if (maptype == 'scatter') and (colorcolumn is not None):
        m_in.scatter(
            x, y, marker=marker, c=colordata, s=size, cmap=cmap,
            vmin=zrangeplot[0], vmax=zrangeplot[1], alpha=alpha,
            edgecolor=markeredgecolor, lw=markerlw, ax=ax, **plotkwargs)
    elif (maptype == 'scatter') and (colorcolumn is None):
        m_in.scatter(x, y, marker=marker, c=colordata, s=size,
            edgecolor=markeredgecolor, lw=markerlw, alpha=alpha, ax=ax)
    elif maptype == 'contour':
        m_in.contour(
            x, y, colordata, tri=True, cmap=cmap,
            levels=np.linspace(zrangeplot[0],zrangeplot[1],contourlevels),
            extend='both', ax=ax, **plotkwargs)
    elif maptype == 'contourf':
        m_in.contourf(
            x, y, colordata, tri=True, cmap=cmap,
            levels=np.linspace(zrangeplot[0],zrangeplot[1],contourlevels),
            extend='both', ax=ax, **plotkwargs)
    elif maptype in ['poly','polygon','vector']:
        for index in dfmap.index:
            # if type(colors) == dict:
            #     cvalue = colors[dfmap.loc[index,colorcolumn]]
            # else:
            #     if colorcolumn is not None:
            #         cvalue = (
            #             (dfmap.loc[index,colorcolumn] - zrangeplot[0])
            #             / (zrangeplot[1] - zrangeplot[0])
            #             #* (datamax - datamin) + datamin
            #         )
            #     else:
            #         cvalue = markerfacecolor
            if colorcolumn is not None:
                cvalue = (
                    (dfmap.loc[index,colorcolumn] - zrangeplot[0])
                    / (zrangeplot[1] - zrangeplot[0])
                    #* (datamax - datamin) + datamin
                )
            else:
                cvalue = markerfacecolor
            draw_screen_poly(
                dfmap.loc[index,'geometry'], m_in, ax=ax, 
                cvalue=cvalue, alpha=alpha,
                facecolor=None if colorcolumn is not None else markerfacecolor,
                edgecolor=markeredgecolor,
                # cvalue=dfmap.loc[index,colorcolumn],
                cmap=cmap, linewidth=markerlw,
            )
        ### Plot one point; otherwise basemap won't work
        x,y = m_in([dfmap.loc[index, lonlabel]], [dfmap.loc[index, latlabel]])
        # x,y = m_in([dfmap.loc[0, lonlabel]], [dfmap.loc[0, latlabel]])
        # x,y = m_in([-100],[40])
        m_in.scatter(x, y, color='none', ax=ax)

    ### Mask water if requested
    if maskwater is True:
        m_in.drawlsmask(land_color='none',ocean_color='white',lakes=masklakes,zorder=10000)

    if (colorbarhist == True) and (colorcolumn is not None):
        ### Add legend if categorical
        if type(colors) is dict:
            patchlegend(colors, edgecolor=markeredgecolor, **colorbarkwargs)
        ### Add hist if not categorial
        else:
            cax = addcolorbarhist(
                f=f, ax0=ax, data=colordata, title=colorbartitle, cmap=cmap,
                vmin=zrangeplot[0], vmax=zrangeplot[1],
                # title_fontsize='x-large',
                **colorbarkwargs)
            returncax = True

    return f, ax
    # return f, ((ax, cax) if (returncax and colorbarhist) else ax)


def sparkline(ax, dsplot, endlabels=True,
    endlabelxpad=3, endlabelypad=0,
    xticks='default', minordivisions=None,
    ):
    """
    """
    dsplot.plot(ax=ax)

    ### Format axes
    ax.tick_params(axis='x', which='both', top='off')
    for which in ['left', 'bottom', 'right', 'top']:
        ax.spines[which].set_visible(False)
    ax.yaxis.set_visible(False)
    if xticks == 'default':
        ax.set_xticks([dsplot.index[0], dsplot.index[-1]])
    elif xticks is None:
        pass
    else:
        ax.set_xticks(xticks)

    if minordivisions is not None:
        ax.xaxis.set_minor_locator(AutoMinorLocator(minordivisions))

    ### Annotate the ends
    if endlabels:
        ## Left
        ax.annotate(
            s='{:.1f}'.format(dsplot.iloc[0]), 
            xy=(dsplot.index[0], dsplot.iloc[0]),
            xycoords='data', xytext=(-endlabelxpad,endlabelypad), 
            textcoords='offset points',
            horizontalalignment='right', verticalalignment='center')
        ## Right
        ax.annotate(
            s='{:.1f}'.format(dsplot.iloc[-1]), 
            xy=(dsplot.index[-1], dsplot.iloc[-1]),
            xycoords='data', xytext=(endlabelxpad,endlabelypad), 
            textcoords='offset points',
            horizontalalignment='left', verticalalignment='center')


def plotyearbymonth(dfs, plotcols=None, colors=None, 
    style='fill', lwforline=1, figsize=(12,6), dpi=None,
    normalize=False, alpha=1, f=None, ax=None):
    """
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    def monthifier(x):
        return pd.Timestamp('2001-01-{} {:02}:00'.format(x.day, x.hour))
    
    if (f is None) and (ax is None):
        f,ax=plt.subplots(12,1,figsize=figsize,sharex=True,sharey=True, dpi=dpi)
    else:
        pass

    for i, month in enumerate(months):
        if isinstance(dfs, pd.Series):
            plotcols = str(dfs.name)
            dfs = pd.DataFrame(dfs.rename(plotcols))
                
        if isinstance(dfs, pd.DataFrame):
            if plotcols is None:
                plotcols = dfs.columns.tolist()

            if isinstance(plotcols, str):
                dfplot = dfs.loc['{} {}'.format(month, dfs.index[0].year)][[plotcols]]
                if normalize:
                    dfplot = dfplot / dfplot.max()
                dfplot.index = dfplot.index.map(monthifier)

                if style in ['fill', 'fill_between', 'f']:
                    ax[i].fill_between(
                        dfplot.index, dfplot[plotcols].values, lw=0, alpha=alpha,
                        color=(colors if type(colors) in [str,mpl.colors.ListedColormap] 
                               else ('C0' if colors == None else colors[0])))
                elif style in ['line', 'l']:
                    ax[i].plot(
                        dfplot.index, dfplot[plotcols].values, lw=lwforline, alpha=alpha,
                        color=(colors if type(colors) in [str,mpl.colors.ListedColormap] 
                               else ('C0' if colors == None else colors[0])))
                    
            elif isinstance(plotcols, list):
                if isinstance(colors, str):
                    colors = [colors]*len(plotcols)
                elif colors == None:
                    colors = ['C{}'.format(i%10) for i in range(len(plotcols))]
                for j, plotcol in enumerate(plotcols):
                    dfplot = dfs.loc['{} {}'.format(month, dfs.index[0].year)][[plotcol]]
                    if normalize:
                        dfplot = dfplot / dfplot.max()
                    dfplot.index = dfplot.index.map(monthifier)

                    if style in ['fill', 'fill_between', 'f']:
                        ax[i].fill_between(dfplot.index, dfplot[plotcol].values, 
                                           lw=0, alpha=alpha, color=colors[j], label=plotcol)
                    elif style in ['line', 'l']:
                        ax[i].plot(dfplot.index, dfplot[plotcol].values, 
                                   lw=lwforline, alpha=alpha, color=colors[j], label=plotcol)
                                        
        ax[i].set_ylabel(month, rotation=0, ha='right', va='top')
        for which in ['left', 'right', 'top', 'bottom']:
                     ax[i].spines[which].set_visible(False)
        ax[i].tick_params(left=False,right=False,top=False,bottom=False)
        ax[i].set_yticks([])
        ax[i].set_xticks([])

    ax[0].set_xlim('2001-01-01 00:00', '2001-02-01 00:00')
    if normalize:
        ax[0].set_ylim(0, 1)
    else:
        pass
        # ax[0].set_ylim(0,dfs[plotcols].max())
    
    return f, ax


def add_parasite_axis_converter(
    ax, converter, side='right', converterparams={},
    offset=0.1, tickdirection='out', ylabel=False,
    minor=True,
    ticks=None, ticklabels=None,
    ):
    """
    Add a parasite axis, modeled on
    'https://matplotlib.org/gallery/ticks_and_spines/'
    'multiple_yaxis_with_spines.html'.
    The ticks are taken directily from the primary axis, with
    new values generated by the converter function.

    Inputs
    ------
    ax: axis to parasitize
    converter: function, where first argument takes a float (which
        will be the tick values from the primay axis to copy)
    converterparams: dict of additional keywordparameters to
        pass to converter function
    """
    ### Some cleanup
    offset = 1 + offset if side == 'right' else -offset

    ### Input formatting
    if ticklabels is None:
        ticklabels = ticks

    ### Get parasite axis properties
    if ticks is None:
        oldticks = ax.get_yticks()
        newticks = [converter(oldtick,**converterparams)
                    for oldtick in oldticks]
        ticklabels = newticks
    else:
        newticks = ticks

    oldlim = ax.get_ylim()
    newlim = [converter(oldend,**converterparams)
              for oldend in oldlim]
    
    ### Add the parasite
    par = ax.twinx()
    par.spines[side].set_position(('axes',offset))
    par.set_frame_on(True)
    par.patch.set_visible(False)
    for sp in par.spines.values():
        sp.set_visible(False)
        par.spines[side].set_visible(True)
    par.set_yticks(newticks)
    par.set_yticklabels(ticklabels)
    par.set_ylim(*newlim)
    if minor:
        minorlocator = ax.yaxis.get_minor_locator()
        par.yaxis.set_minor_locator(minorlocator)
    if side == 'left':
        par.yaxis.set_label_position('left')
        par.yaxis.set_ticks_position('left')
    if ylabel:
        par.set_ylabel(ylabel)
    par.tick_params(axis='y', which='both', direction=tickdirection)
    
    return par

def _despine_sub(ax, 
    top=False, right=False, left=True, bottom=True,
    direction='out'):
    """
    """
    if not top: ax.spines['top'].set_visible(False)
    if not right: ax.spines['right'].set_visible(False)
    if not left: ax.spines['left'].set_visible(False)
    if not bottom: ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both',
                   direction=direction, 
                   top=top, right=right, 
                   left=left, bottom=bottom)

def despine(ax=None, 
    top=False, right=False, left=True, bottom=True,
    direction='out'):
    """
    """
    if ax is None:
        ax = plt.gca()
    if type(ax) is np.ndarray:
        for sub in ax:
            if type(sub) is np.ndarray:
                for subsub in sub:
                    _despine_sub(subsub, top, right, left, bottom, direction)
            else:
                _despine_sub(sub, top, right, left, bottom, direction)
    else:
        _despine_sub(ax, top, right, left, bottom, direction)

def patchlegend(colors, edgecolor='none', alpha=1, reverse=False, **kwargs):
    patches = [mpl.patches.Patch(
                   facecolor=colors[i], 
                   edgecolor=edgecolor, 
                   alpha=alpha,
                   label=i)
               for i in colors.keys()]
    if reverse == True:
        leg = plt.legend(handles=patches[::-1], **kwargs)
    else:
        leg = plt.legend(handles=patches, **kwargs)
    return leg

def _differentiate_lines_sub(ax, cycle=10, linestyles=['--',':','-.']):
    """
    """
    if type(linestyles) is str:
        ls = [linestyles,linestyles,linestyles]
    else:
        ls = linestyles
    lines = ax.get_lines()
    if len(lines) > 1*cycle:
        for line in lines[1*cycle:]:
            line.set_linestyle(ls[0])
    if len(lines) > 2*cycle:
        for line in lines[2*cycle:]:
            line.set_linestyle(ls[1])
    if len(lines) > 3*cycle:
        for line in lines[3*cycle:]:
            line.set_linestyle(ls[2])

def differentiate_lines(ax, cycle=10, linestyles=['--',':','-.']):
    """
    Notes
    -----
    * Use before calling the legend constructor
    """
    if ax is None:
        ax = plt.gca()
    if type(ax) is np.ndarray:
        for sub in ax:
            if type(sub) is np.ndarray:
                for subsub in sub:
                    _differentiate_lines_sub(subsub, cycle, linestyles)
            else:
                _differentiate_lines_sub(sub, cycle, linestyles)
    else:
        _differentiate_lines_sub(ax, cycle, linestyles)
