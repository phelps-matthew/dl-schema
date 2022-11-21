from matplotlib import pyplot as plt
import numpy as np
from pandas.plotting._matplotlib.tools import create_subplots as _subplots

_DEBUG = False


def _x_range(data, extra=0.2):
    """Compute the x_range, i.e., the values for which the
    density will be computed. It should be slightly larger than
    the max and min so that the plot actually reaches 0, and
    also has a bit of a tail on both sides.

    data may be np.ndarray of all elements or a range
    """
    try:
        sample_range = np.nanmax(data) - np.nanmin(data)
    except ValueError:
        return []
    if sample_range < 1e-6:
        return [np.nanmin(data), np.nanmax(data)]
    return np.linspace(
        np.nanmin(data) - extra * sample_range,
        np.nanmax(data) + extra * sample_range,
        1000,
    )


def _setup_axis(ax, x_range, col_name=None, grid=False, ylabelsize=None, yrot=None):
    """Setup the axis for the joyplot:
    - add the y label if required (as an ytick)
    - add y grid if required
    - make the background transparent
    - set the xlim according to the x_range
    - hide the xaxis and the spines
    """
    if col_name is not None:
        ax.set_yticks([0])
        ax.set_yticklabels([col_name], fontsize=ylabelsize, rotation=yrot)
        ax.yaxis.grid(grid)
    else:
        ax.yaxis.set_visible(False)
    ax.patch.set_alpha(0)
    if min(x_range) != max(x_range):
        ax.set_xlim([min(x_range), max(x_range)])
    ax.tick_params(axis="both", which="both", length=0, pad=10)
    ax.xaxis.set_visible(_DEBUG)
    ax.set_frame_on(_DEBUG)


def _get_alpha(i, n, start=0.4, end=1.0):
    """Compute alpha value at position i out of n"""
    return start + (1 + i) * (end - start) / n


def _moving_average(a, n=3, zero_padded=False):
    """Moving average of order n.
    If zero padded, returns an array of the same size as
    the input: the values before a[0] are considered to be 0.
    Otherwise, returns an array of length len(a) - n + 1"""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    if zero_padded:
        return ret / n
    else:
        return ret[n - 1 :] / n


def plot_density(
    ax, counts, bin_edges, fill=False, linecolor=None, clip_on=True, **kwargs
):
    """Draw a density plot given an axis, an array of counts and an array
    of bin edges positions where to return the estimated density.
    """
    if len(counts) == 0 or len(bin_edges) == 0:
        return

    # compute center of bins
    x_range = _moving_average(bin_edges, 2)

    if fill:
        ax.fill_between(x_range, 0.0, counts, clip_on=clip_on, antialiased=True, **kwargs)
        # Hack to have a border at the bottom at the fill patch
        # (of the same color of the fill patch)
        # so that the fill reaches the same bottom margin as the edge lines
        # with y value = 0.0
        kw = kwargs
        kw["label"] = None
        ax.plot(x_range, [0.0] * len(x_range), clip_on=clip_on, **kw)

    if linecolor is not None:
        kwargs["color"] = linecolor

    # Remove the legend labels if we are plotting filled curve:
    # we only want one entry per group in the legend (if shown).
    if fill:
        kwargs["label"] = None

    ax.plot(x_range, counts, clip_on=clip_on, **kwargs)


def ridgeplot(
    counts,
    bin_edges,
    x_range,
    grid=False,
    labels=None,
    sublabels=None,
    xlabels=True,
    xlabelsize=None,
    xrot=None,
    ylabelsize=None,
    yrot=None,
    ax=None,
    figsize=None,
    fade=False,
    ylim="max",
    fill=True,
    linecolor=None,
    overlap=1,
    background=None,
    title=None,
    legend=False,
    loc="upper right",
    colormap=None,
    color=None,
    **kwargs
):
    """
    Draw a joyplot from an appropriately nested collection of lists
    using matplotlib and pandas.
    Parameters
    ----------
    data : DataFrame, Series or nested collection
    grid : boolean, default True
        Whether to show axis grid lines
    labels : boolean or list, default True.
        If list, must be the same size of the de
    xlabelsize : int, default None
        If specified changes the x-axis label size
    xrot : float, default None
        rotation of x axis labels
    ylabelsize : int, default None
        If specified changes the y-axis label size
    yrot : float, default None
        rotation of y axis labels
    ax : matplotlib axes object, default None
    figsize : tuple
        The size of the figure to create in inches by default
    kwarg : other plotting keyword arguments
        To be passed to hist/kde plot function
    """

    if fill is True and linecolor is None:
        linecolor = "k"

    if sublabels is None:
        legend = False

    def _get_color(i, num_axes, j, num_subgroups):
        if isinstance(color, list):
            return color[j] if num_subgroups > 1 else color[i]
        elif color is not None:
            return color
        elif isinstance(colormap, list):
            return colormap[j](i / num_axes)
        elif color is None and colormap is None:
            num_cycle_colors = len(plt.rcParams["axes.prop_cycle"].by_key()["color"])
            return plt.rcParams["axes.prop_cycle"].by_key()["color"][
                j % num_cycle_colors
            ]
        else:
            return colormap(i / num_axes)

    ygrid = grid is True or grid == "y" or grid == "both"
    xgrid = grid is True or grid == "x" or grid == "both"

    num_axes = len(counts)

    global_x_range = _x_range(x_range, 0.0)

    # Each plot will have its own axis
    fig, axes = _subplots(
        naxes=num_axes,
        ax=ax,
        squeeze=False,
        sharex=True,
        sharey=False,
        figsize=figsize,
        layout_type="vertical",
    )
    # _axes = _flatten(axes)
    _axes = axes.flatten()

    # mp -- explicitly clear axes, helps multithreading
    for a in _axes:
        a.clear()

    # The legend must be drawn in the last axis if we want it at the bottom.
    if loc in (3, 4, 8) or "lower" in str(loc):
        legend_axis = num_axes - 1
    else:
        legend_axis = 0

    # A couple of simple checks.
    if labels is not None:
        assert len(labels) == num_axes
    if sublabels is not None:
        # assert all(len(g) == len(sublabels) for g in data)
        pass
    if isinstance(color, list):
        assert all(len(g) <= len(color) for g in counts)
    if isinstance(colormap, list):
        assert all(len(g) == len(colormap) for g in counts)

    for i, group in enumerate(counts):
        a = _axes[i]
        group_zorder = i
        if fade:
            kwargs["alpha"] = _get_alpha(i, num_axes)

        num_subgroups = len(group)

        for j, subgroup in enumerate(group):

            if sublabels is None:
                sublabel = None
            else:
                sublabel = sublabels[j]

            element_zorder = group_zorder + j / (num_subgroups + 1)
            element_color = _get_color(i, num_axes, j, num_subgroups)

            # here it doesn't matter what we pass as xrange, forcing to use bin_edges
            plot_density(
                a,
                subgroup,
                bin_edges=bin_edges[i][j],
                fill=fill,
                linecolor=linecolor,
                label=sublabel,
                zorder=element_zorder,
                color=element_color,
                **kwargs
            )

        # Setup the current axis: transparency, labels, spines.
        col_name = None if labels is None else labels[i]
        _setup_axis(
            a,
            global_x_range,
            col_name=col_name,
            grid=ygrid,
            ylabelsize=ylabelsize,
            yrot=yrot,
        )

        # When needed, draw the legend
        if legend and i == legend_axis:
            a.legend(loc=loc)
            # Bypass alpha values, in case
            for p in a.get_legend().get_patches():
                p.set_facecolor(p.get_facecolor())
                p.set_alpha(1.0)
            for l in a.get_legend().get_lines():
                l.set_alpha(1.0)

    # Final adjustments

    # Set the y limit for the density plots.
    # Since the y range in the subplots can vary significantly,
    # different options are available.
    if ylim == "max":
        # Set all yaxis limit to the same value (max range among all)
        max_ylim = max(a.get_ylim()[1] for a in _axes)
        min_ylim = min(a.get_ylim()[0] for a in _axes)
        for a in _axes:
            a.set_ylim([min_ylim - 0.1 * (max_ylim - min_ylim), max_ylim])

    elif ylim == "own":
        # Do nothing, each axis keeps its own ylim
        pass

    else:
        # Set all yaxis lim to the argument value ylim
        try:
            for a in _axes:
                a.set_ylim(ylim)
        except:
            msg = (
                "Warning: the value of ylim must be either 'max', 'own', or a tuple of "
                + "length 2. The value you provided has no effect."
            )
            print(msg)

    # Compute a final axis, used to apply global settings
    last_axis = fig.add_subplot(1, 1, 1)

    # Background color
    if background is not None:
        last_axis.patch.set_facecolor(background)

    for side in ["top", "bottom", "left", "right"]:
        last_axis.spines[side].set_visible(_DEBUG)

    # This looks hacky, but all the axes share the x-axis,
    # so they have the same lims and ticks
    last_axis.set_xlim(_axes[0].get_xlim())
    if xlabels is True:
        last_axis.set_xticks(np.array(_axes[0].get_xticks()[1:-1]))
        for t in last_axis.get_xticklabels():
            t.set_visible(True)
            t.set_fontsize(xlabelsize)
            t.set_rotation(xrot)

        # If grid is enabled, do not allow xticks (they are ugly)
        if xgrid:
            last_axis.tick_params(axis="both", which="both", length=0)
    else:
        last_axis.xaxis.set_visible(False)

    last_axis.yaxis.set_visible(False)
    last_axis.grid(xgrid)

    # Last axis on the back
    last_axis.zorder = min(a.zorder for a in _axes) - 1
    _axes = list(_axes) + [last_axis]

    if title is not None:
        last_axis.set_title(title)

    # The magic overlap happens here.
    h_pad = 5 + (-5 * (1 + overlap))
    fig.tight_layout(h_pad=h_pad)

    return fig, _axes


def color_gradient(x=0.0, start=(0, 0, 0), stop=(1, 1, 1)):
    r = np.interp(x, [0, 1], [start[0], stop[0]])
    g = np.interp(x, [0, 1], [start[1], stop[1]])
    b = np.interp(x, [0, 1], [start[2], stop[2]])
    return (r, g, b)


if __name__ == "__main__":
    from matplotlib import cm
    data_a = np.random.normal(loc=-2.0, size=10000)
    data_b = np.random.normal(loc=2.0, size=9000)

    xrange_a = _x_range(data_a)
    xrange_b = _x_range(data_b)

    bins = 64
    n_plots = 50

    counts_a, bins_a = np.histogram(
        data_a, bins=bins, range=(min(xrange_a), max(xrange_a))
    )
    counts_b, bins_b = np.histogram(
        data_b, bins=bins, range=(min(xrange_b), max(xrange_b))
    )

    # fig_raw, axes_raw = plt.subplots(2, 1)
    # axes_raw[0].hist(data_a, bins=bins)
    # axes_raw[1].hist(data_b, bins=bins)

    # Note: must pass nested lists of counts and bin_edges as below
    # the global range should be passed as a tuple (min, max) as x_range
    # One can skip labels by passing None

    data_dict = {"a": data_a, "b": data_b}
    counts = ([counts_a], [counts_b]) * n_plots
    bin_edges = [[bins_a], [bins_b]] * n_plots
    x_range = ([min(min(bins_a), min(bins_b)), max(max(bins_a), max(bins_b))],)
    labels = tuple(x * 100 if x % 10 == 0 else None for x in range(0, 2 * n_plots))

    fig, axes = ridgeplot(
        counts,
        bin_edges,
        x_range,
        figsize=(12, 10),
        # colormap=cm.OrRd_r,
        linecolor="w",
        linewidth=0.5,
        overlap=1,
        labels=labels,
        fade=True,
        # grid="y",
        title="gradients/conv1.weight",
    )
    plt.show()
