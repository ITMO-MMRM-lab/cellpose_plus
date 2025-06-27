"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import string
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from cellpose import utils

default_font = 12
rcParams["font.family"] = "Arial"
rcParams["savefig.dpi"] = 300
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.titlelocation"] = "left"
rcParams["axes.titleweight"] = "normal"
rcParams["font.size"] = default_font

ltr = string.ascii_lowercase
fs_title = 16
weight_title = "normal"


def plot_label(ltr, il, ax, trans, fs_title=20):
    """
    Plot a label text on a given axis.

        This method adds a label from a list to a specified axis at a predefined position.

        Args:
            ltr: A list of labels from which to select the label to plot.
            il: An index indicating which label to plot from the list.
            ax: The axis on which to plot the label.
            trans: A transformation to apply to the label's positioning.
            fs_title: The font size of the label text (default is 20).

        Returns:
            The updated index after plotting the label.
    """
    ax.text(
        0.0,
        1.0,
        ltr[il],
        transform=ax.transAxes + trans,
        va="bottom",
        fontsize=fs_title,
        fontweight="bold",
    )
    il += 1
    return il


def outlines_img(imgi, maski, color=[1, 0, 0], weight=2):
    """
    Generates an image with outlines drawn around specified mask areas.

        This method takes an input image and a mask, and draws outlines around
        the areas defined by the mask. The outlines can be customized in color
        and thickness.

        Args:
            imgi: The input image array, which should be normalized between 0 and 1.
            maski: The mask array where outlines will be drawn. It indicates the
                    areas to be outlined.
            color: A list specifying the RGB color of the outline. Default is red.
            weight: An integer indicating the thickness of the outline. Default is 2.

        Returns:
            An array representing the modified image with outlines drawn around
            the masked areas.
    """
    img = np.tile(np.clip(imgi.copy(), 0, 1)[:, :, np.newaxis], (1, 1, 3))
    out = np.nonzero(utils.masks_to_outlines(maski[1:-1, 1:-1]))
    img[out[0], out[1]] = np.array(color)
    if weight > 1:
        if weight == 2:
            ix, iy = np.meshgrid(np.arange(0, 3), np.arange(0, 3))
        else:
            ix = np.array([-1, 1, 0, 0])
            iy = np.array([0, 0, 1, 1])
        ix, iy = ix.flatten(), iy.flatten()
        for i in range(len(ix)):
            img[out[0] + ix[i], out[1] + iy[i]] = np.array(color)
    return img
