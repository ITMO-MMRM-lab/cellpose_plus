"""
Copyright © 2024 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
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
    Plot a label on the specified axes.

        This method adds a text label to the given axes at a specified location
        and with specified formatting parameters.

        Args:
            ltr: A list of labels to choose from.
            il: An index to determine which label from the list to use.
            ax: The axes object on which to plot the label.
            trans: Transformation to apply to the label's position.
            fs_title: The font size of the label title (default is 20).

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
