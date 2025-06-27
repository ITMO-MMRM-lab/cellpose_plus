"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import qtpy
from qtpy.QtWidgets import QAction
from . import io, features
from .. import models


def mainmenu(parent):
    """
    Create and configure the main menu for the application.

        This method sets up the main menu bar for the provided parent widget,
        adding various options under the "File" menu. These options include
        loading images, loading masks, saving sets, and other related file
        operations, each associated with keyboard shortcuts and connected
        to their respective actions.

        Args:
            parent: The parent widget that this menu will be attached to.

        Returns:
            None
    """
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    # load processed data
    loadImg = QAction("&Load image (*.tif, *.png, *.jpg)", parent)
    loadImg.setShortcut("Ctrl+L")
    loadImg.triggered.connect(lambda: io._load_image(parent))
    file_menu.addAction(loadImg)

    parent.autoloadMasks = QAction(
        "Autoload masks from _masks.tif file", parent, checkable=True
    )
    parent.autoloadMasks.setChecked(False)
    file_menu.addAction(parent.autoloadMasks)

    parent.disableAutosave = QAction(
        "Disable autosave _seg.npy file", parent, checkable=True
    )
    parent.disableAutosave.setChecked(False)
    file_menu.addAction(parent.disableAutosave)

    parent.loadMasks = QAction("Load &masks (*.tif, *.png, *.jpg)", parent)
    parent.loadMasks.setShortcut("Ctrl+M")
    parent.loadMasks.triggered.connect(lambda: io._load_masks(parent))
    file_menu.addAction(parent.loadMasks)
    parent.loadMasks.setEnabled(False)

    loadManual = QAction("Load &processed/labelled image (*_seg.npy)", parent)
    loadManual.setShortcut("Ctrl+P")
    loadManual.triggered.connect(lambda: io._load_seg(parent))
    file_menu.addAction(loadManual)

    parent.saveSet = QAction("&Save masks and image (as *_seg.npy)", parent)
    parent.saveSet.setShortcut("Ctrl+S")
    parent.saveSet.triggered.connect(lambda: io._save_sets(parent))
    file_menu.addAction(parent.saveSet)
    parent.saveSet.setEnabled(False)

    parent.savePNG = QAction("Save masks as P&NG/tif", parent)
    parent.savePNG.setShortcut("Ctrl+N")
    parent.savePNG.triggered.connect(lambda: io._save_png(parent))
    file_menu.addAction(parent.savePNG)
    parent.savePNG.setEnabled(False)

    parent.saveOutlines = QAction("Save &Outlines as text for imageJ", parent)
    parent.saveOutlines.setShortcut("Ctrl+O")
    parent.saveOutlines.triggered.connect(lambda: io._save_outlines(parent))
    file_menu.addAction(parent.saveOutlines)
    parent.saveOutlines.setEnabled(False)

    parent.saveROIs = QAction(
        "Save outlines as .zip archive of &ROI files for ImageJ", parent
    )
    parent.saveROIs.setShortcut("Ctrl+R")
    parent.saveROIs.triggered.connect(lambda: io._save_rois(parent))
    file_menu.addAction(parent.saveROIs)
    parent.saveROIs.setEnabled(False)

    parent.saveFlows = QAction("Save &Flows and cellprob as tif", parent)
    parent.saveFlows.setShortcut("Ctrl+F")
    parent.saveFlows.triggered.connect(lambda: io._save_flows(parent))
    file_menu.addAction(parent.saveFlows)
    parent.saveFlows.setEnabled(False)


def editmenu(parent):
    """
    Creates and configures the Edit menu for the application.

        This method establishes an Edit menu on the application's main menu bar.
        It adds several actions related to editing functionalities such as Undo, Redo, Clear, and
        removing or merging cells. Each action is bound to specific keyboard shortcuts and connected
        to their respective event handlers.

        Args:
            parent: The parent widget to which the menu and actions are attached.

        Returns:
            None: The function does not return any value, but modifies the parent widget by adding
            an Edit menu with various actions.
    """
    main_menu = parent.menuBar()
    edit_menu = main_menu.addMenu("&Edit")
    parent.undo = QAction("Undo previous mask/trace", parent)
    parent.undo.setShortcut("Ctrl+Z")
    parent.undo.triggered.connect(parent.undo_action)
    parent.undo.setEnabled(False)
    edit_menu.addAction(parent.undo)

    parent.redo = QAction("Undo remove mask", parent)
    parent.redo.setShortcut("Ctrl+Y")
    parent.redo.triggered.connect(parent.undo_remove_action)
    parent.redo.setEnabled(False)
    edit_menu.addAction(parent.redo)

    parent.ClearButton = QAction("Clear all masks", parent)
    parent.ClearButton.setShortcut("Ctrl+0")
    parent.ClearButton.triggered.connect(parent.clear_all)
    parent.ClearButton.setEnabled(False)
    edit_menu.addAction(parent.ClearButton)

    parent.remcell = QAction("Remove selected cell (Ctrl+CLICK)", parent)
    parent.remcell.setShortcut("Ctrl+Click")
    parent.remcell.triggered.connect(parent.remove_action)
    parent.remcell.setEnabled(False)
    edit_menu.addAction(parent.remcell)

    parent.mergecell = QAction("FYI: Merge cells by Alt+Click", parent)
    parent.mergecell.setEnabled(False)
    edit_menu.addAction(parent.mergecell)


def modelmenu(parent):
    """
    Creates a model menu in the provided parent widget.

        This method initializes a menu in the given parent widget's menu bar,
        specifically for managing custom models. It adds actions for adding,
        removing, and training models, as well as for displaying training instructions.

        Args:
            parent: The parent widget that contains the menu bar where the model menu will be added.

        Returns:
            None
    """
    main_menu = parent.menuBar()
    io._init_model_list(parent)
    model_menu = main_menu.addMenu("&Models")
    parent.addmodel = QAction("Add custom torch model to GUI", parent)
    # parent.addmodel.setShortcut("Ctrl+A")
    parent.addmodel.triggered.connect(parent.add_model)
    parent.addmodel.setEnabled(True)
    model_menu.addAction(parent.addmodel)

    parent.removemodel = QAction("Remove selected custom model from GUI", parent)
    # parent.removemodel.setShortcut("Ctrl+R")
    parent.removemodel.triggered.connect(parent.remove_model)
    parent.removemodel.setEnabled(True)
    model_menu.addAction(parent.removemodel)

    parent.newmodel = QAction("&Train new model with image+masks in folder", parent)
    parent.newmodel.setShortcut("Ctrl+T")
    parent.newmodel.triggered.connect(parent.new_model)
    parent.newmodel.setEnabled(False)
    model_menu.addAction(parent.newmodel)

    openTrainHelp = QAction("Training instructions", parent)
    openTrainHelp.triggered.connect(parent.train_help_window)
    model_menu.addAction(openTrainHelp)


def masksmenu(parent):
    """
    Creates and configures the masks menu in the parent application's menu bar.

        This method adds a menu labeled 'Masks' to the main menu bar of the parent
        application. It includes options to temporarily save a mask and to save
        labeled masks, setting up the necessary actions and connecting them to their
        respective functionalities. The menu actions are initially disabled.

        Args:
            parent: The parent object that contains the menu bar and features class.

        Returns:
            None
    """
    main_menu = parent.menuBar()
    masks_menu = main_menu.addMenu("&Masks")

    parent.keepMask = QAction("Save mask temporarily", parent)
    parent.keepMask.triggered.connect(
        lambda: parent.features_class.save_temp_output(gui_self=parent)
    )
    parent.keepMask.setEnabled(False)
    masks_menu.addAction(parent.keepMask)

    parent.saveMasks = QAction("Save labeled mask", parent)
    parent.saveMasks.triggered.connect(
        lambda: parent.features_class.save_labeled_masks(gui_self=parent)
    )
    parent.saveMasks.setEnabled(False)
    masks_menu.addAction(parent.saveMasks)

    parent.features_class.main_masks_menu = masks_menu


def imagesmenu(parent):
    """
    Creates and adds an Images menu to the main menu bar of the given parent widget.

        This method initializes an "Images" menu and attaches it to the parent widget's menu bar.
        It also stores a reference to the created menu in the parent widget's features class for
        further manipulation or access.

        Args:
            parent: The parent widget that contains the menu bar to which the Images menu will be added.

        Returns:
            None
    """
    main_menu = parent.menuBar()
    images_menu = main_menu.addMenu("&Images")
    parent.features_class.main_images_menu = images_menu


def helpmenu(parent):
    """
    Creates and displays the Help menu in the application.

        This method adds a Help menu to the main menu bar of the given parent widget.
        It includes options for accessing help with the GUI, information about the GUI layout,
        and training instructions, each connected to their respective action handlers.

        Args:
            parent: The parent widget that contains the menu bar.

        Returns:
            None
    """
    main_menu = parent.menuBar()
    help_menu = main_menu.addMenu("&Help")

    openHelp = QAction("&Help with GUI", parent)
    openHelp.setShortcut("Ctrl+H")
    openHelp.triggered.connect(parent.help_window)
    help_menu.addAction(openHelp)

    openGUI = QAction("&GUI layout", parent)
    openGUI.setShortcut("Ctrl+G")
    openGUI.triggered.connect(parent.gui_window)
    help_menu.addAction(openGUI)

    openTrainHelp = QAction("Training instructions", parent)
    openTrainHelp.triggered.connect(parent.train_help_window)
    help_menu.addAction(openTrainHelp)
