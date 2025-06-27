"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import sys, os, pathlib, warnings, datetime, time, copy, math

from qtpy import QtGui, QtCore
from superqt import QRangeSlider, QCollapsible
from qtpy.QtWidgets import (
    QScrollArea,
    QMainWindow,
    QAction,
    QMenu,
    QApplication,
    QWidget,
    QScrollBar,
    QComboBox,
    QGridLayout,
    QPushButton,
    QFrame,
    QCheckBox,
    QLabel,
    QProgressBar,
    QLineEdit,
    QMessageBox,
    QGroupBox,
)
import pyqtgraph as pg

import pandas as pd
import numpy as np
from scipy.stats import mode
import cv2

from . import guiparts, menus, io, symmetry, features
from .. import models, core, dynamics, version, denoise, train
from ..utils import download_url_to_file, masks_to_outlines, diameters, download_font
from ..io import get_image_files, imsave, imread
from ..transforms import resize_image, normalize99, normalize99_tile, smooth_sharpen_img
from ..models import normalize_default
from ..plot import disk

from scipy.ndimage import find_objects
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import ndimage
import diplib as dip
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB = True
except:
    MATPLOTLIB = False

try:
    from google.cloud import storage

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "key/cellpose-data-writer.json"
    )
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False

Horizontal = QtCore.Qt.Orientation.Horizontal


class Slider(QRangeSlider):
    """
    A class that represents a slider component for user interfaces.

    The Slider class provides functionality to create a slider component
    that allows users to select a value from a defined range. It notifies
    its parent object whenever the slider's value changes.

    Methods:
        __init__
        levelChanged

    Attributes:
        parent
        name
        color

    The __init__ method initializes the slider with properties such as
    its parent component, its name, and its color. The levelChanged method
    serves to notify the parent component whenever the slider's value changes,
    allowing the parent to react to the change accordingly.
    """

    def __init__(self, parent, name, color):
        """
        Initializes a new instance of the class.

            This constructor sets up the slider with specified properties,
            connects the value change event to a method, and configures the
            appearance of the slider.

            Args:
                parent: The parent component that will contain this slider.
                name: The name to be assigned to this slider instance.
                color: The color that may be used for styling the slider.

            Returns:
                None
        """
        super().__init__(Horizontal)
        self.setEnabled(False)
        self.valueChanged.connect(lambda: self.levelChanged(parent))
        self.name = name

        self.setStyleSheet(
            """ QSlider{
                             background-color: transparent;
                             }
        """
        )
        self.show()

    def levelChanged(self, parent):
        """
        Notifies the parent object of a level change.

            This method calls the level_change method on the parent object,
            passing the name of the current object to update the parent's
            state based on the level change.

            Args:
                parent: The parent object that needs to be notified about the
                        level change.

            Returns:
                None: This method does not return a value.
        """
        parent.level_change(self.name)


class QHLine(QFrame):
    """
    Represents a horizontal line frame.

    This class is designed to create and manage a horizontal line frame with specific dimensions and line width.

    Methods:
        __init__: Initializes a horizontal line frame.

    Attributes:
        None

    The __init__ method sets up the horizontal line frame by calling the superclass initializer to define the shape and line width.
    """

    def __init__(self):
        """
        Initializes a horizontal line frame.

            This method sets up a horizontal line frame with a defined shape and line width by calling the superclass initializer.

            Parameters:
                None

            Returns:
                None
        """
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        # self.setFrameShadow(QFrame.Sunken)
        self.setLineWidth(8)


def make_bwr():
    """
    Generate a blue-white-red colormap.

        This method creates a blue-white-red (BWR) colormap which transitions from blue to white to red.
        The colormap is constructed using a gradient of RGB values and is suitable for visualizing data that
        ranges both above and below a central value.

        Returns:
            A ColorMap object representing the BWR colormap.
    """
    # make a bwr colormap
    b = np.append(255 * np.ones(128), np.linspace(0, 255, 128)[::-1])[:, np.newaxis]
    r = np.append(np.linspace(0, 255, 128), 255 * np.ones(128))[:, np.newaxis]
    g = np.append(np.linspace(0, 255, 128), np.linspace(0, 255, 128)[::-1])[
        :, np.newaxis
    ]
    color = np.concatenate((r, g, b), axis=-1).astype(np.uint8)
    bwr = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return bwr


def make_spectral():
    """
    Generate a spectral colormap.

    This method creates a colormap that transitions through a spectral range of colors.
    The colormap is constructed using predefined RGB values arranged in a gradient.
    It returns a `ColorMap` object that can be used for visualizing data with a spectral color scheme.

    Returns:
        A `ColorMap` object representing the spectral colormap.
    """
    # make spectral colormap
    r = np.array(
        [
            0,
            4,
            8,
            12,
            16,
            20,
            24,
            28,
            32,
            36,
            40,
            44,
            48,
            52,
            56,
            60,
            64,
            68,
            72,
            76,
            80,
            84,
            88,
            92,
            96,
            100,
            104,
            108,
            112,
            116,
            120,
            124,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            120,
            112,
            104,
            96,
            88,
            80,
            72,
            64,
            56,
            48,
            40,
            32,
            24,
            16,
            8,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            7,
            11,
            15,
            19,
            23,
            27,
            31,
            35,
            39,
            43,
            47,
            51,
            55,
            59,
            63,
            67,
            71,
            75,
            79,
            83,
            87,
            91,
            95,
            99,
            103,
            107,
            111,
            115,
            119,
            123,
            127,
            131,
            135,
            139,
            143,
            147,
            151,
            155,
            159,
            163,
            167,
            171,
            175,
            179,
            183,
            187,
            191,
            195,
            199,
            203,
            207,
            211,
            215,
            219,
            223,
            227,
            231,
            235,
            239,
            243,
            247,
            251,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
        ]
    )
    g = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            9,
            9,
            8,
            8,
            7,
            7,
            6,
            6,
            5,
            5,
            5,
            4,
            4,
            3,
            3,
            2,
            2,
            1,
            1,
            0,
            0,
            0,
            7,
            15,
            23,
            31,
            39,
            47,
            55,
            63,
            71,
            79,
            87,
            95,
            103,
            111,
            119,
            127,
            135,
            143,
            151,
            159,
            167,
            175,
            183,
            191,
            199,
            207,
            215,
            223,
            231,
            239,
            247,
            255,
            247,
            239,
            231,
            223,
            215,
            207,
            199,
            191,
            183,
            175,
            167,
            159,
            151,
            143,
            135,
            128,
            129,
            131,
            132,
            134,
            135,
            137,
            139,
            140,
            142,
            143,
            145,
            147,
            148,
            150,
            151,
            153,
            154,
            156,
            158,
            159,
            161,
            162,
            164,
            166,
            167,
            169,
            170,
            172,
            174,
            175,
            177,
            178,
            180,
            181,
            183,
            185,
            186,
            188,
            189,
            191,
            193,
            194,
            196,
            197,
            199,
            201,
            202,
            204,
            205,
            207,
            208,
            210,
            212,
            213,
            215,
            216,
            218,
            220,
            221,
            223,
            224,
            226,
            228,
            229,
            231,
            232,
            234,
            235,
            237,
            239,
            240,
            242,
            243,
            245,
            247,
            248,
            250,
            251,
            253,
            255,
            251,
            247,
            243,
            239,
            235,
            231,
            227,
            223,
            219,
            215,
            211,
            207,
            203,
            199,
            195,
            191,
            187,
            183,
            179,
            175,
            171,
            167,
            163,
            159,
            155,
            151,
            147,
            143,
            139,
            135,
            131,
            127,
            123,
            119,
            115,
            111,
            107,
            103,
            99,
            95,
            91,
            87,
            83,
            79,
            75,
            71,
            67,
            63,
            59,
            55,
            51,
            47,
            43,
            39,
            35,
            31,
            27,
            23,
            19,
            15,
            11,
            7,
            3,
            0,
            8,
            16,
            24,
            32,
            41,
            49,
            57,
            65,
            74,
            82,
            90,
            98,
            106,
            115,
            123,
            131,
            139,
            148,
            156,
            164,
            172,
            180,
            189,
            197,
            205,
            213,
            222,
            230,
            238,
            246,
            254,
        ]
    )
    b = np.array(
        [
            0,
            7,
            15,
            23,
            31,
            39,
            47,
            55,
            63,
            71,
            79,
            87,
            95,
            103,
            111,
            119,
            127,
            135,
            143,
            151,
            159,
            167,
            175,
            183,
            191,
            199,
            207,
            215,
            223,
            231,
            239,
            247,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            251,
            247,
            243,
            239,
            235,
            231,
            227,
            223,
            219,
            215,
            211,
            207,
            203,
            199,
            195,
            191,
            187,
            183,
            179,
            175,
            171,
            167,
            163,
            159,
            155,
            151,
            147,
            143,
            139,
            135,
            131,
            128,
            126,
            124,
            122,
            120,
            118,
            116,
            114,
            112,
            110,
            108,
            106,
            104,
            102,
            100,
            98,
            96,
            94,
            92,
            90,
            88,
            86,
            84,
            82,
            80,
            78,
            76,
            74,
            72,
            70,
            68,
            66,
            64,
            62,
            60,
            58,
            56,
            54,
            52,
            50,
            48,
            46,
            44,
            42,
            40,
            38,
            36,
            34,
            32,
            30,
            28,
            26,
            24,
            22,
            20,
            18,
            16,
            14,
            12,
            10,
            8,
            6,
            4,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            8,
            16,
            24,
            32,
            41,
            49,
            57,
            65,
            74,
            82,
            90,
            98,
            106,
            115,
            123,
            131,
            139,
            148,
            156,
            164,
            172,
            180,
            189,
            197,
            205,
            213,
            222,
            230,
            238,
            246,
            254,
        ]
    )
    color = (np.vstack((r, g, b)).T).astype(np.uint8)
    spectral = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return spectral


def make_cmap(cm=0):
    """
    Creates a single channel colormap.

        This method generates a colormap with intensity values ranging from
        0 to 255 for a specific color channel specified by the parameter.

        Args:
            cm: The index of the color channel to be used (0 for red,
                1 for green, and 2 for blue).

        Returns:
            A colormap object containing the color gradients for the specified
            channel.
    """
    # make a single channel colormap
    r = np.arange(0, 256)
    color = np.zeros((256, 3))
    color[:, cm] = r
    color = color.astype(np.uint8)
    cmap = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return cmap


def run(image=None):
    """
    Run the application to initiate the Cellpose GUI.

        This method initializes the Qt application and sets up the necessary GUI resources,
        including downloading images and icons if they do not already exist. It then launches
        the main application window where the user can interact with the Cellpose functionality.

        Args:
            image: Optional parameter that can be used to pass an initial image for processing.

        Returns:
            An integer exit code indicating the success or failure of the application.
    """
    from ..io import logger_setup

    logger, log_file = logger_setup()
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
    guip_path = pathlib.Path.home().joinpath(".cellpose", "cellpose_gui.png")

    primary_icon_path = pathlib.Path.home().joinpath(".cellpose", "primary.png")
    primary_icon_url = "https://github.com/ITMO-MMRM-lab/cellpose/blob/main/cellpose/resources/primary.png?raw=true"

    secondary_icon_path = pathlib.Path.home().joinpath(".cellpose", "secondary.png")
    secondary_icon_url = "https://github.com/ITMO-MMRM-lab/cellpose/blob/main/cellpose/resources/secondary.png?raw=true"

    if not icon_path.is_file():
        cp_dir = pathlib.Path.home().joinpath(".cellpose")
        cp_dir.mkdir(exist_ok=True)
        print("downloading logo")
        download_url_to_file(
            "https://www.cellpose.org/static/images/cellpose_transparent.png",
            icon_path,
            progress=True,
        )
    if not guip_path.is_file():
        print("downloading help window image")
        download_url_to_file(
            "https://www.cellpose.org/static/images/cellpose_gui.png",
            guip_path,
            progress=True,
        )
    if not primary_icon_path.is_file():
        print("downloading primary mask image")
        download_url_to_file(primary_icon_url, primary_icon_path, progress=True)
    if not secondary_icon_path.is_file():
        print("downloading secondary mask image")
        download_url_to_file(secondary_icon_url, secondary_icon_path, progress=True)

    download_font()

    icon_path = str(icon_path.resolve())
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(64, 64))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    app.setStyle("Fusion")
    app.setPalette(guiparts.DarkPalette())
    # app.setStyleSheet("QLineEdit { color: yellow }")

    # models.download_model_weights() # does not exist
    MainW(image=image, logger=logger)
    ret = app.exec_()
    sys.exit(ret)


class MainW(QMainWindow):
    """
    MainW is the primary class that facilitates the graphical user interface for image processing and manipulation.

    This class provides methods for initializing the application, managing UI components, image loading,
    performing computations for segmentation and denoising, and handling user interactions. It also includes
    functionalities for managing models and their training, adjusting visual parameters, and toggling various
    operations related to image and cell analysis.

    Methods:
        - __init__
        - help_window
        - train_help_window
        - gui_window
        - make_buttons
        - update_px_to_mm
        - level_change
        - keyPressEvent
        - autosave_on
        - check_gpu
        - get_channels
        - model_choose
        - calibrate_size
        - toggle_scale
        - enable_buttons
        - disable_buttons_removeROIs
        - toggle_mask_ops
        - toggle_saving
        - toggle_removals
        - remove_action
        - undo_action
        - undo_remove_action
        - get_files
        - get_prev_image
        - get_next_image
        - dragEnterEvent
        - dropEvent
        - toggle_masks
        - make_viewbox
        - reset
        - delete_restore
        - clear_restore
        - brush_choose
        - clear_all
        - select_cell
        - select_cell_multi
        - unselect_cell
        - unselect_cell_multi
        - remove_cell
        - remove_single_cell
        - remove_region_cells
        - delete_multiple_cells
        - done_remove_multiple_cells
        - merge_cells
        - undo_remove_cell
        - remove_stroke
        - plot_clicked
        - cancel_remove_multiple
        - clear_multi_selected_cells
        - add_roi
        - remove_roi
        - roi_changed
        - mouse_moved
        - color_choose
        - update_plot
        - update_layer
        - update_roi_count
        - add_set
        - add_mask
        - draw_mask
        - compute_scale
        - update_scale
        - redraw_masks
        - draw_masks
        - draw_layer
        - set_restore_button
        - set_normalize_params
        - check_percentile_params
        - check_filter_params
        - get_normalize_params
        - compute_saturation
        - chanchoose
        - get_model_path
        - initialize_model
        - add_model
        - remove_model
        - new_model
        - train_model
        - compute_restore
        - get_thresholds
        - compute_cprob
        - compute_denoise_model
        - compute_segmentation

    Attributes:
        - useGPU
        - scale_on
        - autosave

    The methods within this class primarily manage the user interface, interaction with the graphical elements,
    and implement the core functionalities such as image processing, model training, and data visualization.
    The attributes maintain the state regarding GPU usage, visibility of specific components, and autosave settings.
    """

    def __init__(self, image=None, logger=None):
        """
        Initialize the main application window.

            This method sets up the main application window, including the layout,
            menus, and configuration options for the graphical user interface. It
            also handles loading an image if provided.

            Args:
                image: Optional; an image file to load into the application at startup.
                logger: Optional; a logger instance for logging events and messages.

            Returns:
                None.
        """
        super(MainW, self).__init__()

        self.logger = logger
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(50, 50, 1200, 1000)
        self.setWindowTitle(f"cellpose v{version}")
        self.cp_path = os.path.dirname(os.path.realpath(__file__))
        app_icon = QtGui.QIcon()
        icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
        icon_path = str(icon_path.resolve())
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(64, 64))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)
        # rgb(150,255,150)
        self.setStyleSheet(guiparts.stylesheet())

        self.main_masks_menu = None  # Pointer to masks menu
        self.main_images_menu = None  # Pointer to images menu
        self.temp_masks = []
        self.px_to_mm = 0.0
        self.selected_model = None

        self.features_class = features.FeatureExtraction()

        menus.mainmenu(self)
        menus.editmenu(self)
        menus.modelmenu(self)
        menus.masksmenu(self)
        menus.imagesmenu(self)
        menus.helpmenu(self)

        self.stylePressed = """QPushButton {Text-align: center; 
                             background-color: rgb(150,50,150); 
                             border-color: white;
                             color:white;}
                            QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }"""
        self.styleUnpressed = """QPushButton {Text-align: center; 
                               background-color: rgb(50,50,50);
                                border-color: white;
                               color:white;}
                                QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }"""
        self.loaded = False

        # ---- MAIN WIDGET LAYOUT ---- #
        self.cwidget = QWidget(self)
        self.lmain = QGridLayout()
        self.cwidget.setLayout(self.lmain)
        self.setCentralWidget(self.cwidget)
        self.lmain.setVerticalSpacing(0)
        self.lmain.setContentsMargins(0, 0, 0, 10)

        self.imask = 0
        self.scrollarea = QScrollArea()
        self.scrollarea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollarea.setStyleSheet("""QScrollArea { border: none }""")
        self.scrollarea.setWidgetResizable(True)
        self.swidget = QWidget(self)
        self.scrollarea.setWidget(self.swidget)
        self.l0 = QGridLayout()
        self.swidget.setLayout(self.l0)
        b = self.make_buttons()
        self.lmain.addWidget(self.scrollarea, 0, 0, 39, 9)

        # ---- drawing area ---- #
        self.win = pg.GraphicsLayoutWidget()

        self.lmain.addWidget(self.win, 0, 9, 40, 30)

        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.scene().sigMouseMoved.connect(self.mouse_moved)
        self.make_viewbox()
        self.lmain.setColumnStretch(10, 1)
        bwrmap = make_bwr()
        self.bwr = bwrmap.getLookupTable(start=0.0, stop=255.0, alpha=False)
        self.cmap = []
        # spectral colormap
        self.cmap.append(
            make_spectral().getLookupTable(start=0.0, stop=255.0, alpha=False)
        )
        # single channel colormaps
        for i in range(3):
            self.cmap.append(
                make_cmap(i).getLookupTable(start=0.0, stop=255.0, alpha=False)
            )

        if MATPLOTLIB:
            self.colormap = (
                plt.get_cmap("gist_ncar")(np.linspace(0.0, 0.9, 1000000)) * 255
            ).astype(np.uint8)
            np.random.seed(42)  # make colors stable
            self.colormap = self.colormap[np.random.permutation(1000000)]
        else:
            np.random.seed(42)  # make colors stable
            self.colormap = ((np.random.rand(1000000, 3) * 0.8 + 0.1) * 255).astype(
                np.uint8
            )
        self.NZ = 1
        self.restore = None
        self.ratio = 1.0
        self.reset()

        # if called with image, load it
        if image is not None:
            self.filename = image
            io._load_image(self, self.filename)

        # training settings
        d = datetime.datetime.now()
        self.training_params = {
            "model_index": 0,
            "learning_rate": 0.1,
            "weight_decay": 0.0001,
            "n_epochs": 100,
            "SGD": True,
            "model_name": "CP" + d.strftime("_%Y%m%d_%H%M%S"),
        }

        self.load_3D = False
        self.stitch_threshold = 0.0
        self.flow3D_smooth = 0.0
        self.anisotropy = 1.0
        self.min_size = 15
        self.resample = True

        self.setAcceptDrops(True)
        self.win.show()
        self.show()

    def help_window(self):
        """
        Displays the help window.

            This method creates an instance of the HelpWindow class from the
            guiparts module and displays it to the user.

            Parameters:
                None

            Returns:
                None
        """
        HW = guiparts.HelpWindow(self)
        HW.show()

    def train_help_window(self):
        """
        Displays the Train Help Window.

            This method initializes and shows the Train Help Window,
            which provides assistance and information related to train operations
            within the application.

            Parameters:
                None

            Returns:
                None
        """
        THW = guiparts.TrainHelpWindow(self)
        THW.show()

    def gui_window(self):
        """
        Launches the graphical user interface window.

            This method initializes and displays an instance of the ExampleGUI
            class, providing the main interface for user interaction.

            Parameters:
                None

            Returns:
                None
        """
        EG = guiparts.ExampleGUI(self)
        EG.show()

    def make_buttons(self):
        """
        Create and organize UI buttons and controls for the application.

            This method sets up various buttons, dropdowns, and checkboxes for
            adjusting visualization parameters, drawing settings, segmentation
            controls, and image restoration features.

            The following UI elements are created:
            - Dropdowns for selecting color mode and view types.
            - Sliders for color adjustments.
            - Checkboxes for enabling/disabling features like auto-adjustment
              of saturation and the visibility of masks or outlines.
            - Buttons for performing actions related to segmentation and denoising.

            This method also organizes these elements within a grid layout to
            ensure a consistent and user-friendly interface.

            Returns:
                int: The current row index after adding the button elements to
                the layout, allowing for subsequent UI elements to be added
                in sequence.
        """
        self.boldfont = QtGui.QFont("Arial", 11, QtGui.QFont.Bold)
        self.boldmedfont = QtGui.QFont("Arial", 9, QtGui.QFont.Bold)
        self.medfont = QtGui.QFont("Arial", 9)
        self.smallfont = QtGui.QFont("Arial", 8)

        b = 0
        self.satBox = QGroupBox("Views")
        self.satBox.setFont(self.boldfont)
        self.satBoxG = QGridLayout()
        self.satBox.setLayout(self.satBoxG)
        self.l0.addWidget(self.satBox, b, 0, 1, 9)

        b0 = 0
        self.view = 0  # 0=image, 1=flowsXY, 2=flowsZ, 3=cellprob
        self.color = 0  # 0=RGB, 1=gray, 2=R, 3=G, 4=B
        self.RGBDropDown = QComboBox()
        self.RGBDropDown.addItems(
            ["RGB", "red=R", "green=G", "blue=B", "gray", "spectral"]
        )
        self.RGBDropDown.setFont(self.medfont)
        self.RGBDropDown.currentIndexChanged.connect(self.color_choose)
        self.satBoxG.addWidget(self.RGBDropDown, b0, 0, 1, 3)

        label = QLabel("<p>[&uarr; / &darr; or W/S]</p>")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, b0, 3, 1, 3)
        label = QLabel("[R / G / B \n toggles color ]")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, b0, 6, 1, 3)

        b0 += 1
        self.ViewDropDown = QComboBox()
        self.ViewDropDown.addItems(["image", "gradXY", "cellprob", "restored"])
        self.ViewDropDown.setFont(self.medfont)
        self.ViewDropDown.model().item(3).setEnabled(False)
        self.ViewDropDown.currentIndexChanged.connect(self.update_plot)
        self.satBoxG.addWidget(self.ViewDropDown, b0, 0, 2, 3)

        label = QLabel("[pageup / pagedown]")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, b0, 3, 1, 5)

        b0 += 2
        label = QLabel("")
        label.setToolTip(
            "NOTE: manually changing the saturation bars does not affect normalization in segmentation"
        )
        self.satBoxG.addWidget(label, b0, 0, 1, 5)

        self.autobtn = QCheckBox("auto-adjust saturation")
        self.autobtn.setToolTip("sets scale-bars as normalized for segmentation")
        self.autobtn.setFont(self.medfont)
        self.autobtn.setChecked(True)
        self.satBoxG.addWidget(self.autobtn, b0, 1, 1, 8)

        b0 += 1
        self.sliders = []
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [100, 100, 100]]
        colornames = ["red", "Chartreuse", "DodgerBlue"]
        names = ["red", "green", "blue"]
        for r in range(3):
            b0 += 1
            if r == 0:
                label = QLabel('<font color="gray">gray/</font><br>red')
            else:
                label = QLabel(names[r] + ":")
            label.setStyleSheet(f"color: {colornames[r]}")
            label.setFont(self.boldmedfont)
            self.satBoxG.addWidget(label, b0, 0, 1, 2)
            self.sliders.append(Slider(self, names[r], colors[r]))
            self.sliders[-1].setMinimum(-0.1)
            self.sliders[-1].setMaximum(255.1)
            self.sliders[-1].setValue([0, 255])
            self.sliders[-1].setToolTip(
                "NOTE: manually changing the saturation bars does not affect normalization in segmentation"
            )
            # self.sliders[-1].setTickPosition(QSlider.TicksRight)
            self.satBoxG.addWidget(self.sliders[-1], b0, 2, 1, 7)

        b += 1
        self.drawBox = QGroupBox("Drawing")
        self.drawBox.setFont(self.boldfont)
        self.drawBoxG = QGridLayout()
        self.drawBox.setLayout(self.drawBoxG)
        self.l0.addWidget(self.drawBox, b, 0, 1, 9)
        self.autosave = True

        b0 = 0
        self.brush_size = 3
        self.BrushChoose = QComboBox()
        self.BrushChoose.addItems(["1", "3", "5", "7", "9"])
        self.BrushChoose.currentIndexChanged.connect(self.brush_choose)
        self.BrushChoose.setFixedWidth(40)
        self.BrushChoose.setFont(self.medfont)
        self.drawBoxG.addWidget(self.BrushChoose, b0, 3, 1, 2)
        label = QLabel("brush size:")
        label.setFont(self.medfont)
        self.drawBoxG.addWidget(label, b0, 0, 1, 3)

        b0 += 1
        # turn off masks
        self.layer_off = False
        self.masksOn = True
        self.MCheckBox = QCheckBox("MASKS ON [X]")
        self.MCheckBox.setFont(self.medfont)
        self.MCheckBox.setChecked(True)
        self.MCheckBox.toggled.connect(self.toggle_masks)
        self.drawBoxG.addWidget(self.MCheckBox, b0, 0, 1, 5)

        b0 += 1
        # turn off outlines
        self.outlinesOn = False  # turn off by default
        self.OCheckBox = QCheckBox("outlines on [Z]")
        self.OCheckBox.setFont(self.medfont)
        self.drawBoxG.addWidget(self.OCheckBox, b0, 0, 1, 5)
        self.OCheckBox.setChecked(False)
        self.OCheckBox.toggled.connect(self.toggle_masks)

        b0 += 1
        self.SCheckBox = QCheckBox("single stroke")
        self.SCheckBox.setFont(self.medfont)
        self.SCheckBox.setChecked(True)
        self.SCheckBox.toggled.connect(self.autosave_on)
        self.SCheckBox.setEnabled(True)
        self.drawBoxG.addWidget(self.SCheckBox, b0, 0, 1, 5)

        # buttons for deleting multiple cells
        self.deleteBox = QGroupBox("delete multiple ROIs")
        self.deleteBox.setStyleSheet("color: rgb(200, 200, 200)")
        self.deleteBox.setFont(self.medfont)
        self.deleteBoxG = QGridLayout()
        self.deleteBox.setLayout(self.deleteBoxG)
        self.drawBoxG.addWidget(self.deleteBox, 0, 5, 4, 4)
        self.MakeDeletionRegionButton = QPushButton("region-select")
        self.MakeDeletionRegionButton.clicked.connect(self.remove_region_cells)
        self.deleteBoxG.addWidget(self.MakeDeletionRegionButton, 0, 0, 1, 4)
        self.MakeDeletionRegionButton.setFont(self.smallfont)
        self.MakeDeletionRegionButton.setFixedWidth(70)
        self.DeleteMultipleROIButton = QPushButton("click-select")
        self.DeleteMultipleROIButton.clicked.connect(self.delete_multiple_cells)
        self.deleteBoxG.addWidget(self.DeleteMultipleROIButton, 1, 0, 1, 4)
        self.DeleteMultipleROIButton.setFont(self.smallfont)
        self.DeleteMultipleROIButton.setFixedWidth(70)
        self.DoneDeleteMultipleROIButton = QPushButton("done")
        self.DoneDeleteMultipleROIButton.clicked.connect(
            self.done_remove_multiple_cells
        )
        self.deleteBoxG.addWidget(self.DoneDeleteMultipleROIButton, 2, 0, 1, 2)
        self.DoneDeleteMultipleROIButton.setFont(self.smallfont)
        self.DoneDeleteMultipleROIButton.setFixedWidth(35)
        self.CancelDeleteMultipleROIButton = QPushButton("cancel")
        self.CancelDeleteMultipleROIButton.clicked.connect(self.cancel_remove_multiple)
        self.deleteBoxG.addWidget(self.CancelDeleteMultipleROIButton, 2, 2, 1, 2)
        self.CancelDeleteMultipleROIButton.setFont(self.smallfont)
        self.CancelDeleteMultipleROIButton.setFixedWidth(35)

        b += 1
        b0 = 0
        self.segBox = QGroupBox("Segmentation")
        self.segBoxG = QGridLayout()
        self.segBox.setLayout(self.segBoxG)
        self.l0.addWidget(self.segBox, b, 0, 1, 9)
        self.segBox.setFont(self.boldfont)

        self.diameter = 30
        label = QLabel("diameter (pixels):")
        label.setFont(self.medfont)
        label.setToolTip(
            "you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)"
        )
        self.segBoxG.addWidget(label, b0, 0, 1, 4)
        self.Diameter = QLineEdit()
        self.Diameter.setToolTip(
            'you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the "cyto3" model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)'
        )
        self.Diameter.setText(str(self.diameter))
        self.Diameter.setFont(self.medfont)
        self.Diameter.returnPressed.connect(self.update_scale)
        self.Diameter.setFixedWidth(50)
        self.segBoxG.addWidget(self.Diameter, b0, 4, 1, 2)

        # compute diameter
        self.SizeButton = QPushButton("calibrate")
        self.SizeButton.setFont(self.medfont)
        self.SizeButton.clicked.connect(self.calibrate_size)
        self.segBoxG.addWidget(self.SizeButton, b0, 6, 1, 3)
        # self.SizeButton.setFixedWidth(65)
        self.SizeButton.setEnabled(False)
        self.SizeButton.setToolTip(
            "you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the cyto3 model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)"
        )

        b0 += 1
        label = QLabel("Length in μm:")
        label.setToolTip("Micrometers(μm) per pixel, *.tif file")
        label.setFont(self.medfont)
        self.segBoxG.addWidget(label, b0, 0, 1, 4)

        self.pixTomicro = QLineEdit()
        self.pixTomicro.setText("0.0")
        self.pixTomicro.editingFinished.connect(self.update_px_to_mm)
        self.pixTomicro.setFixedWidth(70)
        self.segBoxG.addWidget(self.pixTomicro, b0, 4, 1, 2)

        b0 += 1
        # choose channel
        self.ChannelChoose = [QComboBox(), QComboBox()]
        self.ChannelChoose[0].addItems(["0: gray", "1: red", "2: green", "3: blue"])
        self.ChannelChoose[1].addItems(["0: none", "1: red", "2: green", "3: blue"])
        cstr = ["chan to segment:", "chan2 (optional): "]
        for i in range(2):
            self.ChannelChoose[i].setFont(self.medfont)
            label = QLabel(cstr[i])
            label.setFont(self.medfont)
            if i == 0:
                label.setToolTip(
                    "this is the channel in which the cytoplasm or nuclei exist that you want to segment"
                )
                self.ChannelChoose[i].setToolTip(
                    "this is the channel in which the cytoplasm or nuclei exist that you want to segment"
                )
            else:
                label.setToolTip(
                    "if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option"
                )
                self.ChannelChoose[i].setToolTip(
                    "if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option"
                )
            self.segBoxG.addWidget(label, b0 + i, 0, 1, 4)
            self.segBoxG.addWidget(self.ChannelChoose[i], b0 + i, 4, 1, 5)

        b0 += 2

        # use GPU
        self.useGPU = QCheckBox("use GPU")
        self.useGPU.setToolTip(
            "if you have specially installed the <i>cuda</i> version of torch, then you can activate this"
        )
        self.useGPU.setFont(self.medfont)
        self.check_gpu()
        self.segBoxG.addWidget(self.useGPU, b0, 0, 1, 3)

        # compute segmentation with general models
        self.net_text = ["run cyto3"]
        nett = ["cellpose super-generalist model"]

        # label = QLabel("Run:")
        # label.setFont(self.boldfont)
        # label.setFont(self.medfont)
        # self.segBoxG.addWidget(label, b0, 0, 1, 2)
        self.StyleButtons = []
        jj = 4
        for j in range(len(self.net_text)):
            self.StyleButtons.append(
                guiparts.ModelButton(self, self.net_text[j], self.net_text[j])
            )
            w = 5
            self.segBoxG.addWidget(self.StyleButtons[-1], b0, jj, 1, w)
            jj += w
            # self.StyleButtons[-1].setFixedWidth(140)
            self.StyleButtons[-1].setToolTip(nett[j])

        b0 += 1
        self.roi_count = QLabel("0 ROIs")
        self.roi_count.setFont(self.boldfont)
        self.roi_count.setAlignment(QtCore.Qt.AlignLeft)
        self.segBoxG.addWidget(self.roi_count, b0, 0, 1, 4)

        self.progress = QProgressBar(self)
        self.segBoxG.addWidget(self.progress, b0, 4, 1, 5)

        b0 += 1
        self.segaBox = QCollapsible("additional settings")
        self.segaBox.setFont(self.medfont)
        self.segaBox._toggle_btn.setFont(self.medfont)
        self.segaBoxG = QGridLayout()
        _content = QWidget()
        _content.setLayout(self.segaBoxG)
        _content.setMaximumHeight(0)
        _content.setMinimumHeight(0)
        # _content.layout().setContentsMargins(QtCore.QMargins(0, -20, -20, -20))
        self.segaBox.setContent(_content)
        self.segBoxG.addWidget(self.segaBox, b0, 0, 1, 9)

        b0 = 0
        # post-hoc paramater tuning
        label = QLabel("flow\nthreshold:")
        label.setToolTip(
            "threshold on flow error to accept a mask (set higher to get more cells, e.g. in range from (0.1, 3.0), OR set to 0.0 to turn off so no cells discarded);\n press enter to recompute if model already run"
        )
        label.setFont(self.medfont)
        self.segaBoxG.addWidget(label, b0, 0, 1, 2)
        self.flow_threshold = QLineEdit()
        self.flow_threshold.setText("0.4")
        self.flow_threshold.returnPressed.connect(self.compute_cprob)
        self.flow_threshold.setFixedWidth(40)
        self.flow_threshold.setFont(self.medfont)
        self.segaBoxG.addWidget(self.flow_threshold, b0, 2, 1, 2)
        self.flow_threshold.setToolTip(
            "threshold on flow error to accept a mask (set higher to get more cells, e.g. in range from (0.1, 3.0), OR set to 0.0 to turn off so no cells discarded);\n press enter to recompute if model already run"
        )

        label = QLabel("cellprob\nthreshold:")
        label.setToolTip(
            "threshold on cellprob output to seed cell masks (set lower to include more pixels or higher to include fewer, e.g. in range from (-6, 6)); \n press enter to recompute if model already run"
        )
        label.setFont(self.medfont)
        self.segaBoxG.addWidget(label, b0, 4, 1, 2)
        self.cellprob_threshold = QLineEdit()
        self.cellprob_threshold.setText("0.0")
        self.cellprob_threshold.returnPressed.connect(self.compute_cprob)
        self.cellprob_threshold.setFixedWidth(40)
        self.cellprob_threshold.setFont(self.medfont)
        self.cellprob_threshold.setToolTip(
            "threshold on cellprob output to seed cell masks (set lower to include more pixels or higher to include fewer, e.g. in range from (-6, 6)); \n press enter to recompute if model already run"
        )
        self.segaBoxG.addWidget(self.cellprob_threshold, b0, 6, 1, 2)

        b0 += 1
        label = QLabel("norm percentiles:")
        label.setToolTip(
            "sets normalization percentiles for segmentation and denoising\n(pixels at lower percentile set to 0.0 and at upper set to 1.0 for network)"
        )
        label.setFont(self.medfont)
        self.segaBoxG.addWidget(label, b0, 0, 1, 8)

        b0 += 1
        self.norm_vals = [1.0, 99.0]
        self.norm_edits = []
        labels = ["lower", "upper"]
        tooltips = [
            "pixels at this percentile set to 0 (default 1.0)",
            "pixels at this percentile set to 1  (default 99.0)",
        ]
        for p in range(2):
            label = QLabel(f"{labels[p]}:")
            label.setToolTip(tooltips[p])
            label.setFont(self.medfont)
            self.segaBoxG.addWidget(label, b0, 4 * (p % 2), 1, 2)
            self.norm_edits.append(QLineEdit())
            self.norm_edits[p].setText(str(self.norm_vals[p]))
            self.norm_edits[p].setFixedWidth(40)
            self.norm_edits[p].setFont(self.medfont)
            self.segaBoxG.addWidget(self.norm_edits[p], b0, 4 * (p % 2) + 2, 1, 2)
            self.norm_edits[p].setToolTip(tooltips[p])

        b0 += 1
        label = QLabel("niter dynamics:")
        label.setFont(self.medfont)
        label.setToolTip(
            "number of iterations for dynamics (0 uses default based on diameter); use 2000 for bacteria"
        )
        self.segaBoxG.addWidget(label, b0, 0, 1, 4)
        self.niter = QLineEdit()
        self.niter.setText("0")
        self.niter.setFixedWidth(40)
        self.niter.setFont(self.medfont)
        self.niter.setToolTip(
            "number of iterations for dynamics (0 uses default based on diameter); use 2000 for bacteria"
        )
        self.segaBoxG.addWidget(self.niter, b0, 4, 1, 2)

        b += 1
        b0 = 0
        self.modelBox = QGroupBox("Other models")
        self.modelBoxG = QGridLayout()
        self.modelBox.setLayout(self.modelBoxG)
        self.l0.addWidget(self.modelBox, b, 0, 1, 9)
        self.modelBox.setFont(self.boldfont)
        # choose models
        self.ModelChooseC = QComboBox()
        self.ModelChooseC.setFont(self.medfont)
        current_index = 0
        self.ModelChooseC.addItems(["custom models"])
        if len(self.model_strings) > 0:
            self.ModelChooseC.addItems(self.model_strings)
        self.ModelChooseC.setFixedWidth(175)
        self.ModelChooseC.setCurrentIndex(current_index)
        tipstr = 'add or train your own models in the "Models" file menu and choose model here'
        self.ModelChooseC.setToolTip(tipstr)
        self.ModelChooseC.activated.connect(lambda: self.model_choose(custom=True))
        self.modelBoxG.addWidget(self.ModelChooseC, b0, 0, 1, 8)

        # compute segmentation w/ custom model
        self.ModelButtonC = QPushButton("run")
        self.ModelButtonC.setFont(self.medfont)
        self.ModelButtonC.setFixedWidth(35)
        self.ModelButtonC.clicked.connect(
            lambda: self.compute_segmentation(custom=True)
        )
        self.modelBoxG.addWidget(self.ModelButtonC, b0, 8, 1, 1)
        self.ModelButtonC.setEnabled(False)

        self.net_names = [
            "nuclei",
            "cyto2_cp3",
            "tissuenet_cp3",
            "livecell_cp3",
            "yeast_PhC_cp3",
            "yeast_BF_cp3",
            "bact_phase_cp3",
            "bact_fluor_cp3",
            "deepbacs_cp3",
            "cyto",
            "cyto2",
            "CPx",
        ]

        nett = [
            "nuclei",
            "cellpose (cyto2_cp3)",
            "tissuenet_cp3",
            "livecell_cp3",
            "yeast_PhC_cp3",
            "yeast_BF_cp3",
            "bact_phase_cp3",
            "bact_fluor_cp3",
            "deepbacs_cp3",
            "cyto",
            "cyto2",
            "CPx (from Cellpose2)",
        ]
        b0 += 1
        self.ModelChooseB = QComboBox()
        self.ModelChooseB.setFont(self.medfont)
        self.ModelChooseB.addItems(["dataset-specific models"])
        self.ModelChooseB.addItems(nett)
        self.ModelChooseB.setFixedWidth(175)
        tipstr = "dataset-specific models"
        self.ModelChooseB.setToolTip(tipstr)
        self.ModelChooseB.activated.connect(lambda: self.model_choose(custom=False))
        self.modelBoxG.addWidget(self.ModelChooseB, b0, 0, 1, 8)

        # compute segmentation w/ cp model
        self.ModelButtonB = QPushButton("run")
        self.ModelButtonB.setFont(self.medfont)
        self.ModelButtonB.setFixedWidth(35)
        self.ModelButtonB.clicked.connect(
            lambda: self.compute_segmentation(custom=False)
        )
        self.modelBoxG.addWidget(self.ModelButtonB, b0, 8, 1, 1)
        self.ModelButtonB.setEnabled(False)

        b += 1
        self.denoiseBox = QGroupBox("Image restoration")
        self.denoiseBox.setFont(self.boldfont)
        self.denoiseBoxG = QGridLayout()
        self.denoiseBox.setLayout(self.denoiseBoxG)
        self.l0.addWidget(self.denoiseBox, b, 0, 1, 9)

        b0 = 0

        # DENOISING
        self.DenoiseButtons = []
        nett = [
            "clear restore/filter",
            "filter image (settings below)",
            "denoise (please set cell diameter first)",
            "deblur (please set cell diameter first)",
            "upsample to 30. diameter (cyto3) or 17. diameter (nuclei) (please set cell diameter first) (disabled in 3D)",
            "one-click model trained to denoise+deblur+upsample (please set cell diameter first)",
        ]
        self.denoise_text = [
            "none",
            "filter",
            "denoise",
            "deblur",
            "upsample",
            "one-click",
        ]
        self.restore = None
        self.ratio = 1.0
        jj = 0
        w = 3
        for j in range(len(self.denoise_text)):
            self.DenoiseButtons.append(
                guiparts.DenoiseButton(self, self.denoise_text[j])
            )
            self.denoiseBoxG.addWidget(self.DenoiseButtons[-1], b0, jj, 1, w)
            self.DenoiseButtons[-1].setFixedWidth(75)
            self.DenoiseButtons[-1].setToolTip(nett[j])
            self.DenoiseButtons[-1].setFont(self.medfont)
            b0 += 1 if j % 2 == 1 else 0
            jj = 0 if j % 2 == 1 else jj + w

        # b0+=1
        self.save_norm = QCheckBox("save restored/filtered image")
        self.save_norm.setFont(self.medfont)
        self.save_norm.setToolTip("save restored/filtered image in _seg.npy file")
        self.save_norm.setChecked(True)
        # self.denoiseBoxG.addWidget(self.save_norm, b0, 0, 1, 8)

        b0 -= 3
        label = QLabel("restore-dataset:")
        label.setToolTip(
            "choose dataset and click [denoise], [deblur], [upsample], or [one-click]"
        )
        label.setFont(self.medfont)
        self.denoiseBoxG.addWidget(label, b0, 6, 1, 3)

        b0 += 1
        self.DenoiseChoose = QComboBox()
        self.DenoiseChoose.setFont(self.medfont)
        self.DenoiseChoose.addItems(["cyto3", "cyto2", "nuclei"])
        self.DenoiseChoose.setFixedWidth(85)
        tipstr = "choose model type and click [denoise], [deblur], or [upsample]"
        self.DenoiseChoose.setToolTip(tipstr)
        self.denoiseBoxG.addWidget(self.DenoiseChoose, b0, 6, 1, 3)

        b0 += 2
        # FILTERING
        self.filtBox = QCollapsible("custom filter settings")
        self.filtBox._toggle_btn.setFont(self.medfont)
        self.filtBoxG = QGridLayout()
        _content = QWidget()
        _content.setLayout(self.filtBoxG)
        _content.setMaximumHeight(0)
        _content.setMinimumHeight(0)
        # _content.layout().setContentsMargins(QtCore.QMargins(0, -20, -20, -20))
        self.filtBox.setContent(_content)
        self.denoiseBoxG.addWidget(self.filtBox, b0, 0, 1, 9)

        self.filt_vals = [0.0, 0.0, 0.0, 0.0]
        self.filt_edits = []
        labels = [
            "sharpen\nradius",
            "smooth\nradius",
            "tile_norm\nblocksize",
            "tile_norm\nsmooth3D",
        ]
        tooltips = [
            "set size of surround-subtraction filter for sharpening image",
            "set size of gaussian filter for smoothing image",
            "set size of tiles to use to normalize image",
            "set amount of smoothing of normalization values across planes",
        ]

        for p in range(4):
            label = QLabel(f"{labels[p]}:")
            label.setToolTip(tooltips[p])
            label.setFont(self.medfont)
            self.filtBoxG.addWidget(label, b0 + p // 2, 4 * (p % 2), 1, 2)
            self.filt_edits.append(QLineEdit())
            self.filt_edits[p].setText(str(self.filt_vals[p]))
            self.filt_edits[p].setFixedWidth(40)
            self.filt_edits[p].setFont(self.medfont)
            self.filtBoxG.addWidget(
                self.filt_edits[p], b0 + p // 2, 4 * (p % 2) + 2, 1, 2
            )
            self.filt_edits[p].setToolTip(tooltips[p])

        b0 += 3
        self.norm3D_cb = QCheckBox("norm3D")
        self.norm3D_cb.setFont(self.medfont)
        self.norm3D_cb.setChecked(True)
        self.norm3D_cb.setToolTip("run same normalization across planes")
        self.filtBoxG.addWidget(self.norm3D_cb, b0, 0, 1, 3)

        self.invert_cb = QCheckBox("invert")
        self.invert_cb.setFont(self.medfont)
        self.invert_cb.setToolTip("invert image")
        self.filtBoxG.addWidget(self.invert_cb, b0, 3, 1, 3)

        ## NEW
        b += 1
        b0 += 1
        self.MB = QGroupBox("Metrics")
        self.MB.setFont(self.boldfont)
        self.MB.setStyleSheet(
            "QGroupBox { border: 1px solid white; color:white; padding: 10px 0px;}"
        )
        self.MBg = QGridLayout()
        self.MB.setLayout(self.MBg)
        self.currentImageMask = ""
        self.indexCytoMask = -1
        self.indexNucleusMask = -1

        # select metrics to calculate
        self.calcSize = False
        self.SMCheckBox = QCheckBox("Area")
        self.SMCheckBox.setStyleSheet("color: rgb(190,190,190);")
        self.SMCheckBox.setFont(self.medfont)
        self.SMCheckBox.setChecked(False)
        self.SMCheckBox.setEnabled(False)
        self.SMCheckBox.toggled.connect(self.toggle_masks)
        tipstr = "Area of the cell in μm2"
        self.SMCheckBox.setToolTip(tipstr)
        self.MBg.addWidget(self.SMCheckBox, 0, 0, 1, 7)

        self.calcRound = False
        self.RMCheckBox = QCheckBox("Roundness")
        self.RMCheckBox.setStyleSheet("color: rgb(190,190,190);")
        self.RMCheckBox.setFont(self.medfont)
        self.RMCheckBox.setChecked(False)
        self.RMCheckBox.setEnabled(False)
        self.RMCheckBox.toggled.connect(self.toggle_masks)
        tipstr = "Closer to 1 means more like a circle"
        self.RMCheckBox.setToolTip(tipstr)
        self.MBg.addWidget(self.RMCheckBox, 0, 5, 1, 7)

        self.calcRatio = False
        self.RTCheckBox = QCheckBox("Ratio")
        self.RTCheckBox.setStyleSheet("color: rgb(190,190,190);")
        self.RTCheckBox.setFont(self.medfont)
        self.RTCheckBox.setChecked(False)
        self.RTCheckBox.setEnabled(False)
        self.RTCheckBox.toggled.connect(self.toggle_masks)
        tipstr = "Ratio between cyto and nucleus"
        self.RTCheckBox.setToolTip(tipstr)
        self.MBg.addWidget(self.RTCheckBox, 1, 0, 1, 7)

        self.calcVoronoi = False
        self.VDCheckBox = QCheckBox("Voronoi")
        self.VDCheckBox.setStyleSheet("color: rgb(190,190,190);")
        self.VDCheckBox.setFont(self.medfont)
        self.VDCheckBox.setChecked(False)
        self.VDCheckBox.setEnabled(False)
        self.VDCheckBox.toggled.connect(self.toggle_masks)
        tipstr = "Ratio between cyto and nucleus"
        self.VDCheckBox.setToolTip(tipstr)
        self.MBg.addWidget(self.VDCheckBox, 1, 5, 1, 7)

        # calculate the selected metrics
        self.CalculateButton = QPushButton("calculate")
        self.CalculateButton.clicked.connect(
            lambda: self.features_class.calculate_metrics(self)
        )
        self.MBg.addWidget(self.CalculateButton, 0, 10, 1, 2)
        self.CalculateButton.setEnabled(False)
        # self.CalculateButton.setStyleSheet(self.styleInactive)

        self.l0.addWidget(self.MB, b, 0, 1, 9)
        ##

        b += 1
        self.l0.addWidget(QLabel(""), b, 0, 1, 9)
        self.l0.setRowStretch(b, 100)

        b += 1
        # scale toggle
        self.scale_on = True
        self.ScaleOn = QCheckBox("scale disk on")
        self.ScaleOn.setFont(self.medfont)
        self.ScaleOn.setStyleSheet("color: rgb(150,50,150);")
        self.ScaleOn.setChecked(True)
        self.ScaleOn.setToolTip("see current diameter as red disk at bottom")
        self.ScaleOn.toggled.connect(self.toggle_scale)
        self.l0.addWidget(self.ScaleOn, b, 0, 1, 5)

        return b

    def update_px_to_mm(self):
        """
        Update the pixel-to-millimeter conversion value.

            This method retrieves the value from the pixTomicro text field, converts it to a float,
            and updates the px_to_mm attribute with the new value.

            Parameters:
                None

            Returns:
                None
        """
        self.px_to_mm = float(self.pixTomicro.text())

    def level_change(self, r):
        """
        Update saturation levels based on the slider value.

            This method adjusts the saturation levels for a given color channel
            based on the current value of the corresponding slider. If automatic
            adjustments are not enabled, it applies the saturation value across
            all positions for that channel. Finally, it refreshes the plot to
            reflect the changes.

            Args:
                r: The index of the color channel which can be "red", "green", or "blue".

            Returns:
                None: This method does not return any value.
        """
        r = ["red", "green", "blue"].index(r)
        if self.loaded:
            sval = self.sliders[r].value()
            self.saturation[r][self.currentZ] = sval
            if not self.autobtn.isChecked():
                for r in range(3):
                    for i in range(len(self.saturation[r])):
                        self.saturation[r][i] = self.saturation[r][self.currentZ]
            self.update_plot()

    def keyPressEvent(self, event):
        """
        Handles key press events for controlling the application's behavior.

            This method responds to key press events to perform various actions
            such as navigating between images, changing colors, and modifying
            the brush settings. It also updates the plot based on the current
            state and inputs.

            Args:
                event: The key press event that contains information about the key
                    that was pressed.

            Returns:
                None
        """
        if self.loaded:
            if not (
                event.modifiers()
                & (
                    QtCore.Qt.ControlModifier
                    | QtCore.Qt.ShiftModifier
                    | QtCore.Qt.AltModifier
                )
                or self.in_stroke
            ):
                updated = False
                if len(self.current_point_set) > 0:
                    if event.key() == QtCore.Qt.Key_Return:
                        self.add_set()
                else:
                    nviews = self.ViewDropDown.count() - 1
                    nviews += int(
                        self.ViewDropDown.model()
                        .item(self.ViewDropDown.count() - 1)
                        .isEnabled()
                    )
                    if event.key() == QtCore.Qt.Key_X:
                        self.MCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_Z:
                        self.OCheckBox.toggle()
                    if (
                        event.key() == QtCore.Qt.Key_Left
                        or event.key() == QtCore.Qt.Key_A
                    ):
                        self.get_prev_image()
                    elif (
                        event.key() == QtCore.Qt.Key_Right
                        or event.key() == QtCore.Qt.Key_D
                    ):
                        self.get_next_image()
                    elif event.key() == QtCore.Qt.Key_PageDown:
                        self.view = (self.view + 1) % (nviews)
                        self.ViewDropDown.setCurrentIndex(self.view)
                    elif event.key() == QtCore.Qt.Key_PageUp:
                        self.view = (self.view - 1) % (nviews)
                        self.ViewDropDown.setCurrentIndex(self.view)

                # can change background or stroke size if cell not finished
                if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_W:
                    self.color = (self.color - 1) % (6)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif (
                    event.key() == QtCore.Qt.Key_Down or event.key() == QtCore.Qt.Key_S
                ):
                    self.color = (self.color + 1) % (6)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_R:
                    if self.color != 1:
                        self.color = 1
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_G:
                    if self.color != 2:
                        self.color = 2
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_B:
                    if self.color != 3:
                        self.color = 3
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif (
                    event.key() == QtCore.Qt.Key_Comma
                    or event.key() == QtCore.Qt.Key_Period
                ):
                    count = self.BrushChoose.count()
                    gci = self.BrushChoose.currentIndex()
                    if event.key() == QtCore.Qt.Key_Comma:
                        gci = max(0, gci - 1)
                    else:
                        gci = min(count - 1, gci + 1)
                    self.BrushChoose.setCurrentIndex(gci)
                    self.brush_choose()
                if not updated:
                    self.update_plot()
        if event.key() == QtCore.Qt.Key_Minus or event.key() == QtCore.Qt.Key_Equal:
            self.p0.keyPressEvent(event)

    def autosave_on(self):
        """
        Toggle autosave feature based on checkbox status.

            This method sets the autosave attribute to True if the associated
            checkbox is checked, and sets it to False otherwise.

            Parameters:
                None

            Returns:
                None
        """
        if self.SCheckBox.isChecked():
            self.autosave = True
        else:
            self.autosave = False

    def check_gpu(self, torch=True):
        """
        Check the availability of a GPU and update the GUI accordingly.

            This method checks if a GPU is available for use with PyTorch.
            It updates the state of a GUI element that indicates whether the GPU can be used.
            If a GPU is available, the GUI element is enabled and checked;
            otherwise, it is disabled and styled to reflect its unavailability.

            Attributes:
                self.useGPU: The GUI element that represents the GPU usage status.

            Returns:
                None: This method does not return any value.
        """
        # also decide whether or not to use torch
        self.useGPU.setChecked(False)
        self.useGPU.setEnabled(False)
        if core.use_gpu(use_torch=True):
            self.useGPU.setEnabled(True)
            self.useGPU.setChecked(True)
        else:
            self.useGPU.setStyleSheet("color: rgb(80,80,80);")

    def get_channels(self):
        """
        Retrieve the currently selected channels from the GUI and adjust
            them based on specific application logic.

            This method collects the currently selected indices from the channel
            selection components of the GUI. It checks conditions related to the
            current model and the number of available channels, making adjustments
            as necessary to ensure valid channel selections.

            If the current model is set to "nuclei", the second channel index is
            overridden. Additionally, if only one channel is available, both
            selected channels will be set to zero. Warnings are printed if the
            user attempts to select an invalid channel configuration given the
            number of channels available.

            Returns:
                A list of integers representing the adjusted indices of the selected channels.
        """
        channels = [
            self.ChannelChoose[0].currentIndex(),
            self.ChannelChoose[1].currentIndex(),
        ]
        if hasattr(self, "current_model"):
            if self.current_model == "nuclei":
                channels[1] = 0
        if channels[0] == 0:
            channels[1] = 0
        if self.nchan == 1:
            channels = [0, 0]
        elif self.nchan == 2:
            if channels[0] == 3:
                channels[0] = 1 if channels[1] != 1 else 2
                print(
                    f"GUI_WARNING: only two channels in image, cannot use blue channel, changing channels"
                )
            if channels[1] == 3:
                channels[1] = 1 if channels[0] != 1 else 2
                print(
                    f"GUI_WARNING: only two channels in image, cannot use blue channel, changing channels"
                )
        self.ChannelChoose[0].setCurrentIndex(channels[0])
        self.ChannelChoose[1].setCurrentIndex(channels[1])
        return channels

    def model_choose(self, custom=False):
        """
        Select and initialize a model based on user input.

            This method retrieves the currently selected model from the GUI and initializes it. If a custom model is selected, it uses the text from a specific dropdown; otherwise, it uses a predefined list of model names. The method also updates the GUI with the model's diameter.

            Args:
                custom: A flag indicating whether a custom model is being selected. If True, it retrieves the model name from a custom dropdown.

            Returns:
                None: This method does not return a value, but it updates the state of the model and the GUI.
        """
        index = (
            self.ModelChooseC.currentIndex()
            if custom
            else self.ModelChooseB.currentIndex()
        )
        if index > 0:
            if custom:
                model_name = self.ModelChooseC.currentText()
            else:
                model_name = self.net_names[index - 1]
            print(f"GUI_INFO: selected model {model_name}, loading now")
            self.initialize_model(model_name=model_name, custom=custom)
            self.diameter = self.model.diam_labels
            self.Diameter.setText("%0.2f" % self.diameter)
            print(
                f"GUI_INFO: diameter set to {self.diameter: 0.2f} (but can be changed)"
            )

    def calibrate_size(self):
        """
        Calibrate the size of cells in the current stack using a predefined model.

            This method initializes the model, evaluates cell diameters using the model
            on the current stack and updates the display with the estimated cell diameter.
            It ensures that the diameter values are at least 5.0 pixels. The progress is
            also updated to reflect completion.

            Args:
                None

            Returns:
                None
        """
        self.initialize_model(model_name="cyto3")
        diams, _ = self.model.sz.eval(
            self.stack[self.currentZ].copy(),
            channels=self.get_channels(),
            progress=self.progress,
        )
        diams = np.maximum(5.0, diams)
        self.logger.info(
            "estimated diameter of cells using %s model = %0.1f pixels"
            % (self.current_model, diams)
        )
        self.Diameter.setText("%0.1f" % diams)
        self.diameter = diams
        self.update_scale()
        self.progress.setValue(100)

    def toggle_scale(self):
        """
        Toggles the visibility of the scale item in the plot.

            This method adds or removes the scale item from the plot depending on its current state.
            If the scale is currently on, it will be removed and the state will be set to off.
            Conversely, if the scale is off, it will be added to the plot and the state will be set to on.

            Attributes:
                scale_on: A boolean indicating whether the scale is currently displayed.

            Returns:
                None
        """
        if self.scale_on:
            self.p0.removeItem(self.scale)
            self.scale_on = False
        else:
            self.p0.addItem(self.scale)
            self.scale_on = True

    def enable_buttons(self):
        """
        Enables various buttons in the user interface based on the model and current state.

            This method checks the availability of model strings and alters the enabled state of
            several buttons accordingly. It also adjusts the enabled state of sliders, depending
            on the number of channels, and updates the plot and window title.

            Parameters:
              None

            Returns:
              None
        """
        if len(self.model_strings) > 0:
            self.ModelButtonC.setEnabled(True)
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(True)
        for i in range(len(self.DenoiseButtons)):
            self.DenoiseButtons[i].setEnabled(True)
        if self.load_3D:
            self.DenoiseButtons[-2].setEnabled(False)
        self.ModelButtonB.setEnabled(True)
        self.SizeButton.setEnabled(True)
        self.newmodel.setEnabled(True)
        self.loadMasks.setEnabled(True)
        self.keepMask.setEnabled(False)  # New
        self.saveMasks.setEnabled(False)  # New

        for n in range(self.nchan):
            self.sliders[n].setEnabled(True)
        for n in range(self.nchan, 3):
            self.sliders[n].setEnabled(True)

        self.toggle_mask_ops()

        self.update_plot()
        self.setWindowTitle(self.filename)

    def disable_buttons_removeROIs(self):
        """
        Disable various buttons in the UI for removing ROIs.

            This method disables specific buttons related to model and style operations
            in the user interface, indicating that removing ROIs is in progress.
            It ensures that the user cannot interact with these buttons while the
            removal operation is active, except for the buttons that allow
            confirming or canceling the deletion of multiple ROIs.

            Parameters:
                None

            Returns:
                None
        """
        if len(self.model_strings) > 0:
            self.ModelButtonC.setEnabled(False)
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(False)
        self.ModelButtonB.setEnabled(False)
        self.SizeButton.setEnabled(False)
        self.newmodel.setEnabled(False)
        self.loadMasks.setEnabled(False)
        self.saveSet.setEnabled(False)
        self.savePNG.setEnabled(False)
        self.saveFlows.setEnabled(False)
        self.saveOutlines.setEnabled(False)
        self.saveROIs.setEnabled(False)

        self.MakeDeletionRegionButton.setEnabled(False)
        self.DeleteMultipleROIButton.setEnabled(False)
        self.DoneDeleteMultipleROIButton.setEnabled(True)
        self.CancelDeleteMultipleROIButton.setEnabled(True)

    def toggle_mask_ops(self):
        """
        Toggle various operations related to mask management.

            This method updates the layer by calling the update_layer method,
            and toggles the saving and removal operations by calling the
            respective methods.

            Parameters:
                None

            Returns:
                None
        """
        self.update_layer()
        self.toggle_saving()
        self.toggle_removals()

    def toggle_saving(self):
        """
        Toggle the enabling of saving options based on the number of cells.

            This method enables or disables various save options depending on the
            current number of cells. If the number of cells is greater than zero,
            the saving options are enabled; otherwise, they are disabled.

            Parameters:
                None

            Returns:
                None
        """
        if self.ncells > 0:
            self.saveSet.setEnabled(True)
            self.savePNG.setEnabled(True)
            self.saveFlows.setEnabled(True)
            self.saveOutlines.setEnabled(True)
            self.saveROIs.setEnabled(True)
        else:
            self.saveSet.setEnabled(False)
            self.savePNG.setEnabled(False)
            self.saveFlows.setEnabled(False)
            self.saveOutlines.setEnabled(False)
            self.saveROIs.setEnabled(False)

    def toggle_removals(self):
        """
        Toggle the enabled state of buttons related to removals based on the number of cells.

            This method checks the number of cells (ncells) and enables or disables various buttons
            in the user interface accordingly. If the number of cells is greater than zero, the buttons
            related to clearing, removing cells, undo actions, and deletion regions are enabled; otherwise,
            they are disabled.

            Parameters:
                None

            Returns:
                None
        """
        if self.ncells > 0:
            self.ClearButton.setEnabled(True)
            self.remcell.setEnabled(True)
            self.undo.setEnabled(True)
            self.MakeDeletionRegionButton.setEnabled(True)
            self.DeleteMultipleROIButton.setEnabled(True)
            self.DoneDeleteMultipleROIButton.setEnabled(False)
            self.CancelDeleteMultipleROIButton.setEnabled(False)
        else:
            self.ClearButton.setEnabled(False)
            self.remcell.setEnabled(False)
            self.undo.setEnabled(False)
            self.MakeDeletionRegionButton.setEnabled(False)
            self.DeleteMultipleROIButton.setEnabled(False)
            self.DoneDeleteMultipleROIButton.setEnabled(False)
            self.CancelDeleteMultipleROIButton.setEnabled(False)

    def remove_action(self):
        """
        Removes the currently selected cell if a selection exists.

            If a cell is currently selected (indicated by the `selected` attribute being greater than 0),
            this method will invoke the `remove_cell` method to delete that cell.

            Parameters:
                None

            Returns:
                None
        """
        if self.selected > 0:
            self.remove_cell(self.selected)

    def undo_action(self):
        """
        Reverses the most recent action taken in the application.

            This method checks if there are any strokes recorded. If the last stroke's Z-coordinate matches
            the current Z-coordinate, it removes the last stroke. If no matching stroke is found and there
            are remaining cells, it removes the most recently added cell.

            Parameters:
                None

            Returns:
                None
        """
        if len(self.strokes) > 0 and self.strokes[-1][0][0] == self.currentZ:
            self.remove_stroke()
        else:
            # remove previous cell
            if self.ncells > 0:
                self.remove_cell(self.ncells)

    def undo_remove_action(self):
        """
        Reverts the last remove action on a cell.

            This method restores the most recently removed cell to its previous state,
            effectively cancelling the last removal action performed on it.

            Parameters:
                None

            Returns:
                None
        """
        self.undo_remove_cell()

    def get_files(self):
        """
        Retrieves image files from a specified folder and identifies the index of a specific file.

            This method scans a folder for image files that match a predefined mask filter,
            retrieves the filenames of these images, and determines the index of the current
            file within the list of retrieved images.

            Returns:
                A tuple containing:
                    - images: A list of paths to the image files found in the folder.
                    - idx: The index of the current file in the list of image filenames.
        """
        folder = os.path.dirname(self.filename)
        mask_filter = "_masks"
        images = get_image_files(folder, mask_filter)
        fnames = [os.path.split(images[k])[-1] for k in range(len(images))]
        f0 = os.path.split(self.filename)[-1]
        idx = np.nonzero(np.array(fnames) == f0)[0][0]
        return images, idx

    def get_prev_image(self):
        """
        Retrieve the previous image in a cyclic manner.

            This method fetches the previous image based on the current index.
            If the current index is the first image, it wraps around to the last image.

            Returns:
                The loaded image corresponding to the previous index.
        """
        images, idx = self.get_files()
        idx = (idx - 1) % len(images)
        io._load_image(self, filename=images[idx])

    def get_next_image(self, load_seg=True):
        """
        Retrieve the next image in the sequence and loads it.

            This method retrieves the next image from a predefined list of images.
            If the current index exceeds the length of the image list, it wraps around
            to the beginning. The method also allows for loading segmentation information
            associated with the image if specified.

            Args:
                load_seg: Indicates whether to load the segmentation associated with the image.

            Returns:
                None: This method does not return a value but performs the loading of the image.
        """
        images, idx = self.get_files()
        idx = (idx + 1) % len(images)
        io._load_image(self, filename=images[idx], load_seg=load_seg)

    def dragEnterEvent(self, event):
        """
        Handle the drag enter event for the widget.

            This method processes the drag enter event by checking if the
            dragged data contains URLs. If URLs are detected, the event is
            accepted; otherwise, it is ignored.

            Args:
                event: The event object containing information about the
                       drag-and-drop operation.

            Returns:
                None
        """
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        Handles the drop event for loading files.

            This method processes the drop event, retrieves the file path from
            the dropped items, and loads the appropriate data based on the
            file extension. If the file is a NumPy (.npy) file, it loads the
            segmentation data; otherwise, it loads the image data.

            Args:
                event: The event object that contains the MIME data and
                       information about the dropped files.

            Returns:
                None
        """
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if os.path.splitext(files[0])[-1] == ".npy":
            io._load_seg(self, filename=files[0], load_3D=self.load_3D)
        else:
            io._load_image(self, filename=files[0], load_seg=True, load_3D=self.load_3D)

    def toggle_masks(self):
        """
        Toggles the visibility of masks and outlines based on the state of checkboxes.

            This method checks the state of various checkboxes to determine whether to
            enable or disable masks, outlines, size calculations, round calculations,
            ratio calculations, and Voronoi calculations. It also updates the visual
            representation of elements accordingly based on the current settings.

            Parameters:
                None

            Returns:
                None
        """
        if self.MCheckBox.isChecked():
            self.masksOn = True
        else:
            self.masksOn = False

        if self.OCheckBox.isChecked():
            self.outlinesOn = True
        else:
            self.outlinesOn = False

        if self.SMCheckBox.isChecked():
            self.calcSize = True
        else:
            self.calcSize = False

        if self.RMCheckBox.isChecked():
            self.calcRound = True
        else:
            self.calcRound = False

        if self.RTCheckBox.isChecked():
            self.calcRatio = True
        else:
            self.calcRatio = False

        if self.VDCheckBox.isChecked():
            self.calcVoronoi = True
        else:
            self.calcVoronoi = False

        if not self.masksOn and not self.outlinesOn:
            self.p0.removeItem(self.layer)
            self.layer_off = True
        else:
            if self.layer_off:
                self.p0.addItem(self.layer)
            self.draw_layer()
            self.update_layer()

        if self.loaded:
            self.update_plot()
            self.update_layer()

    def make_viewbox(self):
        """
        Creates and configures a viewbox for displaying images and drawing.

            This method initializes a viewbox with specific properties, adds image items,
            and sets up a drawing layer. The viewbox is designed to allow interaction with
            mouse events for drawing purposes while maintaining aspect ratio and other visual
            characteristics.

            Parameters:
                None

            Returns:
                None
        """
        self.p0 = guiparts.ViewBoxNoRightDrag(
            parent=self,
            lockAspect=True,
            name="plot1",
            border=[100, 100, 100],
            invertY=True,
        )
        self.p0.setCursor(QtCore.Qt.CrossCursor)
        self.brush_size = 3
        self.win.addItem(self.p0, 0, 0, rowspan=1, colspan=1)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.img = pg.ImageItem(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        self.layer = guiparts.ImageDraw(viewbox=self.p0, parent=self)
        self.layer.setLevels([0, 255])
        self.scale = pg.ImageItem(viewbox=self.p0, parent=self)
        self.scale.setLevels([0, 255])
        self.p0.scene().contextMenuItem = self.p0
        # self.p0.setMouseEnabled(x=False,y=False)
        self.Ly, self.Lx = 512, 512
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)

    def reset(self):
        """
        Resets the internal state of the object to its initial configuration.

            This method is responsible for clearing and resetting various attributes
            related to the object's state, including the selected index, channel
            configurations, image stack parameters, and interface settings. It also
            ensures that all modifications made to the data are reverted to their
            defaults and prepares the object for new input.

            Parameters:
                None

            Returns:
                None
        """
        # ---- start sets of points ---- #
        self.selected = 0
        self.nchan = 3
        self.loaded = False
        self.channel = [0, 1]
        self.current_point_set = []
        self.in_stroke = False
        self.strokes = []
        self.stroke_appended = True
        self.resize = False
        self.ncells = 0
        self.zdraw = []
        self.removed_cell = []
        self.cellcolors = np.array([255, 255, 255])[np.newaxis, :]

        # -- zero out image stack -- #
        self.opacity = 128  # how opaque masks should be
        self.outcolor = [200, 200, 255, 200]
        self.NZ, self.Ly, self.Lx = 1, 224, 224
        self.saturation = []
        for r in range(3):
            self.saturation.append([[0, 255] for n in range(self.NZ)])
            self.sliders[r].setValue([0, 255])
            self.sliders[r].setEnabled(False)
            self.sliders[r].show()
        self.currentZ = 0
        self.flows = [[], [], [], [], [[]]]
        # masks matrix
        # image matrix with a scale disk
        self.stack = np.zeros((1, self.Ly, self.Lx, 3))
        self.Lyr, self.Lxr = self.Ly, self.Lx
        self.Ly0, self.Lx0 = self.Ly, self.Lx
        self.radii = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
        self.layerz = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
        self.cellpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        self.outpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        if self.restore and "upsample" in self.restore:
            self.cellpix_resize = self.cellpix
            self.cellpix_orig = self.cellpix
            self.outpix_resize = self.cellpix
            self.outpix_orig = self.cellpix
        self.ismanual = np.zeros(0, "bool")

        # -- set menus to default -- #
        self.color = 0
        self.RGBDropDown.setCurrentIndex(self.color)
        self.view = 0
        self.ViewDropDown.setCurrentIndex(0)
        self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(False)
        self.delete_restore()

        self.clear_all()

        # self.update_plot()
        self.filename = []
        self.loaded = False
        self.recompute_masks = False

        self.deleting_multiple = False
        self.removing_cells_list = []
        self.removing_region = False
        self.remove_roi_obj = None

    def delete_restore(self):
        """delete restored imgs but don't reset settings"""
        if hasattr(self, "stack_filtered"):
            del self.stack_filtered
        if hasattr(self, "cellpix_orig"):
            self.cellpix = self.cellpix_orig.copy()
            self.outpix = self.outpix_orig.copy()
            del self.outpix_orig, self.outpix_resize
            del self.cellpix_orig, self.cellpix_resize

    def clear_restore(self):
        """delete restored imgs and reset settings"""
        print("GUI_INFO: clearing restored image")
        self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(False)
        if self.ViewDropDown.currentIndex() == self.ViewDropDown.count() - 1:
            self.ViewDropDown.setCurrentIndex(0)
        self.delete_restore()
        self.restore = None
        self.ratio = 1.0
        self.set_normalize_params(self.get_normalize_params())

    def brush_choose(self):
        """
        Selects the brush size based on the current index of the BrushChoose widget.

            This method sets the brush size property of the object by calculating it
            from the current index of the BrushChoose widget. It updates the drawing
            layer with the new brush size if a layer has been loaded.

            Args:
                None

            Returns:
                None
        """
        self.brush_size = self.BrushChoose.currentIndex() * 2 + 1
        if self.loaded:
            self.layer.setDrawKernel(kernel_size=self.brush_size)
            self.update_layer()

    def clear_all(self):
        """
        Clears all selections and resets the internal state of the object.

            This method resets various attributes related to selections and pixel values,
            initializing them to their default states. It handles the case where
            a restoration process may or may not involve upsampling, setting pixel arrays
            and colors accordingly.

            Parameters:
                None

            Returns:
                None
        """
        self.prev_selected = 0
        self.selected = 0
        if self.restore and "upsample" in self.restore:
            self.layerz = 0 * np.ones((self.Lyr, self.Lxr, 4), np.uint8)
            self.cellpix = np.zeros((self.NZ, self.Lyr, self.Lxr), np.uint16)
            self.outpix = np.zeros((self.NZ, self.Lyr, self.Lxr), np.uint16)
            self.cellpix_resize = self.cellpix.copy()
            self.outpix_resize = self.outpix.copy()
            self.cellpix_orig = np.zeros((self.NZ, self.Ly0, self.Lx0), np.uint16)
            self.outpix_orig = np.zeros((self.NZ, self.Ly0, self.Lx0), np.uint16)
        else:
            self.layerz = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
            self.cellpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)
            self.outpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)

        self.cellcolors = np.array([255, 255, 255])[np.newaxis, :]
        self.ncells = 0
        self.toggle_removals()
        self.update_scale()
        self.update_layer()

    def select_cell(self, idx):
        """
        Selects a cell based on the provided index and processes its pixel data.

            This method updates the selected cell, creates a mask of the corresponding pixels,
            and performs analysis on the selected cell's size and properties. It also generates
            and saves images of the processed mask before and after removing small labels.

            Args:
                idx: The index of the cell to be selected.

            Returns:
                None
        """
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected > 0:
            np.set_printoptions(threshold=sys.maxsize)

            slices = find_objects(self.cellpix[0].astype(int))
            si = slices[self.selected - 1]
            sr, sc = si
            # mask = (self.cellpix[0][sr, sc] == (self.selected)).astype(np.uint8)
            tmp_cellpix = np.copy(self.cellpix[0])
            tmp_cellpix[self.selected != self.cellpix[0]] = 0
            tmp_cellpix[self.selected == self.cellpix[0]] = 255

            # mask_shape = mask.shape
            # for i in range(0, mask_shape[0]):
            #     for j in range(0, mask_shape[1]):
            #         mask[i][j] = 255 if mask[i][j] > 0 else 0

            mask = tmp_cellpix.astype(np.uint8)

            mask = np.pad(mask, 1, mode="constant")
            im = Image.fromarray(mask)
            im.save("hola1.jpg")

            Zlabeled, Nlabels = ndimage.label(mask)
            label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
            for label, size in enumerate(label_size):
                print("label %s is %s pixels in size" % (label, size))

            # now remove the labels
            for label, size in enumerate(label_size):
                if size < 5:
                    mask[Zlabeled == label] = 0

            im = Image.fromarray(mask)
            im.save("hola2.jpg")

            labels = dip.Label(mask[:, :] > 0)
            msr = dip.MeasurementTool.Measure(
                labels,
                features=[
                    "Perimeter",
                    "SolidArea",
                    "Roundness",
                    "Circularity",
                    "Center",
                ],
            )
            print(msr)
            print("IDX: ", self.selected)
            print("Size in px: ", msr[1]["SolidArea"][0])
            print(
                "Size in μm: ", round(msr[1]["SolidArea"][0] * pow(self.px_to_mm, 2), 2)
            )

            z = self.currentZ
            self.layerz[self.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.opacity]
            )
            self.update_layer()

    def select_cell_multi(self, idx):
        """
        Selects multiple cells based on the provided index.

            This method updates the selected status of multiple cells within a layer.
            If the provided index is greater than zero, it modifies the layer's pixel values
            to indicate selection, setting the corresponding pixels to white with a specified opacity.

            Args:
                idx: The index representing the cell(s) to be selected. Must be greater than zero
                      to effectively update cell selection.

            Returns:
                None
        """
        if idx > 0:
            z = self.currentZ
            self.layerz[self.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.opacity]
            )
            self.update_layer()

    def unselect_cell(self):
        """
        Deselects the currently selected cell and updates the visual representation.

            This method clears the selection of the currently selected cell by resetting its
            visual properties and updating the layer. It ensures that the cell's appearance reflects
            its unselected state, and if outlines are enabled, the outline color is also updated.

            The method modifies the layerz and outlines based on the current state of the selected
            cell and applies the necessary opacity settings.

            Returns:
                None: This method does not return a value.
        """
        if self.selected > 0:
            idx = self.selected
            if idx < self.ncells + 1:
                z = self.currentZ
                self.layerz[self.cellpix[z] == idx] = np.append(
                    self.cellcolors[idx], self.opacity
                )
                if self.outlinesOn:
                    self.layerz[self.outpix[z] == idx] = np.array(self.outcolor).astype(
                        np.uint8
                    )
                    # [0,0,0,self.opacity])
                self.update_layer()
        self.selected = 0

    def unselect_cell_multi(self, idx):
        """
        Unselects multiple cells specified by their index.

            This method updates the color and opacity of cells in the current layer based on the provided index.
            If outlines are enabled, it also modifies the outline color for the specified cells.

            Args:
                idx: The index of the cells to unselect.

            Returns:
                None
        """
        z = self.currentZ
        self.layerz[self.cellpix[z] == idx] = np.append(
            self.cellcolors[idx], self.opacity
        )
        if self.outlinesOn:
            self.layerz[self.outpix[z] == idx] = np.array(self.outcolor).astype(
                np.uint8
            )
            # [0,0,0,self.opacity])
        self.update_layer()

    def remove_cell(self, idx):
        """
        Removes specified cells from the data structure.

            This method updates the internal state of the object by removing one or more cells
            identified by their indices. It ensures that the removal is done in reverse order
            to maintain the integrity of the cell indexing. After the removal operation, it
            updates the state of the graphical interface and the underlying data structures.

            Args:
                idx: A single index or a list of indices representing the cells to be removed.

            Returns:
                None: This method does not return a value.
        """
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        # because the function remove_single_cell updates the state of the cellpix and outpix arrays
        # by reindexing cells to avoid gaps in the indices, we need to remove the cells in reverse order
        # so that the indices are correct
        idx.sort(reverse=True)
        for i in idx:
            self.remove_single_cell(i)
        self.ncells -= len(idx)  # _save_sets uses ncells

        if self.ncells == 0:
            self.ClearButton.setEnabled(False)
        if self.NZ == 1:
            io._save_sets_with_check(self)

        self.update_layer()

    def remove_single_cell(self, idx):
        """
        Remove a single cell identified by its index from the data structures.

            This method updates the internal arrays to remove references to the specified cell
            in both the pixel data and the masking layer. It also adjusts cell indices accordingly
            and logs the removal action.

            Args:
                idx: The index of the cell to be removed.

            Returns:
                None
        """
        # remove from manual array
        self.selected = 0
        if self.NZ > 1:
            zextent = ((self.cellpix == idx).sum(axis=(1, 2)) > 0).nonzero()[0]
        else:
            zextent = [0]
        for z in zextent:
            cp = self.cellpix[z] == idx
            op = self.outpix[z] == idx
            # remove from self.cellpix and self.outpix
            self.cellpix[z, cp] = 0
            self.outpix[z, op] = 0
            if z == self.currentZ:
                # remove from mask layer
                self.layerz[cp] = np.array([0, 0, 0, 0])

        # reduce other pixels by -1
        self.cellpix[self.cellpix > idx] -= 1
        self.outpix[self.outpix > idx] -= 1

        if self.NZ == 1:
            self.removed_cell = [
                self.ismanual[idx - 1],
                self.cellcolors[idx],
                np.nonzero(cp),
                np.nonzero(op),
            ]
            self.redo.setEnabled(True)
            ar, ac = self.removed_cell[2]
            d = datetime.datetime.now()
            self.track_changes.append(
                [d.strftime("%m/%d/%Y, %H:%M:%S"), "removed mask", [ar, ac]]
            )
        # remove cell from lists
        self.ismanual = np.delete(self.ismanual, idx - 1)
        self.cellcolors = np.delete(self.cellcolors, [idx], axis=0)
        del self.zdraw[idx - 1]
        print("GUI_INFO: removed cell %d" % (idx - 1))

    def remove_region_cells(self):
        """
        Removes the selected region of cells and creates a new region of interest (ROI).

            This method unselects any cells currently in the removal list, clears the list, disables relevant
            buttons, and establishes a new ROI centered in the current view, which is half the size of the view.
            It also connects the ROI to signal handlers for further processing.

            Parameters:
                None

            Returns:
                None
        """
        if self.removing_cells_list:
            for idx in self.removing_cells_list:
                self.unselect_cell_multi(idx)
            self.removing_cells_list.clear()
        self.disable_buttons_removeROIs()
        self.removing_region = True

        self.clear_multi_selected_cells()

        # make roi region here in center of view, making ROI half the size of the view
        roi_width = self.p0.viewRect().width() / 2
        x_loc = self.p0.viewRect().x() + (roi_width / 2)
        roi_height = self.p0.viewRect().height() / 2
        y_loc = self.p0.viewRect().y() + (roi_height / 2)

        pos = [x_loc, y_loc]
        roi = pg.RectROI(
            pos, [roi_width, roi_height], pen=pg.mkPen("y", width=2), removable=True
        )
        roi.sigRemoveRequested.connect(self.remove_roi)
        roi.sigRegionChangeFinished.connect(self.roi_changed)
        self.p0.addItem(roi)
        self.remove_roi_obj = roi
        self.roi_changed(roi)

    def delete_multiple_cells(self):
        """
        Delete multiple cells from the current selection.

            This method handles the process of deleting multiple cells from
            the user interface by unselecting the current cell, disabling
            certain buttons related to the removal of regions of interest (ROIs),
            and enabling buttons for confirming or canceling the deletion.

            Parameters:
                None

            Returns:
                None
        """
        self.unselect_cell()
        self.disable_buttons_removeROIs()
        self.DoneDeleteMultipleROIButton.setEnabled(True)
        self.MakeDeletionRegionButton.setEnabled(True)
        self.CancelDeleteMultipleROIButton.setEnabled(True)
        self.deleting_multiple = True

    def done_remove_multiple_cells(self):
        """
        Handles the completion of removing multiple cells in the GUI.

            This method finalizes the cell removal process by disabling relevant buttons,
            clearing the list of cells marked for removal, and updating the GUI accordingly.
            If there are cells to remove, it processes the removal and unselects any selected cells.
            Additionally, it handles the removal of a specified region of interest (ROI) if applicable.

            Args:
                self: The instance of the class that owns this method.

            Returns:
                None: This method does not return a value.
        """
        self.deleting_multiple = False
        self.removing_region = False
        self.DoneDeleteMultipleROIButton.setEnabled(False)
        self.MakeDeletionRegionButton.setEnabled(False)
        self.CancelDeleteMultipleROIButton.setEnabled(False)

        if self.removing_cells_list:
            self.removing_cells_list = list(set(self.removing_cells_list))
            display_remove_list = [i - 1 for i in self.removing_cells_list]
            print(f"GUI_INFO: removing cells: {display_remove_list}")
            self.remove_cell(self.removing_cells_list)
            self.removing_cells_list.clear()
            self.unselect_cell()
        self.enable_buttons()

        if self.remove_roi_obj is not None:
            self.remove_roi(self.remove_roi_obj)

    def merge_cells(self, idx):
        """
        Merge two selected cells in a multi-dimensional array.

            This method updates the selected cell by merging it with another previously selected cell.
            It checks if the cells are touching and, if so, combines their pixel representations and graphical
            contours. It updates the visual representation of the cells and saves the changes.

            Args:
                idx: The index of the cell to be merged with the previously selected cell.

            Returns:
                None: The method primarily modifies the object's state and does not return a value.
        """
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected != self.prev_selected:
            for z in range(self.NZ):
                ar0, ac0 = np.nonzero(self.cellpix[z] == self.prev_selected)
                ar1, ac1 = np.nonzero(self.cellpix[z] == self.selected)
                touching = np.logical_and(
                    (ar0[:, np.newaxis] - ar1) < 3, (ac0[:, np.newaxis] - ac1) < 3
                ).sum()
                ar = np.hstack((ar0, ar1))
                ac = np.hstack((ac0, ac1))
                vr0, vc0 = np.nonzero(self.outpix[z] == self.prev_selected)
                vr1, vc1 = np.nonzero(self.outpix[z] == self.selected)
                self.outpix[z, vr0, vc0] = 0
                self.outpix[z, vr1, vc1] = 0
                if touching > 0:
                    mask = np.zeros((np.ptp(ar) + 4, np.ptp(ac) + 4), np.uint8)
                    mask[ar - ar.min() + 2, ac - ac.min() + 2] = 1
                    contours = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )
                    pvc, pvr = contours[-2][0].squeeze().T
                    vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2

                else:
                    vr = np.hstack((vr0, vr1))
                    vc = np.hstack((vc0, vc1))
                color = self.cellcolors[self.prev_selected]
                self.draw_mask(z, ar, ac, vr, vc, color, idx=self.prev_selected)
            self.remove_cell(self.selected)
            print("GUI_INFO: merged two cells")
            self.update_layer()
            io._save_sets_with_check(self)
            self.undo.setEnabled(False)
            self.redo.setEnabled(False)

    def undo_remove_cell(self):
        """
        Restores the most recently removed cell to the active drawing.

            This method checks if there is a removed cell available for restoration.
            If such a cell exists, it updates the drawing mask, toggles mask operations,
            appends the cell color and manual entry to the relevant arrays, and
            updates the internal state accordingly. It also saves the current state
            and disables the redo operation.

            Returns:
                None: This method does not return a value.
        """
        if len(self.removed_cell) > 0:
            z = 0
            ar, ac = self.removed_cell[2]
            vr, vc = self.removed_cell[3]
            color = self.removed_cell[1]
            self.draw_mask(z, ar, ac, vr, vc, color)
            self.toggle_mask_ops()
            self.cellcolors = np.append(self.cellcolors, color[np.newaxis, :], axis=0)
            self.ncells += 1
            self.ismanual = np.append(self.ismanual, self.removed_cell[0])
            self.zdraw.append([])
            print(">>> added back removed cell")
            self.update_layer()
            io._save_sets_with_check(self)
            self.removed_cell = []
            self.redo.setEnabled(False)

    def remove_stroke(self, delete_points=True, stroke_ind=-1):
        """
        Removes a stroke from the current drawing.

            This method deletes a specified stroke from the strokes list and updates
            the drawing layer accordingly. If the stroke is currently visible, it
            will also update the pixel colors and potentially remove points associated
            with that stroke.

            Args:
                stroke_ind: The index of the stroke to be removed from the strokes list.
                delete_points: A flag indicating whether to delete points associated with
                               the stroke from the current point set.
                e_points: An optional parameter that may control the existence of points
                          in the operation (its specific role must be defined in the class context).

            Returns:
                None: This method does not return any value but modifies the internal state
                      of the object, specifically the strokes and drawing layer.
        """
        stroke = np.array(self.strokes[stroke_ind])
        cZ = self.currentZ
        inZ = stroke[0, 0] == cZ
        if inZ:
            outpix = self.outpix[cZ, stroke[:, 1], stroke[:, 2]] > 0
            self.layerz[stroke[~outpix, 1], stroke[~outpix, 2]] = np.array([0, 0, 0, 0])
            cellpix = self.cellpix[cZ, stroke[:, 1], stroke[:, 2]]
            ccol = self.cellcolors.copy()
            if self.selected > 0:
                ccol[self.selected] = np.array([255, 255, 255])
            col2mask = ccol[cellpix]
            if self.masksOn:
                col2mask = np.concatenate(
                    (col2mask, self.opacity * (cellpix[:, np.newaxis] > 0)), axis=-1
                )
            else:
                col2mask = np.concatenate(
                    (col2mask, 0 * (cellpix[:, np.newaxis] > 0)), axis=-1
                )
            self.layerz[stroke[:, 1], stroke[:, 2], :] = col2mask
            if self.outlinesOn:
                self.layerz[stroke[outpix, 1], stroke[outpix, 2]] = np.array(
                    self.outcolor
                )
            if delete_points:
                # self.current_point_set = self.current_point_set[:-1*(stroke[:,-1]==1).sum()]
                del self.current_point_set[stroke_ind]
            self.update_layer()

        del self.strokes[stroke_ind]

    def plot_clicked(self, event):
        """
        Handles mouse click events to adjust plot ranges.

            This method checks if the left mouse button was clicked without any
            modifier keys (Shift or Alt) and not during a region removal. If
            the event is a double-click, it attempts to set the Y-range of the
            plot to specific limits. If an exception occurs during this operation,
            it defaults to a predefined limit.

            Args:
                event: The mouse event triggered by the user's interaction with
                       the plot. This contains information about the button pressed
                       and other relevant data.

            Returns:
                None: This method does not return a value.
        """
        if (
            event.button() == QtCore.Qt.LeftButton
            and not event.modifiers()
            & (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier)
            and not self.removing_region
        ):
            if event.double():
                try:
                    self.p0.setYRange(0, self.Ly + self.pr)
                except:
                    self.p0.setYRange(0, self.Ly)
                self.p0.setXRange(0, self.Lx)

    def cancel_remove_multiple(self):
        """
        Cancels the removal of multiple selected cells.

            This method resets the state of the application by clearing the multi-selected cells
            and completing the removal process for the selected cells.

            Parameters:
                None

            Returns:
                None
        """
        self.clear_multi_selected_cells()
        self.done_remove_multiple_cells()

    def clear_multi_selected_cells(self):
        """
        Clear all selected cells from the multi-selection list.

            This method unselects all cells that are currently marked for removal
            by iterating through the list of indices stored in `removing_cells_list`
            and calling the unselection method on each of them. It then clears the
            `removing_cells_list`.

            Parameters:
                None

            Returns:
                None
        """
        # unselect all previously selected cells:
        for idx in self.removing_cells_list:
            self.unselect_cell_multi(idx)
        self.removing_cells_list.clear()

    def add_roi(self, roi):
        """
        Adds a Region of Interest (ROI) to the current object.

            This method is responsible for adding a specified ROI to the internal
            structure for display or further processing.

            Args:
                roi: The Region of Interest object to be added.

            Returns:
                None
        """
        self.p0.addItem(roi)
        self.remove_roi_obj = roi

    def remove_roi(self, roi):
        """
        Removes the specified region of interest (ROI) from the graphical interface.

            This method clears any multi-selected cells, verifies that the given ROI matches the
            current object marked for removal, and then removes the ROI from the interface.
            Finally, it resets the removal state.

            Args:
                roi: The region of interest to be removed from the graphical interface.

            Returns:
                None: This method does not return any value.
        """
        self.clear_multi_selected_cells()
        assert roi == self.remove_roi_obj
        self.remove_roi_obj = None
        self.p0.removeItem(roi)
        self.removing_region = False

    def roi_changed(self, roi):
        """
        Update the selected cells based on the region of interest (ROI) changes.

            This method calculates the overlap between the defined region of interest (ROI)
            and the grid of cells, updating the selection of the cells accordingly.
            It ensures that selected cells are within the bounds of the ROI and are within
            the dimensions of the grid.

            Args:
                roi: The region of interest which contains the position and size to be considered.

            Returns:
                None: This method does not return a value, but updates the selected cells
                in the current view based on the ROI.
        """
        # find the overlapping cells and make them selected
        pos = roi.pos()
        size = roi.size()
        x0 = int(pos.x())
        y0 = int(pos.y())
        x1 = int(pos.x() + size.x())
        y1 = int(pos.y() + size.y())
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > self.Lx:
            x1 = self.Lx
        if y1 > self.Ly:
            y1 = self.Ly

        # find cells in that region
        cell_idxs = np.unique(self.cellpix[self.currentZ, y0:y1, x0:x1])
        cell_idxs = np.trim_zeros(cell_idxs)
        # deselect cells not in region by deselecting all and then selecting the ones in the region
        self.clear_multi_selected_cells()

        for idx in cell_idxs:
            self.select_cell_multi(idx)
            self.removing_cells_list.append(idx)

        self.update_layer()

    def mouse_moved(self, pos):
        """
        Handles mouse movement events in the application.

            This method retrieves the items positioned at the provided mouse coordinates
            within the application's scene.

            Args:
                pos: The position of the mouse in the scene.

            Returns:
                A list of items located at the specified position.
        """
        items = self.win.scene().items(pos)

    def color_choose(self):
        """
        Selects a color from the RGB dropdown and updates the plot.

            This method retrieves the currently selected index from the RGB dropdown
            menu and assigns it to the color attribute. It then sets the current index
            of the view dropdown to a default value and calls the update_plot method
            to refresh the plot visualization based on the selected color.

            Parameters:
                None

            Returns:
                None
        """
        self.color = self.RGBDropDown.currentIndex()
        self.view = 0
        self.ViewDropDown.setCurrentIndex(self.view)
        self.update_plot()

    def update_plot(self):
        """
        Update the displayed plot based on the current settings.

            This method updates the visual representation of a data layer or image in a plot.
            It adjusts the plot based on the selected view, handles layer resizing, and sets
            saturation levels according to the current configuration.

            Parameters:
                None

            Returns:
                None
        """
        self.view = self.ViewDropDown.currentIndex()
        self.Ly, self.Lx, _ = self.stack[self.currentZ].shape

        if self.restore and "upsample" in self.restore:
            if self.view != 0:
                if self.view == 3:
                    self.resize = True
                elif len(self.flows[0]) > 0 and self.flows[0].shape[1] == self.Lyr:
                    self.resize = True
                else:
                    self.resize = False
            else:
                self.resize = False
            self.draw_layer()
            self.update_scale()
            self.update_layer()

        if self.view == 0 or self.view == self.ViewDropDown.count() - 1:
            image = (
                self.stack[self.currentZ]
                if self.view == 0
                else self.stack_filtered[self.currentZ]
            )
            if self.nchan == 1:
                # show single channel
                image = image[..., 0]
            if self.color == 0:
                self.img.setImage(image, autoLevels=False, lut=None)
                if self.nchan > 1:
                    levels = np.array(
                        [
                            self.saturation[0][self.currentZ],
                            self.saturation[1][self.currentZ],
                            self.saturation[2][self.currentZ],
                        ]
                    )
                    self.img.setLevels(levels)
                else:
                    self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color > 0 and self.color < 4:
                if self.nchan > 1:
                    image = image[:, :, self.color - 1]
                self.img.setImage(image, autoLevels=False, lut=self.cmap[self.color])
                if self.nchan > 1:
                    self.img.setLevels(self.saturation[self.color - 1][self.currentZ])
                else:
                    self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color == 4:
                if self.nchan > 1:
                    image = image.mean(axis=-1)
                self.img.setImage(image, autoLevels=False, lut=None)
                self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color == 5:
                if self.nchan > 1:
                    image = image.mean(axis=-1)
                self.img.setImage(image, autoLevels=False, lut=self.cmap[0])
                self.img.setLevels(self.saturation[0][self.currentZ])
        else:
            image = np.zeros((self.Ly, self.Lx), np.uint8)
            if len(self.flows) >= self.view - 1 and len(self.flows[self.view - 1]) > 0:
                image = self.flows[self.view - 1][self.currentZ]
            if self.view > 1:
                self.img.setImage(image, autoLevels=False, lut=self.bwr)
            else:
                self.img.setImage(image, autoLevels=False, lut=None)
            self.img.setLevels([0.0, 255.0])

        for r in range(3):
            self.sliders[r].setValue(
                [
                    self.saturation[r][self.currentZ][0],
                    self.saturation[r][self.currentZ][1],
                ]
            )
        self.win.show()
        self.show()

    def update_layer(self):
        """
        Updates the visual representation of the layer and the count of regions of interest.

            This method checks if either masks or outlines are enabled, and if so, it updates the layer's image
            with the current data while also managing the visibility of the layer and its related components.

            It also refreshes the count of regions of interest and updates the display window to reflect changes.

            Parameters:
                None

            Returns:
                None
        """
        if self.masksOn or self.outlinesOn:
            # self.draw_layer()
            self.layer.setImage(self.layerz, autoLevels=False)
        self.update_roi_count()
        self.win.show()
        self.show()

    def update_roi_count(self):
        """
        Update the display of the count of Regions of Interest (ROIs).

            This method updates the user interface by setting the text of the
            ROI count display with the current number of cells (ncells).

            Args:
                None

            Returns:
                None
        """
        self.roi_count.setText(f"{self.ncells} ROIs")

    def add_set(self):
        """
        Add a new cell set based on the current point set and update the visualization.

            This method checks if there are any current points in the point set and
            processes the associated strokes. If the current point set has enough points,
            it adds a mask using these points and updates the cell colors and metadata.
            If the points are insufficient, an error message is displayed. Lastly, it
            clears the current stroke and point set data, and updates the visual layer.

            Parameters:
                None

            Returns:
                None
        """
        if len(self.current_point_set) > 0:
            while len(self.strokes) > 0:
                self.remove_stroke(delete_points=False)
            if len(self.current_point_set[0]) > 8:
                color = self.colormap[self.ncells, :3]
                median = self.add_mask(points=self.current_point_set, color=color)
                if median is not None:
                    self.removed_cell = []
                    self.toggle_mask_ops()
                    self.cellcolors = np.append(
                        self.cellcolors, color[np.newaxis, :], axis=0
                    )
                    self.ncells += 1
                    self.ismanual = np.append(self.ismanual, True)
                    if self.NZ == 1:
                        # only save after each cell if single image
                        io._save_sets_with_check(self)
            else:
                print("GUI_ERROR: cell too small, not drawn")
            self.current_stroke = []
            self.strokes = []
            self.current_point_set = []
            self.update_layer()

    def add_mask(self, points=None, color=(100, 200, 50), dense=True):
        """
        Add a mask based on provided stroke points.

            This method processes a list of stroke points to create and add a mask
            to the current state. It checks for overlaps with existing cells,
            ensuring the newly drawn mask has a sufficient number of non-overlapping pixels.

            Args:
                points: A list of strokes, where each stroke is a collection of points.
                color: The RGB color value used for the mask (default is (100, 200, 50)).
                dense: A boolean indicating whether to create a dense outline (default is True).

            Returns:
                A list containing the median x and y coordinates of the added mask,
                or None if the mask contains insufficient non-overlapping pixels.
        """
        # points is list of strokes
        points_all = np.concatenate(points, axis=0)

        # loop over z values
        median = []
        zdraw = np.unique(points_all[:, 0])
        z = 0
        ars, acs, vrs, vcs = (
            np.zeros(0, "int"),
            np.zeros(0, "int"),
            np.zeros(0, "int"),
            np.zeros(0, "int"),
        )
        for stroke in points:
            stroke = np.concatenate(stroke, axis=0).reshape(-1, 4)
            vr = stroke[:, 1]
            vc = stroke[:, 2]
            # get points inside drawn points
            mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
            pts = np.stack((vc - vc.min() + 2, vr - vr.min() + 2), axis=-1)[
                :, np.newaxis, :
            ]
            mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
            ar, ac = np.nonzero(mask)
            ar, ac = ar + vr.min() - 2, ac + vc.min() - 2
            # get dense outline
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0][:, 0].T
            vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            # concatenate all points
            ar, ac = np.hstack((np.vstack((vr, vc)), np.vstack((ar, ac))))
            # if these pixels are overlapping with another cell, reassign them
            ioverlap = self.cellpix[z][ar, ac] > 0
            if (~ioverlap).sum() < 10:
                print("GUI_ERROR: cell < 10 pixels without overlaps, not drawn")
                return None
            elif ioverlap.sum() > 0:
                ar, ac = ar[~ioverlap], ac[~ioverlap]
                # compute outline of new mask
                mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
                mask[ar - vr.min() + 2, ac - vc.min() + 2] = 1
                contours = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                pvc, pvr = contours[-2][0][:, 0].T
                vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            ars = np.concatenate((ars, ar), axis=0)
            acs = np.concatenate((acs, ac), axis=0)
            vrs = np.concatenate((vrs, vr), axis=0)
            vcs = np.concatenate((vcs, vc), axis=0)

        self.draw_mask(z, ars, acs, vrs, vcs, color)
        median.append(np.array([np.median(ars), np.median(acs)]))

        self.zdraw.append(zdraw)
        d = datetime.datetime.now()
        self.track_changes.append(
            [d.strftime("%m/%d/%Y, %H:%M:%S"), "added mask", [ar, ac]]
        )
        return median

    def draw_mask(self, z, ar, ac, vr, vc, color, idx=None):
        """draw single mask using outlines and area"""
        if idx is None:
            idx = self.ncells + 1
        self.cellpix[z, vr, vc] = idx
        self.cellpix[z, ar, ac] = idx
        self.outpix[z, vr, vc] = idx
        if self.restore and "upsample" in self.restore:
            if self.resize:
                self.cellpix_resize[z, vr, vc] = idx
                self.cellpix_resize[z, ar, ac] = idx
                self.outpix_resize[z, vr, vc] = idx
                self.cellpix_orig[
                    z, (vr / self.ratio).astype(int), (vc / self.ratio).astype(int)
                ] = idx
                self.cellpix_orig[
                    z, (ar / self.ratio).astype(int), (ac / self.ratio).astype(int)
                ] = idx
                self.outpix_orig[
                    z, (vr / self.ratio).astype(int), (vc / self.ratio).astype(int)
                ] = idx
            else:
                self.cellpix_orig[z, vr, vc] = idx
                self.cellpix_orig[z, ar, ac] = idx
                self.outpix_orig[z, vr, vc] = idx

                # get upsampled mask
                vrr = (vr.copy() * self.ratio).astype(int)
                vcr = (vc.copy() * self.ratio).astype(int)
                mask = np.zeros((np.ptp(vrr) + 4, np.ptp(vcr) + 4), np.uint8)
                pts = np.stack((vcr - vcr.min() + 2, vrr - vrr.min() + 2), axis=-1)[
                    :, np.newaxis, :
                ]
                mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
                arr, acr = np.nonzero(mask)
                arr, acr = arr + vrr.min() - 2, acr + vcr.min() - 2
                # get dense outline
                contours = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                pvc, pvr = contours[-2][0].squeeze().T
                vrr, vcr = pvr + vrr.min() - 2, pvc + vcr.min() - 2
                # concatenate all points
                arr, acr = np.hstack((np.vstack((vrr, vcr)), np.vstack((arr, acr))))
                self.cellpix_resize[z, vrr, vcr] = idx
                self.cellpix_resize[z, arr, acr] = idx
                self.outpix_resize[z, vrr, vcr] = idx

        if z == self.currentZ:
            self.layerz[ar, ac, :3] = color
            if self.masksOn:
                self.layerz[ar, ac, -1] = self.opacity
            if self.outlinesOn:
                self.layerz[vr, vc] = np.array(self.outcolor)

    def compute_scale(self):
        """
        Computes the scale based on the diameter and updates the radii array.

        This method retrieves the diameter from a text input, calculates the padded radii
        based on this value, initializes a radii array, and fills it with a specific RGB color.
        It also updates the display ranges for visualization.

        Parameters:
            None

        Returns:
            None
        """
        self.diameter = float(self.Diameter.text())
        self.pr = int(float(self.Diameter.text()))
        self.radii_padding = int(self.pr * 1.25)
        self.radii = np.zeros((self.Ly + self.radii_padding, self.Lx, 4), np.uint8)
        yy, xx = disk(
            [self.Ly + self.radii_padding / 2 - 1, self.pr / 2 + 1],
            self.pr / 2,
            self.Ly + self.radii_padding,
            self.Lx,
        )
        # rgb(150,50,150)
        self.radii[yy, xx, 0] = 150
        self.radii[yy, xx, 1] = 50
        self.radii[yy, xx, 2] = 150
        self.radii[yy, xx, 3] = 255
        self.p0.setYRange(0, self.Ly + self.radii_padding)
        self.p0.setXRange(0, self.Lx)

    def update_scale(self):
        """
        Updates the scale of the image display with computed values.

            This method recalculates the scale based on the current radii and
            sets the image scale levels to a predefined range. It ensures that
            the updated scale is displayed in the application window.

            Parameters:
                None

            Returns:
                None
        """
        self.compute_scale()
        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0, 255.0])
        self.win.show()
        self.show()

    def redraw_masks(self, masks=True, outlines=True, draw=True):
        """
        Redraws the masks in the current layer.

            This method is responsible for updating the visual representation
            of the masks in the layer. It can control whether outlines are
            drawn and whether the drawing should occur.

            Args:
                outlines: Indicates if outlines should be drawn around the masks.
                draw: Indicates if the actual drawing action should be performed.

            Returns:
                None: This method does not return any value.
        """
        self.draw_layer()

    def draw_masks(self):
        """
        Draws masks on the current layer.

            This method invokes the draw_layer method to render masks on the existing layer.

            Returns:
                None: This method does not return any value.
        """
        self.draw_layer()

    def draw_layer(self):
        """
        Draws the current layer based on various conditions and parameters.

            This method updates the layer image by applying cell colors, opacity,
            and outlines if specified. It handles resizing based on the state
            of the resize attribute and modifies the layerz array to reflect
            the current drawing state.

            Attributes:
                - self.resize: Indicates whether to resize the layer.
                - self.Ly, self.Lx: Dimensions of the layer.
                - self.masksOn: Flag to determine if masks should be applied.
                - self.outlinesOn: Flag to determine if outlines should be drawn.
                - self.restore: Optional attribute for restoring previous states.
                - self.opacity: Opacity value for the current layer.
                - self.selected: Currently selected item for highlighting.
                - self.cellpix: Array representing cell pixel values.
                - self.outpix: Array representing outline pixel values.
                - self.strokes: List of strokes to apply to the layer.
                - self.cellcolors: Array of colors corresponding to cell pixel values.
                - self.outcolor: Color to use for outlines.

            Returns:
                None
        """
        if self.resize:
            self.Ly, self.Lx = self.Lyr, self.Lxr
        else:
            self.Ly, self.Lx = self.Ly0, self.Lx0

        if self.masksOn or self.outlinesOn:
            if self.restore and "upsample" in self.restore:
                if self.resize:
                    self.cellpix = self.cellpix_resize.copy()
                    self.outpix = self.outpix_resize.copy()
                else:
                    self.cellpix = self.cellpix_orig.copy()
                    self.outpix = self.outpix_orig.copy()

        # print(self.cellpix.shape, self.outpix.shape, self.cellpix.max(), self.outpix.max())
        self.layerz = np.zeros((self.Ly, self.Lx, 4), np.uint8)
        if self.masksOn:
            self.layerz[..., :3] = self.cellcolors[self.cellpix[self.currentZ], :]
            self.layerz[..., 3] = self.opacity * (
                self.cellpix[self.currentZ] > 0
            ).astype(np.uint8)
            if self.selected > 0:
                self.layerz[self.cellpix[self.currentZ] == self.selected] = np.array(
                    [255, 255, 255, self.opacity]
                )
            cZ = self.currentZ
            stroke_z = np.array([s[0][0] for s in self.strokes])
            inZ = np.nonzero(stroke_z == cZ)[0]
            if len(inZ) > 0:
                for i in inZ:
                    stroke = np.array(self.strokes[i])
                    self.layerz[stroke[:, 1], stroke[:, 2]] = np.array(
                        [255, 0, 255, 100]
                    )
        else:
            self.layerz[..., 3] = 0

        if self.outlinesOn:
            self.layerz[self.outpix[self.currentZ] > 0] = np.array(
                self.outcolor
            ).astype(np.uint8)

    def set_restore_button(self):
        """
        Sets the style of restore buttons based on the current restore state.

            This method iterates through the denoise text keys and updates the
            appearance of the corresponding buttons. If a key is not "none" and
            is part of the current restore state, the button's style is set
            to indicate that it is pressed. If the key is "none" and there is
            no current restore state, the style is also set to pressed. Otherwise,
            the button is reset to its unpressed style if it is enabled.

            Parameters:
                None

            Returns:
                None
        """
        keys = self.denoise_text
        for i, key in enumerate(keys):
            if key != "none" and (self.restore and key in self.restore):
                self.DenoiseButtons[i].setStyleSheet(self.stylePressed)
            elif key == "none" and self.restore is None:
                self.DenoiseButtons[i].setStyleSheet(self.stylePressed)
            else:
                if self.DenoiseButtons[i].isEnabled():
                    self.DenoiseButtons[i].setStyleSheet(self.styleUnpressed)

    def set_normalize_params(self, normalize_params):
        """
        Set and update normalization parameters.

            This method updates the normalization parameters based on default values
            unless a specific condition is met. It uses predefined parameters and ensures
            that certain values are set appropriately.

            Args:
                normalize_params: A dictionary containing normalization parameters.
                    Keys may include 'percentile', 'sharpen_radius', 'smooth_radius',
                    'tile_norm_blocksize', 'tile_norm_smooth3D', 'norm3D', and 'invert'.

            Returns:
                None: This method modifies the normalize_params dictionary in place
                and does not return any value.
        """
        from cellpose.models import normalize_default

        if self.restore != "filter":
            keys = list(normalize_params.keys()).copy()
            for key in keys:
                if key != "percentile":
                    normalize_params[key] = normalize_default[key]
        normalize_params = {**normalize_default, **normalize_params}
        percentile = self.check_percentile_params(normalize_params["percentile"])
        out = self.check_filter_params(
            normalize_params["sharpen_radius"],
            normalize_params["smooth_radius"],
            normalize_params["tile_norm_blocksize"],
            normalize_params["tile_norm_smooth3D"],
            normalize_params["norm3D"],
            normalize_params["invert"],
        )

    def check_percentile_params(self, percentile):
        """
        Check and normalize percentile parameters.

            This method validates the provided percentile values to ensure they
            are within the acceptable range of 0 to 100, with the lower percentile
            being less than the upper percentile. If the provided percentiles are
            invalid or None, it defaults to a range of [1, 99].

            Args:
                percentile: A list containing two values that represent the lower
                            and upper percentiles.

            Returns:
                A list containing the normalized lower and upper percentiles.
        """
        # check normalization params
        if percentile is not None and not (
            percentile[0] >= 0
            and percentile[1] > 0
            and percentile[0] < 100
            and percentile[1] <= 100
            and percentile[1] > percentile[0]
        ):
            print(
                "GUI_ERROR: percentiles need be between 0 and 100, and upper > lower, using defaults"
            )
            self.norm_edits[0].setText("1.")
            self.norm_edits[1].setText("99.")
            percentile = [1.0, 99.0]
        elif percentile is None:
            percentile = [1.0, 99.0]
        self.norm_edits[0].setText(str(percentile[0]))
        self.norm_edits[1].setText(str(percentile[1]))
        return percentile

    def check_filter_params(self, sharpen, smooth, tile_norm, smooth3D, norm3D, invert):
        """
        Checks and updates filter parameters for image processing.

            This method ensures that the filter parameters are non-negative where appropriate
            and updates the corresponding UI elements with these values. It also checks that
            the tile size does not exceed the dimensions of the image and sets appropriate
            defaults if necessary.

            Args:
                sharpen: The sharpening filter strength.
                smooth: The smoothing filter strength.
                tile_norm: The tile size for normalization.
                smooth3D: The 3D smoothing filter strength.
                norm3D: A flag to indicate whether to use 3D normalization.
                invert: A flag to indicate whether the image should be inverted.

            Returns:
                A tuple containing the updated filter parameters:
                sharpen, smooth, tile_norm, smooth3D, norm3D, invert.
        """
        tile_norm = 0 if tile_norm < 0 else tile_norm
        sharpen = 0 if sharpen < 0 else sharpen
        smooth = 0 if smooth < 0 else smooth
        smooth3D = 0 if smooth3D < 0 else smooth3D
        norm3D = bool(norm3D)
        invert = bool(invert)
        if tile_norm > self.Ly and tile_norm > self.Lx:
            print(
                "GUI_ERROR: tile size (tile_norm) bigger than both image dimensions, disabling"
            )
            tile_norm = 0
        self.filt_edits[0].setText(str(sharpen))
        self.filt_edits[1].setText(str(smooth))
        self.filt_edits[2].setText(str(tile_norm))
        self.filt_edits[3].setText(str(smooth3D))
        self.norm3D_cb.setChecked(norm3D)
        self.invert_cb.setChecked(invert)
        return sharpen, smooth, tile_norm, smooth3D, norm3D, invert

    def get_normalize_params(self):
        """
        Retrieve normalization parameters for image processing.

            This method collects and validates normalization parameters from user inputs,
            constructing a dictionary with settings for normalization based on current state
            and options selected in the user interface. It considers options for 3D normalization
            and filter parameters when applicable.

            Returns:
                A dictionary containing the normalization parameters, including percentile values,
                evaluation of whether 3D normalization is enabled, and optional filter settings
                such as sharpen radius and smooth radius.

            Raises:
                ValueError: If the percentile or filter parameters are not valid.
        """
        percentile = [
            float(self.norm_edits[0].text()),
            float(self.norm_edits[1].text()),
        ]
        self.check_percentile_params(percentile)
        normalize_params = {"percentile": percentile}
        norm3D = self.norm3D_cb.isChecked()
        normalize_params["norm3D"] = norm3D
        if self.restore == "filter":
            sharpen = float(self.filt_edits[0].text())
            smooth = float(self.filt_edits[1].text())
            tile_norm = float(self.filt_edits[2].text())
            smooth3D = float(self.filt_edits[3].text())
            invert = self.invert_cb.isChecked()
            out = self.check_filter_params(
                sharpen, smooth, tile_norm, smooth3D, norm3D, invert
            )
            sharpen, smooth, tile_norm, smooth3D, norm3D, invert = out
            normalize_params["sharpen_radius"] = sharpen
            normalize_params["smooth_radius"] = smooth
            normalize_params["tile_norm_blocksize"] = tile_norm
            normalize_params["tile_norm_smooth3D"] = smooth3D
            normalize_params["invert"] = invert

        from cellpose.models import normalize_default

        normalize_params = {**normalize_default, **normalize_params}

        return normalize_params

    def compute_saturation(self, return_img=False):
        """
        Compute the saturation levels for each channel of the image stack.

            This method normalizes the image stack based on specified parameters, applies filtering and normalization if necessary,
            and calculates saturation values based on the computed image for each channel. The results are stored in the instance
            variable `saturation`.

            Args:
                return_img: Boolean flag indicating whether to return the normalized image.
                             If True, the normalized image will be returned, otherwise the method will just compute saturation.

            Returns:
                The normalized image if `return_img` is True, otherwise None.
        """
        norm = self.get_normalize_params()
        print(norm)
        sharpen, smooth = norm["sharpen_radius"], norm["smooth_radius"]
        percentile = norm["percentile"]
        tile_norm = norm["tile_norm_blocksize"]
        invert = norm["invert"]
        norm3D = norm["norm3D"]
        smooth3D = norm["tile_norm_smooth3D"]
        tile_norm = norm["tile_norm_blocksize"]

        # if grayscale, use gray img
        channels = self.get_channels()
        if channels[0] == 0:
            img_norm = self.stack.mean(axis=-1, keepdims=True)
        elif sharpen > 0 or smooth > 0 or tile_norm > 0:
            img_norm = self.stack.copy()
        else:
            img_norm = self.stack

        if sharpen > 0 or smooth > 0 or tile_norm > 0:
            self.clear_restore()
            self.restore = "filter"
            print(
                "GUI_INFO: computing filtered image because sharpen > 0 or tile_norm > 0"
            )
            print(
                "GUI_WARNING: will use memory to create filtered image -- make sure to have RAM for this"
            )
            img_norm = self.stack.copy()
            if sharpen > 0 or smooth > 0:
                img_norm = smooth_sharpen_img(
                    self.stack, sharpen_radius=sharpen, smooth_radius=smooth
                )

            if tile_norm > 0:
                img_norm = normalize99_tile(
                    img_norm,
                    blocksize=tile_norm,
                    lower=percentile[0],
                    upper=percentile[1],
                    smooth3D=smooth3D,
                    norm3D=norm3D,
                )
            # convert to 0->255
            img_norm_min = img_norm.min()
            img_norm_max = img_norm.max()
            for c in range(img_norm.shape[-1]):
                if np.ptp(img_norm[..., c]) > 1e-3:
                    img_norm[..., c] -= img_norm_min
                    img_norm[..., c] /= img_norm_max - img_norm_min
            img_norm *= 255
            self.stack_filtered = img_norm
            self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(
                True
            )
            self.ViewDropDown.setCurrentIndex(self.ViewDropDown.count() - 1)
        elif invert:
            img_norm = self.stack.copy()
        else:
            img_norm = (
                self.stack
                if self.restore is None or self.restore == "filter"
                else self.stack_filtered
            )

        self.saturation = []
        for c in range(img_norm.shape[-1]):
            self.saturation.append([])
            if np.ptp(img_norm[..., c]) > 1e-3:
                if norm3D:
                    x01 = np.percentile(img_norm[..., c], percentile[0])
                    x99 = np.percentile(img_norm[..., c], percentile[1])
                    if invert:
                        x01i = 255.0 - x99
                        x99i = 255.0 - x01
                        x01, x99 = x01i, x99i
                    for n in range(self.NZ):
                        self.saturation[-1].append([x01, x99])
                else:
                    for z in range(self.NZ):
                        if self.NZ > 1:
                            x01 = np.percentile(img_norm[z, :, :, c], percentile[0])
                            x99 = np.percentile(img_norm[z, :, :, c], percentile[1])
                        else:
                            x01 = np.percentile(img_norm[..., c], percentile[0])
                            x99 = np.percentile(img_norm[..., c], percentile[1])
                        if invert:
                            x01i = 255.0 - x99
                            x99i = 255.0 - x01
                            x01, x99 = x01i, x99i
                        self.saturation[-1].append([x01, x99])
            else:
                for n in range(self.NZ):
                    self.saturation[-1].append([0, 255.0])
        # if only 2 restore channels, add blue
        if len(self.saturation) < 3:
            for i in range(3 - len(self.saturation)):
                self.saturation.append([])
                for n in range(self.NZ):
                    self.saturation[-1].append([0, 255.0])
        print(self.saturation[2][self.currentZ])

        if invert:
            img_norm = 255.0 - img_norm
            self.stack_filtered = img_norm
            self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(
                True
            )
            self.ViewDropDown.setCurrentIndex(self.ViewDropDown.count() - 1)

        if img_norm.shape[-1] == 1:
            self.saturation.append(self.saturation[0])
            self.saturation.append(self.saturation[0])

        self.autobtn.setChecked(True)
        self.update_plot()

    def chanchoose(self, image):
        """
        Selects channels from a multi-channel image based on user input.

            This method processes an image and selects specific channels based on the
            current index values from the ChannelChoose UI elements. If the image
            has more than two dimensions and multiple channels are available, the
            method will return either the mean of the channels or the specified
            channels according to the user's choice. If the image does not meet
            the criteria, it returns the original image.

            Returns:
                The processed image, which could be either the mean of the channels
                if the first channel index is 0, the selected channels as per the
                specified indices, or the original image if conditions are not met.
        """
        if image.ndim > 2 and self.nchan > 1:
            if self.ChannelChoose[0].currentIndex() == 0:
                return image.mean(axis=-1, keepdims=True)
            else:
                chanid = [self.ChannelChoose[0].currentIndex() - 1]
                if self.ChannelChoose[1].currentIndex() > 0:
                    chanid.append(self.ChannelChoose[1].currentIndex() - 1)
                return image[:, :, chanid]
        else:
            return image

    def get_model_path(self, custom=False):
        """
        Retrieve the file path of the selected model.

            This method determines the path of a model based on the current selection
            in the model chooser. It updates the current model attribute and its path
            accordingly, either using a custom selection or defaulting to a predefined
            list of network names.

            Args:
                custom: A boolean indicating whether to use a custom model selection.
                        If true, the method uses the model selected in the
                        ModelChooseC dropdown. If false, it will use a model
                        based on the selection in the ModelChooseB dropdown.

            Returns:
                str: The file path of the currently selected model.
        """
        if custom:
            self.current_model = self.ModelChooseC.currentText()
            self.current_model_path = os.fspath(
                models.MODEL_DIR.joinpath(self.current_model)
            )
        else:
            self.current_model = self.net_names[
                max(0, self.ModelChooseB.currentIndex() - 1)
            ]
            self.current_model_path = models.model_path(self.current_model)

    def initialize_model(self, model_name=None, custom=False):
        """
        Initializes the model based on the specified parameters.

            This method sets up the model for use in the application. It checks for
            the validity of the model name and retrieves the appropriate model path.
            If necessary, it creates an instance of the CellposeModel or Cellpose
            class based on the selected model type.

            Args:
                model_name: The name of the model to initialize. If set to None
                            or if it's "dataset-specific models", a custom model must be specified.
                del_name: An optional name for deletion, not used in the current context.
                custom: A boolean indicating whether a custom model path should be used.

            Returns:
                None
        """
        if model_name == "dataset-specific models":
            raise ValueError("need to specify model (use dropdown)")
        elif model_name is None or custom:
            self.get_model_path(custom=custom)
            if not os.path.exists(self.current_model_path):
                raise ValueError("need to specify model (use dropdown)")

        if model_name is None or not isinstance(model_name, str):
            self.model = models.CellposeModel(
                gpu=self.useGPU.isChecked(), pretrained_model=self.current_model_path
            )
        else:
            self.current_model = model_name
            if self.current_model == "cyto" or self.current_model == "nuclei":
                self.current_model_path = models.model_path(self.current_model, 0)
            else:
                self.current_model_path = os.fspath(
                    models.MODEL_DIR.joinpath(self.current_model)
                )

            if self.current_model != "cyto3":
                diam_mean = 17.0 if self.current_model == "nuclei" else 30.0
                self.model = models.CellposeModel(
                    gpu=self.useGPU.isChecked(),
                    diam_mean=diam_mean,
                    model_type=self.current_model,
                )
            else:
                self.model = models.Cellpose(
                    gpu=self.useGPU.isChecked(), model_type=self.current_model
                )

    def add_model(self):
        """
        Add a model to the system.

            This method is responsible for adding a new model to the underlying
            data structure managed by the system. It delegates the operation to
            an internal I/O handler.

            Parameters:
                None

            Returns:
                None
        """
        io._add_model(self)
        return

    def remove_model(self):
        """
        Remove the current model.

            This method is responsible for removing the currently active model
            from the system. It performs necessary cleanup and deallocates
            resources associated with the model.

            Parameters:
              None

            Returns:
              None
        """
        io._remove_model(self)
        return

    def new_model(self):
        """
        Trains a new model based on the provided training dataset.

            This method first checks if the data is two-dimensional. If not, it prints an error message and exits.
            If the data is suitable, it retrieves the training dataset and opens a training window. The user can
            decide whether to proceed with the training. If training is initiated, the model is trained using the
            specified parameters.

            Parameters:
                None

            Returns:
                None
        """
        if self.NZ != 1:
            print("ERROR: cannot train model on 3D data")
            return

        # train model
        image_names = self.get_files()[0]
        (
            self.train_data,
            self.train_labels,
            self.train_files,
            restore,
            normalize_params,
        ) = io._get_train_set(image_names)
        TW = guiparts.TrainWindow(self, models.MODEL_NAMES)
        train = TW.exec_()
        if train:
            self.logger.info(
                f"training with {[os.path.split(f)[1] for f in self.train_files]}"
            )
            self.train_model(restore=restore, normalize_params=normalize_params)
        else:
            print("GUI_INFO: training cancelled")

    def train_model(self, restore=None, normalize_params=None):
        """
        Trains a new Cellpose model or continues training an existing model.

            This method initializes the model training, either by training a new model or continuing from a specified checkpoint based on the provided training parameters. It configures the training environment and logs the progress while saving the model's training losses.

            Args:
                normalize_params: Parameters used to normalize the training data. If not provided, default normalization parameters are used.

            Returns:
                None: The method does not return any value but saves the trained model and its associated training losses to the specified path.
        """
        from cellpose.models import normalize_default

        if normalize_params is None:
            normalize_params = copy.deepcopy(normalize_default)
        if self.training_params["model_index"] < len(models.MODEL_NAMES):
            model_type = models.MODEL_NAMES[self.training_params["model_index"]]
            self.logger.info(f"training new model starting at model {model_type}")
        else:
            model_type = None
            self.logger.info(f"training new model starting from scratch")
        self.current_model = model_type
        self.channels = self.training_params["channels"]

        self.logger.info(
            f"training with chan = {self.ChannelChoose[0].currentText()}, chan2 = {self.ChannelChoose[1].currentText()}"
        )

        self.model = models.CellposeModel(
            gpu=self.useGPU.isChecked(), model_type=model_type
        )
        self.SizeButton.setEnabled(False)
        save_path = os.path.dirname(self.filename)

        print("GUI_INFO: name of new model: " + self.training_params["model_name"])
        print(f"GUI_INFO: SGD activated: {self.training_params['SGD']}")
        self.new_model_path, train_losses = train.train_seg(
            self.model.net,
            train_data=self.train_data,
            train_labels=self.train_labels,
            channels=self.channels,
            normalize=normalize_params,
            min_train_masks=0,
            save_path=save_path,
            nimg_per_epoch=max(8, len(self.train_data)),
            learning_rate=self.training_params["learning_rate"],
            weight_decay=self.training_params["weight_decay"],
            n_epochs=self.training_params["n_epochs"],
            SGD=self.training_params["SGD"],
            model_name=self.training_params["model_name"],
        )[:2]
        # save train losses
        np.save(str(self.new_model_path) + "_train_losses.npy", train_losses)
        # run model on next image
        io._add_model(self, self.new_model_path)
        diam_labels = self.model.net.diam_labels.item()  # .copy()
        self.new_model_ind = len(self.model_strings)
        self.autorun = True
        channels = self.channels.copy()
        self.clear_all()
        # keep same channels
        self.ChannelChoose[0].setCurrentIndex(channels[0])
        self.ChannelChoose[1].setCurrentIndex(channels[1])
        self.diameter = diam_labels
        self.Diameter.setText("%0.2f" % self.diameter)
        self.logger.info(f">>>> diameter set to diam_labels ( = {diam_labels: 0.3f} )")
        self.restore = restore
        self.set_normalize_params(normalize_params)
        self.get_next_image(load_seg=False)

        self.compute_segmentation(custom=True)
        self.logger.info(
            f"!!! computed masks for {os.path.split(self.filename)[1]} from new model !!!"
        )

    def compute_restore(self):
        """
        Executes the image restoration process based on the specified settings.

            This method checks if a restoration operation is required, and if so, it logs the action,
            processes the restoration type, and configures the appropriate parameters for image denoising
            or saturation computation.

            Parameters:
                None

            Returns:
                None
        """
        if self.restore:
            self.logger.info(f"running image restoration {self.restore}")
            if self.restore != "filter":
                rstr = self.restore.split("_")
                model_type = rstr[0]
                if len(rstr) > 1:
                    dset = rstr[1]
                    if dset == "cyto3":
                        self.DenoiseChoose.setCurrentIndex(0)
                    else:
                        self.DenoiseChoose.setCurrentIndex(1)
                if "upsample" in self.restore:
                    i = self.DenoiseChoose.currentIndex()
                    diam_up = 30.0 if i == 0 or i == 1 else 17.0
                    print(diam_up, self.ratio)
                    self.Diameter.setText(str(diam_up / self.ratio))
                self.compute_denoise_model(model_type=model_type)
            else:
                self.compute_saturation()

    def get_thresholds(self):
        """
        Retrieve flow and cell probability thresholds.

            This method attempts to parse the flow and cell probability thresholds
            from their respective text fields. If successful, it returns these thresholds
            as floats. In the case of a parsing failure, it sets default values and
            returns them instead.

            Returns:
                A tuple containing:
                    flow_threshold: The flow threshold as a float or None if the threshold is zero or NZ is greater than 1.
                    cellprob_threshold: The cell probability threshold as a float.
        """
        try:
            flow_threshold = float(self.flow_threshold.text())
            cellprob_threshold = float(self.cellprob_threshold.text())
            if flow_threshold == 0.0 or self.NZ > 1:
                flow_threshold = None
            return flow_threshold, cellprob_threshold
        except Exception as e:
            print(
                "flow threshold or cellprob threshold not a valid number, setting to defaults"
            )
            self.flow_threshold.setText("0.4")
            self.cellprob_threshold.setText("0.0")
            return 0.4, 0.0

    def compute_cprob(self):
        """
        Compute the probability masks based on flow thresholds.

            This method computes masks using provided flow data and cell probability thresholds.
            It logs the thresholds used for the computation and updates the GUI with the result.

            If the recompute_masks flag is set, it retrieves thresholds to determine how masks are computed.
            The computed masks are then displayed in the GUI and the number of detected cells is logged.

            Args:
                None

            Returns:
                None: This method does not return any value but updates internal state and GUI components.
        """
        if self.recompute_masks:
            flow_threshold, cellprob_threshold = self.get_thresholds()
            if flow_threshold is None:
                self.logger.info(
                    "computing masks with cell prob=%0.3f, no flow error threshold"
                    % (cellprob_threshold)
                )
            else:
                self.logger.info(
                    "computing masks with cell prob=%0.3f, flow error threshold=%0.3f"
                    % (cellprob_threshold, flow_threshold)
                )
            maski = dynamics.resize_and_compute_masks(
                self.flows[4][:-1],
                self.flows[4][-1],
                p=self.flows[3].copy(),
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                resize=self.cellpix.shape[-2:],
            )[0]

            self.masksOn = True
            if not self.OCheckBox.isChecked():
                self.MCheckBox.setChecked(True)
            if maski.ndim < 3:
                maski = maski[np.newaxis, ...]
            self.logger.info("%d cells found" % (len(np.unique(maski)[1:])))
            io._masks_to_gui(self, maski, outlines=None)
            self.show()

    def compute_denoise_model(self, model_type=None):
        """
        Computes the denoising model for an image stack.

            This method initializes and evaluates a denoising model based on the selected parameters
            and updates the progress indicator. It handles both upsampling and non-upsampling of the
            images while normalizing the data and computing saturation values for the image channels.

            Args:
                model_type: A string that specifies the type of the denoising model to use.
                            If not provided, defaults to None.

            Returns:
                None: This method does not return a value but updates the internal state of the instance,
                including the filtered image stack and saturation values for the channels.
        """
        self.progress.setValue(0)
        try:
            tic = time.time()
            nstr = self.DenoiseChoose.currentText()
            nstr.replace("-", "")
            self.clear_restore()
            model_name = model_type + "_" + nstr
            print(model_name)
            # denoising model
            self.denoise_model = denoise.DenoiseModel(
                gpu=self.useGPU.isChecked(), model_type=model_name
            )
            self.progress.setValue(10)
            diam_up = 30.0 if "cyto" in model_name else 17.0

            # params
            channels = self.get_channels()
            self.diameter = float(self.Diameter.text())
            normalize_params = self.get_normalize_params()
            print("GUI_INFO: channels: ", channels)
            print("GUI_INFO: normalize_params: ", normalize_params)
            print("GUI_INFO: diameter (before upsampling): ", self.diameter)

            data = self.stack.copy()
            print(data.shape)
            self.Ly, self.Lx = data.shape[-3:-1]
            if "upsample" in model_name:
                # get upsampling factor
                if self.diameter >= diam_up:
                    print(
                        f"GUI_ERROR: cannot upsample, already set to pixel diameter >= {diam_up}"
                    )
                    self.progress.setValue(0)
                    return
                self.ratio = diam_up / self.diameter
                print(
                    "GUI_WARNING: upsampling image, this will also duplicate mask layer and resize it, will use more RAM"
                )
                print(
                    f"GUI_INFO: upsampling image to {diam_up} pixel diameter ({self.ratio:0.2f} times)"
                )
                self.Lyr, self.Lxr = int(self.Ly * self.ratio), int(
                    self.Lx * self.ratio
                )
                self.Ly0, self.Lx0 = self.Ly, self.Lx
                # moved resize into eval
                # data = resize_image(data, Ly=self.Lyr, Lx=self.Lxr)
                # self.diameter = diam_up
                # self.Diameter.setText(str(diam_up))
            else:
                self.Lyr, self.Lxr = self.Ly, self.Lx
                self.Ly0, self.Lx0 = self.Ly, self.Lx
                diam_up = self.diameter

            img_norm = self.denoise_model.eval(
                data,
                channels=channels,
                z_axis=0,
                channel_axis=3,
                diameter=self.diameter,
                normalize=normalize_params,
            )
            print(img_norm.shape)
            self.diameter = diam_up
            self.Diameter.setText(str(diam_up))

            if img_norm.ndim == 2:
                img_norm = img_norm[:, :, np.newaxis]
            if img_norm.ndim == 3:
                img_norm = img_norm[np.newaxis, ...]

            self.progress.setValue(100)
            self.logger.info(
                f"{model_name} finished in %0.3f sec" % (time.time() - tic)
            )

            # compute saturation
            percentile = normalize_params["percentile"]
            img_norm_min = img_norm.min()
            img_norm_max = img_norm.max()
            chan = [0] if channels[0] == 0 else [channels[0] - 1, channels[1] - 1]
            self.saturation = [[], [], []]
            for c in range(img_norm.shape[-1]):
                if np.ptp(img_norm[..., c]) > 1e-3:
                    img_norm[..., c] -= img_norm_min
                    img_norm[..., c] /= img_norm_max - img_norm_min
                for z in range(self.NZ):
                    x01 = np.percentile(img_norm[z, :, :, c], percentile[0]) * 255.0
                    x99 = np.percentile(img_norm[z, :, :, c], percentile[1]) * 255.0
                    self.saturation[chan[c]].append([x01, x99])
            notchan = np.ones(3, "bool")
            notchan[np.array(chan)] = False
            notchan = np.nonzero(notchan)[0]
            for c in notchan:
                for z in range(self.NZ):
                    self.saturation[c].append([0, 255.0])

            img_norm *= 255.0
            self.autobtn.setChecked(True)

            # assign to denoised channels
            self.stack_filtered = np.zeros(
                (self.NZ, self.Lyr, self.Lxr, self.stack.shape[-1]), "float32"
            )
            for i, c in enumerate(chan[: img_norm.shape[-1]]):
                for z in range(self.NZ):
                    self.stack_filtered[z, :, :, c] = img_norm[z, :, :, i]

            # make upsampled masks
            if model_type == "upsample":
                self.cellpix_orig = self.cellpix.copy()
                self.outpix_orig = self.outpix.copy()
                self.cellpix_resize = cv2.resize(
                    self.cellpix_orig[0],
                    (self.Lxr, self.Lyr),
                    interpolation=cv2.INTER_NEAREST,
                )[np.newaxis, :, :]
                outlines = masks_to_outlines(self.cellpix_resize[0])[np.newaxis, :, :]
                self.outpix_resize = outlines * self.cellpix_resize

            self.restore = model_name

            # draw plot
            if model_type == "upsample":
                self.resize = True
            else:
                self.resize = False
            self.draw_layer()
            self.update_layer()
            self.update_scale()
            # if denoised in grayscale, show in grayscale
            if channels[0] == 0:
                self.RGBDropDown.setCurrentIndex(4)

            self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(
                True
            )
            self.ViewDropDown.setCurrentIndex(self.ViewDropDown.count() - 1)

            self.update_plot()

        except Exception as e:
            print("ERROR: %s" % e)

    def compute_segmentation(self, custom=False, model_name=None, load_model=True):
        """
        Computes the segmentation of images using a deep learning model.

            This method initializes the model (if needed) and processes the image stack to generate
            segmentation masks and flow data. It updates the progress of the computation and handles
            possible exceptions during the process. The results include the computed masks and flows,
            which can be resized to match the original image dimensions.

            Args:
                custom: Indicates whether a custom model should be used.
                model_name: The name of the model to be loaded.
                load_model: A flag that determines if the model should be loaded at the start of the
                             computation.

            Returns:
                None: The method does not return any value. It updates instance variables with the
                      computed segmentation results and manages the progress of the operation.
        """
        self.progress.setValue(0)
        try:
            tic = time.time()
            self.clear_all()
            self.flows = [[], [], []]
            if load_model:
                self.initialize_model(model_name=model_name, custom=custom)
            self.progress.setValue(10)
            do_3D = self.load_3D
            stitch_threshold = (
                float(self.stitch_threshold.text())
                if not isinstance(self.stitch_threshold, float)
                else self.stitch_threshold
            )
            anisotropy = (
                float(self.anisotropy.text())
                if not isinstance(self.anisotropy, float)
                else self.anisotropy
            )
            flow3D_smooth = (
                float(self.flow3D_smooth.text())
                if not isinstance(self.flow3D_smooth, float)
                else self.flow3D_smooth
            )
            min_size = (
                int(self.min_size.text())
                if not isinstance(self.min_size, int)
                else self.min_size
            )
            resample = (
                self.resample.isChecked()
                if not isinstance(self.resample, bool)
                else self.resample
            )

            do_3D = False if stitch_threshold > 0.0 else do_3D

            channels = self.get_channels()
            if self.restore is not None and self.restore != "filter":
                data = self.stack_filtered.copy().squeeze()
            else:
                data = self.stack.copy().squeeze()
            flow_threshold, cellprob_threshold = self.get_thresholds()
            self.diameter = float(self.Diameter.text())
            niter = max(0, int(self.niter.text()))
            niter = None if niter == 0 else niter
            normalize_params = self.get_normalize_params()
            print(normalize_params)
            try:
                masks, flows = self.model.eval(
                    data,
                    channels=channels,
                    diameter=self.diameter,
                    cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold,
                    do_3D=do_3D,
                    niter=niter,
                    normalize=normalize_params,
                    stitch_threshold=stitch_threshold,
                    anisotropy=anisotropy,
                    resample=resample,
                    flow3D_smooth=flow3D_smooth,
                    min_size=min_size,
                    progress=self.progress,
                    z_axis=0 if self.NZ > 1 else None,
                )[:2]
            except Exception as e:
                print("NET ERROR: %s" % e)
                self.progress.setValue(0)
                return

            self.progress.setValue(75)

            # convert flows to uint8 and resize to original image size
            flows_new = []
            flows_new.append(flows[0].copy())  # RGB flow
            flows_new.append(
                (np.clip(normalize99(flows[2].copy()), 0, 1) * 255).astype("uint8")
            )  # cellprob
            if self.load_3D:
                if stitch_threshold == 0.0:
                    flows_new.append((flows[1][0] / 10 * 127 + 127).astype("uint8"))
                else:
                    flows_new.append(np.zeros(flows[1][0].shape, dtype="uint8"))

            if not self.load_3D:
                if self.restore and "upsample" in self.restore:
                    self.Ly, self.Lx = self.Lyr, self.Lxr

                if flows_new[0].shape[-3:-1] != (self.Ly, self.Lx):
                    self.flows = []
                    for j in range(len(flows_new)):
                        self.flows.append(
                            resize_image(
                                flows_new[j],
                                Ly=self.Ly,
                                Lx=self.Lx,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )
                else:
                    self.flows = flows_new
            else:
                if not resample:
                    self.flows = []
                    Lz, Ly, Lx = self.NZ, self.Ly, self.Lx
                    Lz0, Ly0, Lx0 = flows_new[0].shape[:3]
                    print("GUI_INFO: resizing flows to original image size")
                    for j in range(len(flows_new)):
                        flow0 = flows_new[j]
                        if Ly0 != Ly:
                            flow0 = resize_image(
                                flow0,
                                Ly=Ly,
                                Lx=Lx,
                                no_channels=flow0.ndim == 3,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        if Lz0 != Lz:
                            flow0 = np.swapaxes(
                                resize_image(
                                    np.swapaxes(flow0, 0, 1),
                                    Ly=Lz,
                                    Lx=Lx,
                                    no_channels=flow0.ndim == 3,
                                    interpolation=cv2.INTER_NEAREST,
                                ),
                                0,
                                1,
                            )
                        self.flows.append(flow0)
                else:
                    self.flows = flows_new

            # add first axis
            if self.NZ == 1:
                masks = masks[np.newaxis, ...]
                self.flows = [
                    self.flows[n][np.newaxis, ...] for n in range(len(self.flows))
                ]

            self.logger.info(
                "%d cells found with model in %0.3f sec"
                % (len(np.unique(masks)[1:]), time.time() - tic)
            )
            self.progress.setValue(80)
            z = 0

            io._masks_to_gui(self, masks, outlines=None)
            self.masksOn = True
            self.MCheckBox.setChecked(True)
            self.keepMask.setEnabled(True)
            self.saveMasks.setEnabled(True)

            self.progress.setValue(100)
            if self.restore != "filter" and self.restore is not None:
                self.compute_saturation()
            if not do_3D and not stitch_threshold > 0:
                self.recompute_masks = True
            else:
                self.recompute_masks = False
        except Exception as e:
            print("ERROR: %s" % e)
