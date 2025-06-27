"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtGui import QPainter, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QRadioButton,
    QWidget,
    QDialog,
    QButtonGroup,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QGridLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QDialogButtonBox,
    QComboBox,
    QCheckBox,
)
import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph import Point
import numpy as np
import pathlib, os


def stylesheet():
    """
    Generate a stylesheet for the application's UI components.

    This method returns a string containing CSS-like rules used to style
    various UI elements, such as tooltips, combo boxes, scroll areas,
    group boxes, and push buttons.

    Returns:
        str: A string representing the stylesheet, containing styles
        for different UI components.
    """
    return """
        QToolTip { 
                            background-color: black; 
                            color: white; 
                            border: black solid 1px
                            }
        QComboBox {color: white;
                    background-color: rgb(40,40,40);}
                    QComboBox::item:enabled { color: white;
                    background-color: rgb(40,40,40);
                    selection-color: white;
                    selection-background-color: rgb(50,100,50);}
                    QComboBox::item:!enabled {
                            background-color: rgb(40,40,40);
                            color: rgb(100,100,100);
                        }
        QScrollArea > QWidget > QWidget
                {
                    background: transparent;
                    border: none;
                    margin: 0px 0px 0px 0px;
                } 
                           
        QGroupBox 
            { border: 1px solid white; color: rgb(255,255,255);
                           border-radius: 6px;
                            margin-top: 8px;
                            padding: 0px 0px;}            
                           
        QPushButton:pressed {Text-align: center; 
                             background-color: rgb(150,50,150); 
                             border-color: white;
                             color:white;}
                            QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }
        QPushButton:!pressed {Text-align: center; 
                               background-color: rgb(50,50,50);
                                border-color: white;
                               color:white;}
                                QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }
        QPushButton:disabled {Text-align: center; 
                             background-color: rgb(30,30,30);
                             border-color: white;
                              color:rgb(80,80,80);}
                               QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }
                        
        """


class DarkPalette(QtGui.QPalette):
    """Class that inherits from pyqtgraph.QtGui.QPalette and renders dark colours for the application.
    (from pykilosort/kilosort4)
    """

    def __init__(self):
        """
        Initialize the class and set up the user interface.

            This method initializes the parent class and sets up the necessary
            components for the user interface. It is typically called when
            an instance of the class is created.

            Parameters:
                None

            Returns:
                None
        """
        QtGui.QPalette.__init__(self)
        self.setup()

    def setup(self):
        """
        Sets the color palette for the application's GUI elements.

            This method configures various colors for different user interface components, including the background, text, buttons, tooltips, and highlighted elements, ensuring a consistent and visually appealing aesthetic across the application.

            Parameters:
                None

            Returns:
                None
        """
        self.setColor(QtGui.QPalette.Window, QtGui.QColor(40, 40, 40))
        self.setColor(QtGui.QPalette.WindowText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Base, QtGui.QColor(34, 27, 24))
        self.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 50, 47))
        self.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Text, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 50, 47))
        self.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
        self.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
        self.setColor(
            QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(128, 128, 128)
        )
        self.setColor(
            QtGui.QPalette.Disabled,
            QtGui.QPalette.ButtonText,
            QtGui.QColor(128, 128, 128),
        )
        self.setColor(
            QtGui.QPalette.Disabled,
            QtGui.QPalette.WindowText,
            QtGui.QColor(128, 128, 128),
        )


def create_channel_choose():
    """
    Creates UI components for channel selection in image segmentation.

        This method initializes two combo boxes for selecting channels related to
        cytoplasm and nuclei for image segmentation. It also provides labels and tooltips
        to guide the user in choosing the appropriate channels.

        Returns:
            A tuple containing two elements:
                - A list of QComboBox instances for selecting the primary and optional channels.
                - A list of QLabel instances with descriptions for each combo box.
    """
    # choose channel
    ChannelChoose = [QComboBox(), QComboBox()]
    ChannelLabels = []
    ChannelChoose[0].addItems(["gray", "red", "green", "blue"])
    ChannelChoose[1].addItems(["none", "red", "green", "blue"])
    cstr = ["chan to segment:", "chan2 (optional): "]
    for i in range(2):
        ChannelLabels.append(QLabel(cstr[i]))
        if i == 0:
            ChannelLabels[i].setToolTip(
                "this is the channel in which the cytoplasm or nuclei exist \
            that you want to segment"
            )
            ChannelChoose[i].setToolTip(
                "this is the channel in which the cytoplasm or nuclei exist \
            that you want to segment"
            )
        else:
            ChannelLabels[i].setToolTip(
                "if <em>cytoplasm</em> model is chosen, and you also have a \
            nuclear channel, then choose the nuclear channel for this option"
            )
            ChannelChoose[i].setToolTip(
                "if <em>cytoplasm</em> model is chosen, and you also have a \
            nuclear channel, then choose the nuclear channel for this option"
            )

    return ChannelChoose, ChannelLabels


class ModelButton(QPushButton):
    """
    A button widget designed to interact with a specified model for segmentation tasks.

    This class provides a button that, when pressed, triggers the segmentation computation
    using a model name associated with the button, allowing for interactive model operations.

    Methods:
        __init__
        press

    Attributes:
        None

    The __init__ method initializes the button with parameters including the parent object,
    model name, and display text. The press method triggers the segmentation computation
    by invoking a function on the parent object, utilizing the model name for processing.
    """

    def __init__(self, parent, model_name, text):
        """
        Initializes the widget with specified parameters.

            This method sets up the initial state of the widget by configuring
            its font, text, and connection to a click event. It also handles
            the model name to ensure it conforms to expected values.

            Args:
                parent: The parent object from which the widget derives its properties.
                model_name: The name of the model associated with this widget.
                text: The text to be displayed on the widget.

            Returns:
                None
        """
        super().__init__()
        self.setEnabled(False)
        self.setText(text)
        self.setFont(parent.boldfont)
        self.clicked.connect(lambda: self.press(parent))
        self.model_name = model_name if "cyto3" not in model_name else "cyto3"

    def press(self, parent):
        """
        Triggers the segmentation computation using the specified model.

            This method calls the compute_segmentation function on the provided parent
            object, utilizing the model name associated with the current instance.

            Args:
                parent: An object that contains the compute_segmentation method.

            Returns:
                None
        """
        parent.compute_segmentation(model_name=self.model_name)


class DenoiseButton(QPushButton):
    """
    A button that facilitates the denoising process for a given parent object.

    This class manages the initialization and pressing actions of a button that executes
    various denoising operations based on a specified model type. The button is initially
    disabled and becomes active when appropriate conditions are met.

    Methods:
        __init__
        press

    Attributes:
        None

    The __init__ method sets up the button with its text, font, and event connections.
    The press method processes the parent object according to the model type, either
    applying filter parameters, computing saturation, denoising models, or clearing settings.
    This interaction modifies the state of the parent object accordingly.
    """

    def __init__(self, parent, text):
        """
        Initializes a button with specified settings.

            This method sets up the button's initial state, including its
            text, font, and event connections. The button is initially
            disabled and is configured to trigger an action when clicked.

            Args:
                parent: The parent object that contains the font setting.
                text: The text to display on the button.

            Returns:
                None
        """
        super().__init__()
        self.setEnabled(False)
        self.model_type = text
        self.setText(text)
        self.setFont(parent.medfont)
        self.clicked.connect(lambda: self.press(parent))

    def press(self, parent):
        """
        Handles the processing of the specified parent object based on the model type.

            This method determines the action to take for the given parent object based on the
            type of model being used. If the model type is "filter", it checks the filter parameters
            and either applies them, alerts the user if no settings are defined, or computes saturation.
            If the model type is not "none", it computes a denoise model. Otherwise, it clears the
            restore settings.

            Args:
                parent: The parent object that will be processed. It is expected to have methods
                        for getting normalization parameters, computing saturation, denoising models,
                        clearing restore settings, and setting the restore button.

            Returns:
                None: This method does not return a value, but modifies the state of the parent object
                based on the model type and parameters.
        """
        if self.model_type == "filter":
            parent.restore = "filter"
            normalize_params = parent.get_normalize_params()
            if (
                normalize_params["sharpen_radius"] == 0
                and normalize_params["smooth_radius"] == 0
                and normalize_params["tile_norm_blocksize"] == 0
            ):
                print(
                    "GUI_ERROR: no filtering settings on (use custom filter settings)"
                )
                parent.restore = None
                return
            parent.restore = self.model_type
            parent.compute_saturation()
        elif self.model_type != "none":
            parent.compute_denoise_model(model_type=self.model_type)
        else:
            parent.clear_restore()
        parent.set_restore_button()


class TrainWindow(QDialog):
    """
    Manages the training settings for machine learning models.

    This class provides a user interface dialog for configuring training parameters, including
    model selection, channel selection, and various hyperparameters. It allows users to input
    and modify training settings such as model choice, learning rate, and weight decay.

    Methods:
        __init__:
        accept:

    Attributes:
        None

    The __init__ method sets up the training settings dialog and initializes the interface, while
    the accept method retrieves user input from the dialog, updates the training parameters in
    the specified parent object, and indicates the completion of the configuration process.
    """

    def __init__(self, parent, model_strings):
        """
        Initializes the training settings dialog.

            This method sets up a dialog for configuring training parameters, including model selection,
            channel selection, and various hyperparameters. It creates a user interface for the user
            to input and modify the training settings, providing options for model choice, learning rate,
            weight decay, and more.

            Args:
                parent: The parent widget that this dialog belongs to.
                model_strings: A list of model names to populate the model selection dropdown.

            Returns:
                None
        """
        super().__init__(parent)
        self.setGeometry(100, 100, 900, 550)
        self.setWindowTitle("train settings")
        self.win = QWidget(self)
        self.l0 = QGridLayout()
        self.win.setLayout(self.l0)

        yoff = 0
        qlabel = QLabel("train model w/ images + _seg.npy in current folder >>")
        qlabel.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))

        qlabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, yoff, 0, 1, 2)

        # choose initial model
        yoff += 1
        self.ModelChoose = QComboBox()
        self.ModelChoose.addItems(model_strings)
        self.ModelChoose.addItems(["scratch"])
        self.ModelChoose.setFixedWidth(150)
        self.ModelChoose.setCurrentIndex(parent.training_params["model_index"])
        self.l0.addWidget(self.ModelChoose, yoff, 1, 1, 1)
        qlabel = QLabel("initial model: ")
        qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, yoff, 0, 1, 1)

        # choose channels
        self.ChannelChoose, self.ChannelLabels = create_channel_choose()
        for i in range(2):
            yoff += 1
            self.ChannelChoose[i].setFixedWidth(150)
            self.ChannelChoose[i].setCurrentIndex(
                parent.ChannelChoose[i].currentIndex()
            )
            self.l0.addWidget(self.ChannelLabels[i], yoff, 0, 1, 1)
            self.l0.addWidget(self.ChannelChoose[i], yoff, 1, 1, 1)

        # choose parameters
        labels = ["learning_rate", "weight_decay", "n_epochs", "model_name"]
        self.edits = []
        yoff += 1
        for i, label in enumerate(labels):
            qlabel = QLabel(label)
            qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.l0.addWidget(qlabel, i + yoff, 0, 1, 1)
            self.edits.append(QLineEdit())
            self.edits[-1].setText(str(parent.training_params[label]))
            self.edits[-1].setFixedWidth(200)
            self.l0.addWidget(self.edits[-1], i + yoff, 1, 1, 1)

        yoff += 1
        use_SGD = "SGD"
        self.useSGD = QCheckBox(f"{use_SGD}")
        self.useSGD.setToolTip(
            "use SGD, if unchecked uses AdamW (recommended learning_rate then 0.001)"
        )
        self.useSGD.setChecked(True)
        self.l0.addWidget(self.useSGD, i + yoff, 1, 1, 1)

        yoff += len(labels)

        yoff += 1
        self.use_norm = QCheckBox(f"use restored/filtered image")
        self.use_norm.setChecked(True)
        # self.l0.addWidget(self.use_norm, yoff, 0, 2, 4)

        yoff += 2
        qlabel = QLabel(
            "(to remove files, click cancel then remove \nfrom folder and reopen train window)"
        )
        self.l0.addWidget(qlabel, yoff, 0, 2, 4)

        # click button
        yoff += 3
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(lambda: self.accept(parent))
        self.buttonBox.rejected.connect(self.reject)
        self.l0.addWidget(self.buttonBox, yoff, 0, 1, 4)

        # list files in folder
        qlabel = QLabel("filenames")
        qlabel.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.l0.addWidget(qlabel, 0, 4, 1, 1)
        qlabel = QLabel("# of masks")
        qlabel.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.l0.addWidget(qlabel, 0, 5, 1, 1)

        for i in range(10):
            if i > len(parent.train_files) - 1:
                break
            elif i == 9 and len(parent.train_files) > 10:
                label = "..."
                nmasks = "..."
            else:
                label = os.path.split(parent.train_files[i])[-1]
                nmasks = str(parent.train_labels[i].max())
            qlabel = QLabel(label)
            self.l0.addWidget(qlabel, i + 1, 4, 1, 1)
            qlabel = QLabel(nmasks)
            qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.l0.addWidget(qlabel, i + 1, 5, 1, 1)

    def accept(self, parent):
        """
        Sets training parameters for the model and signals completion.

        This method initializes the training parameters using values from
        the user interface and sets them in the provided parent object.
        It also indicates that the operation is complete.

        Args:
            parent: The object where the training parameters will be stored.

        Returns:
            None
        """
        # set training params
        parent.training_params = {
            "model_index": self.ModelChoose.currentIndex(),
            "learning_rate": float(self.edits[0].text()),
            "weight_decay": float(self.edits[1].text()),
            "n_epochs": int(self.edits[2].text()),
            "model_name": self.edits[3].text(),
            "SGD": True if self.useSGD.isChecked() else False,
            "channels": [
                self.ChannelChoose[0].currentIndex(),
                self.ChannelChoose[1].currentIndex(),
            ],
            # "use_norm": True if self.use_norm.isChecked() else False,
        }
        self.done(1)


class ExampleGUI(QDialog):
    """
    A simple graphical user interface (GUI) for demonstrating basic GUI functionalities.

    This class initializes a main window, sets its properties, and
    loads an image for display within the GUI.

    Methods:
        __init__: Initializes the ExampleGUI window and loads an image.

    Attributes:
        parent: The parent widget for this GUI, if applicable.
        window: The main window of the GUI.
        image_label: The label used to display the loaded image.

    The methods and attributes of this class work together to create
    an interactive GUI environment, allowing for basic image display and
    user interaction.
    """

    def __init__(self, parent=None):
        """
        Initialize the ExampleGUI window.

            This method sets up the main GUI window by defining its geometry,
            title, and layout. It also loads and displays an image in the GUI.

            Args:
                parent: The parent widget, if any, for this GUI. Defaults to None.

            Returns:
                None
        """
        super(ExampleGUI, self).__init__(parent)
        self.setGeometry(100, 100, 1300, 900)
        self.setWindowTitle("GUI layout")
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        guip_path = pathlib.Path.home().joinpath(".cellpose", "cellpose_gui.png")
        guip_path = str(guip_path.resolve())
        pixmap = QPixmap(guip_path)
        label = QLabel(self)
        label.setPixmap(pixmap)
        pixmap.scaled
        layout.addWidget(label, 0, 0, 1, 1)


class HelpWindow(QDialog):
    """
    A window that displays help content for the application.

    This class is responsible for initializing and presenting a help interface,
    which includes loading help content from an HTML file and displaying it to
    the user in a structured layout.

    Methods:
        __init__: Initializes the HelpWindow with specified configurations.

    Attributes:
        parent: The parent widget of the HelpWindow, or None if there is no parent.

    The __init__ method configures the size, title, and layout of the HelpWindow
    and loads content from an HTML file to provide users with necessary assistance.
    """

    def __init__(self, parent=None):
        """
        Initialize the HelpWindow.

            This method sets up the HelpWindow by configuring its size, title,
            and layout, and loading help content from an HTML file.

            Args:
                parent: The parent widget of the HelpWindow, or None if there is no parent.

            Returns:
                None: This method does not return a value.
        """
        super(HelpWindow, self).__init__(parent)
        self.setGeometry(100, 50, 700, 1000)
        self.setWindowTitle("cellpose help")
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)

        text_file = pathlib.Path(__file__).parent.joinpath("guihelpwindowtext.html")
        with open(str(text_file.resolve()), "r") as f:
            text = f.read()

        label = QLabel(text)
        label.setFont(QtGui.QFont("Arial", 8))
        label.setWordWrap(True)
        layout.addWidget(label, 0, 0, 1, 1)
        self.show()


class TrainHelpWindow(QDialog):
    """
    A class to create a training help window for the application.

    This window displays instructions and information related to the
    training process. It utilizes an HTML file to present content in
    a user-friendly manner, structured within a grid layout.

    Methods:
        __init__: Initializes the TrainHelpWindow with specified parameters.

    Attributes:
        parent: The parent widget to which the training help window is attached.
        geometry: The size and position of the window.
        title: The title of the training help window.
        html_content: The content loaded from an HTML file for display.

    The __init__ method sets up the window's geometry, title, and
    loads content from an HTML file to provide users with helpful
    training instructions. The layout is managed using a grid for
    organized content display.
    """

    def __init__(self, parent=None):
        """
        Initialize the TrainHelpWindow.

            This method sets up the training help window with a specific geometry,
            a title, and loads content from an HTML file to display instructions
            regarding training. It organizes the layout with a grid and includes
            a QLabel for showing the text content.

            Args:
                parent: The parent widget, if any, to which this window will be attached.

            Returns:
                None
        """
        super(TrainHelpWindow, self).__init__(parent)
        self.setGeometry(100, 50, 700, 300)
        self.setWindowTitle("training instructions")
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)

        text_file = pathlib.Path(__file__).parent.joinpath(
            "guitrainhelpwindowtext.html"
        )
        with open(str(text_file.resolve()), "r") as f:
            text = f.read()

        label = QLabel(text)
        label.setFont(QtGui.QFont("Arial", 8))
        label.setWordWrap(True)
        layout.addWidget(label, 0, 0, 1, 1)
        self.show()


class ViewBoxNoRightDrag(pg.ViewBox):
    """
    A class that represents a view box with customized mouse interactions and zooming capabilities.

    Methods:
        __init__: Initializes an instance of the class.
        keyPressEvent: Captures key presses for zooming functionalities.

    Attributes:
        None

    The __init__ method initializes the view box with parameters to configure aspects such as
    border style, aspect ratio locking, and mouse interaction. The keyPressEvent method handles
    keyboard inputs to facilitate navigation through a zooming stack.
    """

    def __init__(
        self,
        parent=None,
        border=None,
        lockAspect=False,
        enableMouse=True,
        invertY=False,
        enableMenu=True,
        name=None,
        invertX=False,
    ):
        """
        Initializes an instance of the class.

            This constructor initializes the view box with optional parameters for border style,
            aspect ratio locking, mouse interaction, and other configuration options.

            Args:
                parent: The parent object to which this instance belongs.
                border: The border configuration for the view box.
                lockAspect: A flag indicating whether to lock the aspect ratio.
                enableMouse: A flag to enable or disable mouse interactions.
                invertY: A flag to invert the Y-axis.
                enableMenu: A flag to enable or disable the menu.
                name: An optional name for the instance.
                invertX: A flag to invert the X-axis.

            Returns:
                None
        """
        pg.ViewBox.__init__(
            self,
            None,
            border,
            lockAspect,
            enableMouse,
            invertY,
            enableMenu,
            name,
            invertX,
        )
        self.parent = parent
        self.axHistoryPointer = -1

    def keyPressEvent(self, ev):
        """
        This routine should capture key presses in the current view box.
        The following events are implemented:
        +/= : moves forward in the zooming stack (if it exists)
        - : moves backward in the zooming stack (if it exists)

        """
        ev.accept()
        if ev.text() == "-":
            self.scaleBy([1.1, 1.1])
        elif ev.text() in ["+", "="]:
            self.scaleBy([0.9, 0.9])
        else:
            ev.ignore()


class ImageDraw(pg.ImageItem):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    GraphicsObject displaying an image. Optimized for rapid update (ie video display).
    This item displays either a 2D numpy array (height, width) or
    a 3D array (height, width, RGBa). This array is optionally scaled (see
    :func:`setLevels <pyqtgraph.ImageItem.setLevels>`) and/or colored
    with a lookup table (see :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`)
    before being displayed.
    ImageItem is frequently used in conjunction with
    :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>` or
    :class:`HistogramLUTWidget <pyqtgraph.HistogramLUTWidget>` to provide a GUI
    for controlling the levels and lookup table used to display the image.
    """

    sigImageChanged = QtCore.Signal()

    def __init__(self, image=None, viewbox=None, parent=None, **kargs):
        """
        Initializes an instance of the ImageDraw class.

            This method sets up the initial state of the ImageDraw object,
            configuring parameters such as the image, viewbox, and parent.
            It also initializes default values for levels, lookup tables,
            and drawing parameters.

            Args:
                image: The image to be drawn upon (optional).
                viewbox: The viewport dimensions (optional).
                parent: The parent object that holds reference to this ImageDraw
                        instance (optional).
                **kargs: Additional keyword arguments for flexibility.

            Returns:
                None
        """
        super(ImageDraw, self).__init__()
        # self.image=None
        # self.viewbox=viewbox
        self.levels = np.array([0, 255])
        self.lut = None
        self.autoDownsample = False
        self.axisOrder = "row-major"
        self.removable = False

        self.parent = parent
        # kernel[1,1] = 1
        self.setDrawKernel(kernel_size=self.parent.brush_size)
        self.parent.current_stroke = []
        self.parent.in_stroke = False

    def mouseClickEvent(self, ev):
        """
        Handles mouse click events for drawing and interacting with cells.

            This method processes mouse click events, differentiating between right-click
            and left-click actions, and updates the state of the associated graphical interface
            accordingly. It allows for starting and ending strokes, selecting or unselecting
            cells, deleting or merging cells based on modifier keys, and handles adjustments
            to the current stroke state.

            Args:
                ev: The mouse event containing information about the click, including the button
                    pressed and the position of the click.

            Returns:
                None
        """
        if (
            self.parent.masksOn or self.parent.outlinesOn
        ) and not self.parent.removing_region:
            is_right_click = ev.button() == QtCore.Qt.RightButton
            if (
                self.parent.loaded
                and (
                    is_right_click
                    or ev.modifiers() & QtCore.Qt.ShiftModifier
                    and not ev.double()
                )
                and not self.parent.deleting_multiple
            ):
                if not self.parent.in_stroke:
                    ev.accept()
                    self.create_start(ev.pos())
                    self.parent.stroke_appended = False
                    self.parent.in_stroke = True
                    self.drawAt(ev.pos(), ev)
                else:
                    ev.accept()
                    self.end_stroke()
                    self.parent.in_stroke = False
            elif not self.parent.in_stroke:
                y, x = int(ev.pos().y()), int(ev.pos().x())
                if y >= 0 and y < self.parent.Ly and x >= 0 and x < self.parent.Lx:
                    if ev.button() == QtCore.Qt.LeftButton and not ev.double():
                        idx = self.parent.cellpix[self.parent.currentZ][y, x]
                        if idx > 0:
                            if ev.modifiers() & QtCore.Qt.ControlModifier:
                                # delete mask selected
                                self.parent.remove_cell(idx)
                            elif ev.modifiers() & QtCore.Qt.AltModifier:
                                self.parent.merge_cells(idx)
                            elif (
                                self.parent.masksOn
                                and not self.parent.deleting_multiple
                            ):
                                self.parent.unselect_cell()
                                self.parent.select_cell(idx)
                            elif self.parent.deleting_multiple:
                                if idx in self.parent.removing_cells_list:
                                    self.parent.unselect_cell_multi(idx)
                                    self.parent.removing_cells_list.remove(idx)
                                else:
                                    self.parent.select_cell_multi(idx)
                                    self.parent.removing_cells_list.append(idx)

                        elif self.parent.masksOn and not self.parent.deleting_multiple:
                            self.parent.unselect_cell()

    def mouseDragEvent(self, ev):
        """
        Handle mouse drag events.

            This method is invoked when a mouse drag event occurs.
            It ignores the incoming event to prevent further processing.

            Args:
                ev: The event object representing the mouse drag event.

            Returns:
                None: This method does not return any value.
        """
        ev.ignore()
        return

    def hoverEvent(self, ev):
        """
        Handles hover events for drawing operations.

            This method processes the hover events to manage drawing strokes,
            including determining whether to continue a stroke based on the
            position of the hover event. If the drawing process is in progress,
            it checks if the current position is at the start of the stroke to
            determine if the stroke should end.

            Args:
                ev: The event object containing information about the hover event.

            Returns:
                None: This method does not return a value.
        """
        # QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
        if self.parent.in_stroke:
            if self.parent.in_stroke:
                # continue stroke if not at start
                self.drawAt(ev.pos())
                if self.is_at_start(ev.pos()):
                    # self.parent.in_stroke = False
                    self.end_stroke()
        else:
            ev.acceptClicks(QtCore.Qt.RightButton)
            # ev.acceptClicks(QtCore.Qt.LeftButton)

    def create_start(self, pos):
        """
        Creates a scatter plot item at the specified position.

            This method initializes a scatter plot item at the given position
            and adds it to the parent plot item. The scatter plot item is
            styled with a specified pen color and size.

            Args:
                pos: The position where the scatter plot item will be created,
                      containing x and y coordinates.

            Returns:
                None: This method does not return any value.
        """
        self.scatter = pg.ScatterPlotItem(
            [pos.x()],
            [pos.y()],
            pxMode=False,
            pen=pg.mkPen(color=(255, 0, 0), width=self.parent.brush_size),
            size=max(3 * 2, self.parent.brush_size * 1.8 * 2),
            brush=None,
        )
        self.parent.p0.addItem(self.scatter)

    def is_at_start(self, pos):
        """
        Determines if the current position is considered to be at the start of a stroke.

            This method checks whether the user has returned to the starting position of
            a stroke after moving away from it. It calculates distances based on the
            brush size and a series of strokes to determine if the user has left the
            initial point and then returned back within a certain threshold.

            Args:
                pos: The current position to evaluate.

            Returns:
                bool: True if the current position is considered at the start,
                      otherwise False.
        """
        thresh_out = max(6, self.parent.brush_size * 3)
        thresh_in = max(3, self.parent.brush_size * 1.8)
        # first check if you ever left the start
        if len(self.parent.current_stroke) > 3:
            stroke = np.array(self.parent.current_stroke)
            dist = (
                ((stroke[1:, 1:] - stroke[:1, 1:][np.newaxis, :, :]) ** 2).sum(axis=-1)
            ) ** 0.5
            dist = dist.flatten()
            # print(dist)
            has_left = (dist > thresh_out).nonzero()[0]
            if len(has_left) > 0:
                first_left = np.sort(has_left)[0]
                has_returned = (dist[max(4, first_left + 1) :] < thresh_in).sum()
                if has_returned > 0:
                    return True
                else:
                    return False
            else:
                return False

    def end_stroke(self):
        """
        Finalize the current stroke and update the stroke records.

            This method removes the current stroke's graphical representation,
            appends the stroke data to the list of strokes if it hasn't been added
            already, and updates the current point set. It also handles automatic
            saving of stroke sets based on the autosave flag.

            Parameters:
                None

            Returns:
                None
        """
        self.parent.p0.removeItem(self.scatter)
        if not self.parent.stroke_appended:
            self.parent.strokes.append(self.parent.current_stroke)
            self.parent.stroke_appended = True
            self.parent.current_stroke = np.array(self.parent.current_stroke)
            ioutline = self.parent.current_stroke[:, 3] == 1
            self.parent.current_point_set.append(
                list(self.parent.current_stroke[ioutline])
            )
            self.parent.current_stroke = []
            if self.parent.autosave:
                self.parent.add_set()
        if (
            len(self.parent.current_point_set)
            and len(self.parent.current_point_set[0]) > 0
            and self.parent.autosave
        ):
            self.parent.add_set()
        self.parent.in_stroke = False

    def tabletEvent(self, ev):
        """
        Handles tablet events.

            This method processes events from a tablet input device, such as
            pen pressure and pointer type. Currently, the implementation is a
            placeholder.

            Args:
                ev: The event object containing details of the tablet event.

            Returns:
                None
        """
        pass
        # print(ev.device())
        # print(ev.pointerType())
        # print(ev.pressure())

    def drawAt(self, pos, ev=None):
        """
        Draws a stroke at the specified position on the canvas.

            This method modifies the internal image representation by applying
            a drawing kernel at a given position. It updates the current stroke
            with the coordinates of the drawn area and ensures that any drawn
            areas do not exceed the boundaries of the canvas.

            Args:
                pos: The position (coordinates) on the canvas where the stroke should be drawn.
                ev: An optional event parameter that may be pertinent for specific
                    drawing contexts (default is None).

            Returns:
                None: This method modifies the state of the object and does not return a value.
        """
        mask = self.strokemask
        stroke = self.parent.current_stroke
        pos = [int(pos.y()), int(pos.x())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0, dk.shape[0]]
        sy = [0, dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0] + dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1] + dk.shape[1]]
        kcent = kc.copy()
        if tx[0] <= 0:
            sx[0] = 0
            sx[1] = kc[0] + 1
            tx = sx
            kcent[0] = 0
        if ty[0] <= 0:
            sy[0] = 0
            sy[1] = kc[1] + 1
            ty = sy
            kcent[1] = 0
        if tx[1] >= self.parent.Ly - 1:
            sx[0] = dk.shape[0] - kc[0] - 1
            sx[1] = dk.shape[0]
            tx[0] = self.parent.Ly - kc[0] - 1
            tx[1] = self.parent.Ly
            kcent[0] = tx[1] - tx[0] - 1
        if ty[1] >= self.parent.Lx - 1:
            sy[0] = dk.shape[1] - kc[1] - 1
            sy[1] = dk.shape[1]
            ty[0] = self.parent.Lx - kc[1] - 1
            ty[1] = self.parent.Lx
            kcent[1] = ty[1] - ty[0] - 1

        ts = (slice(tx[0], tx[1]), slice(ty[0], ty[1]))
        ss = (slice(sx[0], sx[1]), slice(sy[0], sy[1]))
        self.image[ts] = mask[ss]

        for ky, y in enumerate(np.arange(ty[0], ty[1], 1, int)):
            for kx, x in enumerate(np.arange(tx[0], tx[1], 1, int)):
                iscent = np.logical_and(kx == kcent[0], ky == kcent[1])
                stroke.append([self.parent.currentZ, x, y, iscent])
        self.updateImage()

    def setDrawKernel(self, kernel_size=3):
        """
        Sets the draw kernel used for rendering operations.

            This method initializes a drawing kernel of the specified size, which is used for operations such
            as drawing shapes or applying effects. The kernel is created as a square matrix filled with ones,
            and additional attributes for the kernel's center and corresponding masks are also set.

            Args:
                kernel_size: The size of the square kernel to be created. Defaults to 3.

            Returns:
                None: This method does not return a value. It modifies the object's internal state by setting
                the draw kernel and related masks.
        """
        bs = kernel_size
        kernel = np.ones((bs, bs), np.uint8)
        self.drawKernel = kernel
        self.drawKernelCenter = [
            int(np.floor(kernel.shape[0] / 2)),
            int(np.floor(kernel.shape[1] / 2)),
        ]
        onmask = 255 * kernel[:, :, np.newaxis]
        offmask = np.zeros((bs, bs, 1))
        opamask = 100 * kernel[:, :, np.newaxis]
        self.redmask = np.concatenate((onmask, offmask, offmask, onmask), axis=-1)
        self.strokemask = np.concatenate((onmask, offmask, onmask, opamask), axis=-1)
