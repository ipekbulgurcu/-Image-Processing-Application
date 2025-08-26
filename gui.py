import os
import sys
from typing import Union

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QFont, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QAction,
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextBrowser,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from skimage import img_as_ubyte
from skimage.transform import resize

from conversion_operations import (
    BinaryThresholdOperation,
    GrayscaleOperation,
    HsvOperation,
)
from edge_detection_operations import (
    EdgeDetectionDialog,
    PrewittOperation,
    RobertsOperation,
    ScharrOperation,
    SobelOperation,
)
from edit import EditActionsHandler
from file import FileActionsHandler
from segmentation_operations import (
    ChanVeseOperation,
    MorphSnakesOperation,
    MultiOtsuOperation,
)

# Yeni dosyalardan içe aktarıyoruz
from worker import WorkerThread
from progress_dialog import ProgressPopup
from operation_handler import OperationHandler
from image_operations import ImageOperations


class MainWindow(QMainWindow):
    """Main application window for Image Processing and Lane Tracking."""

    def __init__(self):
        """Initializes the main window and its UI components."""
        super().__init__()
        self.setWindowTitle("Image Processing Lab")
        
        # Uygulama ikonunu ayarlama
        app_icon = QIcon("icons/app.png")
        self.setWindowIcon(app_icon)
        
        # Windows görev çubuğu için özel TaskBar Icon ayarı
        try:
            import ctypes
            myappid = 'IPEK.ImageProcessingLab.App.1.0'  # Unique app ID
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception as e:
            self._logMessage(f"Windows taskbar icon ayarlanamadı: {str(e)}", "warning")
        
        self.resize(1280, 800)
        self.worker_thread = None
        self.progress_popup = None

        # Image state
        self.sourceFilePath = None
        self.outputFilePath = None
        self.current_source_image: Union[np.ndarray, None] = None
        self.current_output_image: Union[np.ndarray, None] = None

        # Operation handling
        self.operation_handler = OperationHandler(self)
        self.image_operations = ImageOperations(self.operation_handler)
        
        # UI components
        self.progress_popup = ProgressPopup(self)
        self.worker_thread = None

        self.logBox = None
        self.sourcePixmapLabel = None
        self.outputPixmapLabel = None
        self.clearSourceButton = None
        self.clearOutputButton = None
        self.undoButton = None
        self.redoButton = None
        self.openSourceButton = None
        self.saveOutputButton = None
        self.saveAsOutputButton = None
        self.exportSourceButton = None
        self.exportOutputButton = None
        self.clearSourcePanelButton = None
        self.clearOutputPanelButton = None
        self.categoryList = None
        self.operationsStack = None

        self.rgbToGrayButton = None
        self.rgbToHsvButton = None
        self.multiOtsuButton = None
        self.chanVeseButton = None
        self.morphSnakesButton = None
        self.robertsButton = None
        self.sobelButton = None
        self.scharrButton = None
        self.prewittButton = None
        self.applyBinaryThresholdButton = None

        self.fileMenu = None
        self.editMenu = None
        self.conversionMenu = None
        self.segmentationMenu = None
        self.edgeDetectionMenu = None

        self.fileToolBar = None
        self.sourceTitleLabel = None
        self.outputTitleLabel = None
        self.statusBar = None

        self.edit_handler = EditActionsHandler(self)
        self.file_handler = FileActionsHandler(self)

        self._createMenuBar()
        self._createToolBars()
        self._createCentralWidget()
        self._createStatusBar()
        self._applyStyles()
        self._connectActions()
        self._setInitialState()

        self._logMessage("Application started. Ready.")
        self.show()

    def _logMessage(self, message, level="info"):
        """Appends a message to the log box with level indication."""
        prefix_map = {
            "info": "[INFO]",
            "success": "[OK]",
            "warning": "[WARN]",
            "error": "[ERROR]",
        }
        prefix = prefix_map.get(level.lower(), "[INFO]")
        if self.logBox:
            color_map = {"success": "#388E3C", "warning": "#FFA000", "error": "#B3261E"}
            text_color = color_map.get(level.lower(), "#1C1B1F")
            log_entry = f'<span style="color:{text_color};white-space:pre">{prefix} {message}</span><br>'
            self.logBox.insertHtml(log_entry)
            self.logBox.moveCursor(self.logBox.textCursor().End)
            QApplication.processEvents()

    def _createMenuBar(self):
        """Creates the main menu bar and its menus/actions."""
        menuBar = self.menuBar()
        menuBar.setObjectName("MainMenuBar")

        self.fileMenu = menuBar.addMenu("&File")
        self.file_handler.add_actions_to_menu(self.fileMenu)

        self.editMenu = menuBar.addMenu("&Edit")
        self.edit_handler.add_actions_to_menu(self.editMenu)

        self.conversionMenu = menuBar.addMenu("&Conversion")
        self.rgbToGrayAction = QAction("RGB -> Grayscale", self)
        self.rgbToGrayAction.setToolTip("Convert the image to grayscale.")
        self.conversionMenu.addAction(self.rgbToGrayAction)
        self.rgbToHsvAction = QAction("RGB -> HSV", self)
        self.rgbToHsvAction.setToolTip("Convert the image from RGB to HSV color space.")
        self.conversionMenu.addAction(self.rgbToHsvAction)

        self.segmentationMenu = menuBar.addMenu("&Segmentation")
        self.multiOtsuAction = QAction("Multi-Otsu Thresholding", self)
        self.multiOtsuAction.setToolTip(
            "Segment the image using Multi-Otsu thresholding."
        )
        self.segmentationMenu.addAction(self.multiOtsuAction)
        self.chanVeseAction = QAction("Chan-Vese Segmentation", self)
        self.chanVeseAction.setToolTip(
            "Segment the image using the Chan-Vese algorithm."
        )
        self.segmentationMenu.addAction(self.chanVeseAction)
        self.morphSnakesAction = QAction("Morphological Snakes", self)
        self.morphSnakesAction.setToolTip(
            "Segment the image using Morphological Snakes (ACWE)."
        )
        self.segmentationMenu.addAction(self.morphSnakesAction)

        self.edgeDetectionMenu = menuBar.addMenu("&Edge Detection")
        self.robertsAction = QAction("Roberts", self)
        self.robertsAction.setToolTip("Apply Roberts cross edge detection.")
        self.edgeDetectionMenu.addAction(self.robertsAction)
        self.sobelAction = QAction("Sobel", self)
        self.sobelAction.setToolTip("Apply Sobel edge detection.")
        self.edgeDetectionMenu.addAction(self.sobelAction)
        self.scharrAction = QAction("Scharr", self)
        self.scharrAction.setToolTip("Apply Scharr edge detection.")
        self.edgeDetectionMenu.addAction(self.scharrAction)
        self.prewittAction = QAction("Prewitt", self)
        self.prewittAction.setToolTip("Apply Prewitt edge detection.")
        self.edgeDetectionMenu.addAction(self.prewittAction)

    def _createToolBars(self):
        """Creates the toolbars for common actions."""
        self.fileToolBar = self.addToolBar("File/Edit")
        self.fileToolBar.setObjectName("MainToolBar")
        self.fileToolBar.setIconSize(QSize(30, 30))

        # --- File Buttons ---
        self.openSourceButton = QToolButton(self)
        self.openSourceButton.setIcon(QIcon("icons/folder.png"))
        self.openSourceButton.setToolTip("Open Source Image (Ctrl+O)")
        self.fileToolBar.addWidget(self.openSourceButton)

        self.saveOutputButton = QToolButton(self)
        self.saveOutputButton.setIcon(QIcon("icons/save_output.png"))
        self.saveOutputButton.setToolTip("Save Output (Ctrl+S)")
        self.fileToolBar.addWidget(self.saveOutputButton)

        self.saveAsOutputButton = QToolButton(self)
        self.saveAsOutputButton.setIcon(QIcon("icons/save_as.png"))
        self.saveAsOutputButton.setToolTip("Save Output As... (Ctrl+Shift+S)")
        self.fileToolBar.addWidget(self.saveAsOutputButton)

        self.exportSourceButton = QToolButton(self)
        self.exportSourceButton.setIcon(QIcon("icons/export_source.png"))
        self.exportSourceButton.setText("Exp Src")
        self.exportSourceButton.setToolTip("Export Source Image (.jpg <-> .png)")
        self.fileToolBar.addWidget(self.exportSourceButton)

        self.exportOutputButton = QToolButton(self)
        self.exportOutputButton.setIcon(QIcon("icons/export_output.png"))
        self.exportOutputButton.setText("Exp Out")
        self.exportOutputButton.setToolTip("Export Output Image (.jpg <-> .png)")
        self.fileToolBar.addWidget(self.exportOutputButton)

        self.fileToolBar.addSeparator()

        # --- Edit Buttons ---
        self.undoButton = QToolButton(self)
        self.undoButton.setIcon(QIcon("icons/undo.png"))
        self.undoButton.setToolTip("Undo (Ctrl+Z)")
        self.fileToolBar.addWidget(self.undoButton)

        self.redoButton = QToolButton(self)
        self.redoButton.setIcon(QIcon("icons/redo.png"))
        self.redoButton.setToolTip("Redo (Ctrl+Y)")
        self.fileToolBar.addWidget(self.redoButton)

        self.fileToolBar.addSeparator()

        # --- Clear Buttons ---
        self.clearSourceButton = QToolButton(self)
        self.clearSourceButton.setIcon(QIcon("icons/clean.png"))
        self.clearSourceButton.setToolTip("Clear Source Image")
        self.fileToolBar.addWidget(self.clearSourceButton)

        self.clearOutputButton = QToolButton(self)
        self.clearOutputButton.setIcon(QIcon("icons/clean.png"))
        self.clearOutputButton.setToolTip("Clear Output Image")
        self.fileToolBar.addWidget(self.clearOutputButton)

    def _createCentralWidget(self):
        """Creates the central widget with sidebar, operations stack, and image areas."""
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        # Main Horizontal Splitter (Sidebar | Main Area)
        mainSplitter = QSplitter(Qt.Horizontal, centralWidget)
        mainSplitter.setObjectName("MainSplitter")
        mainSplitter.setHandleWidth(4)

        # --- Left Panel (Sidebar) ---
        leftPanel = QWidget(mainSplitter)
        leftPanel.setObjectName("LeftPanel")
        leftLayout = QVBoxLayout(leftPanel)
        leftLayout.setContentsMargins(5, 5, 5, 5)
        leftLayout.setSpacing(8)

        # Category List
        self.categoryList = QListWidget(leftPanel)
        self.categoryList.setObjectName("CategoryList")
        self.categoryList.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.categoryList.setMaximumHeight(100)  # Adjusted for 3 items
        self.categoryList.setSpacing(0)  # Remove spacing between items
        self.categoryList.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # Hide scrollbar if not needed
        listFont = QFont("Segoe UI", 9)
        self.categoryList.setFont(listFont)

        # Conversion Item
        conversion_item = QListWidgetItem(QIcon("icons/conversion.png"), "Conversion")
        conversion_item.setToolTip("Image format and color space conversions.")
        conversion_item.setSizeHint(QSize(conversion_item.sizeHint().width(), 30)) # Set fixed height
        self.categoryList.addItem(conversion_item)

        # Segmentation Item
        segmentation_item = QListWidgetItem(
            QIcon("icons/segmentation.png"), "Segmentation"
        )
        segmentation_item.setToolTip("Image segmentation algorithms.")
        segmentation_item.setSizeHint(QSize(segmentation_item.sizeHint().width(), 30)) # Set fixed height
        self.categoryList.addItem(segmentation_item)

        # Edge Detection Item
        edge_item = QListWidgetItem(QIcon("icons/edge_detection.png"), "Edge Detection")
        edge_item.setToolTip("Edge detection filters.")
        edge_item.setSizeHint(QSize(edge_item.sizeHint().width(), 30)) # Set fixed height
        self.categoryList.addItem(edge_item)

        leftLayout.addWidget(self.categoryList)

        # Operations Stack
        self.operationsStack = QStackedWidget(leftPanel)
        self.operationsStack.setObjectName("OperationsStack")
        self.operationsStack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._createOperationPages()  # *** Create pages AFTER stack exists ***
        leftLayout.addWidget(self.operationsStack)

        # --- Right Panel (Image Areas + Log) ---
        rightPanel = QWidget(mainSplitter)
        rightPanel.setObjectName("RightPanel")
        rightLayout = QVBoxLayout(rightPanel)
        rightLayout.setContentsMargins(5, 5, 5, 5)
        rightLayout.setSpacing(8)

        # Image Splitter (Source | Output)
        imageSplitter = QSplitter(Qt.Horizontal, rightPanel)
        imageSplitter.setObjectName("ImageSplitter")
        imageSplitter.setHandleWidth(4)
        imageSplitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Source Image Panel
        sourcePanel = QWidget(imageSplitter)
        sourcePanel.setObjectName("ImagePanel")
        sourceLayout = QVBoxLayout(sourcePanel)
        sourceLayout.setContentsMargins(4, 4, 4, 4)
        self.sourceTitleLabel = QLabel("Source")
        self.sourceTitleLabel.setObjectName("PanelTitleLabel")
        self.sourcePixmapLabel = QLabel("Drop or Open Image/Video")
        self.sourcePixmapLabel.setObjectName("PixmapLabel")
        self.sourcePixmapLabel.setAlignment(Qt.AlignCenter)
        self.sourcePixmapLabel.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Ignored
        )  # Important for scaling
        sourceLayout.addWidget(self.sourceTitleLabel)
        sourceLayout.addWidget(self.sourcePixmapLabel, 1)  # Stretch factor 1
        
        # Source Panel butonları için yatay düzen
        sourceButtonsLayout = QHBoxLayout()
        
        # Sadece Clear Source butonu kalıyor, undo/redo butonları kaldırıldı
        self.clearSourcePanelButton = QPushButton("Clear Source")
        self.clearSourcePanelButton.setObjectName("UtilityButton")
        self.clearSourcePanelButton.setIcon(QIcon("icons/clean.png"))
        sourceButtonsLayout.addWidget(self.clearSourcePanelButton)
        
        # Buton düzenini ana Source paneli düzenine ekliyoruz
        sourceLayout.addLayout(sourceButtonsLayout)
        
        # Output Image Panel
        outputPanel = QWidget(imageSplitter)
        outputPanel.setObjectName("ImagePanel")
        outputLayout = QVBoxLayout(outputPanel)
        outputLayout.setContentsMargins(4, 4, 4, 4)
        self.outputTitleLabel = QLabel("Output")
        self.outputTitleLabel.setObjectName("PanelTitleLabel")
        self.outputPixmapLabel = QLabel("Processing Result Area")
        self.outputPixmapLabel.setObjectName("PixmapLabel")
        self.outputPixmapLabel.setAlignment(Qt.AlignCenter)
        self.outputPixmapLabel.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Ignored
        )  # Important for scaling
        outputLayout.addWidget(self.outputTitleLabel)
        outputLayout.addWidget(self.outputPixmapLabel, 1)  # Stretch factor 1
        
        # Output Panel butonları için yatay düzen
        outputButtonsLayout = QHBoxLayout()
        
        # Undo butonu
        self.outputUndoButton = QPushButton("Undo")
        self.outputUndoButton.setObjectName("UtilityButton")
        self.outputUndoButton.setIcon(QIcon("icons/undo.png"))
        self.outputUndoButton.setToolTip("Undo last operation")
        outputButtonsLayout.addWidget(self.outputUndoButton)
        
        # Redo butonu
        self.outputRedoButton = QPushButton("Redo")
        self.outputRedoButton.setObjectName("UtilityButton")
        self.outputRedoButton.setIcon(QIcon("icons/redo.png"))
        self.outputRedoButton.setToolTip("Redo last undone operation")
        outputButtonsLayout.addWidget(self.outputRedoButton)
        
        # Clear Output butonu (mevcut butonu yeni düzene taşıyoruz)
        self.clearOutputPanelButton = QPushButton("Clear Output")
        self.clearOutputPanelButton.setObjectName("UtilityButton")
        self.clearOutputPanelButton.setIcon(QIcon("icons/clean.png"))
        outputButtonsLayout.addWidget(self.clearOutputPanelButton)
        
        # Buton düzenini ana Output paneli düzenine ekliyoruz
        outputLayout.addLayout(outputButtonsLayout)
        
        imageSplitter.addWidget(sourcePanel)
        imageSplitter.addWidget(outputPanel)
        imageSplitter.setSizes([600, 600])  # Initial equal sizing

        rightLayout.addWidget(imageSplitter, 1)  # Stretch factor 1

        # Log Area
        logGroup = QGroupBox("Application Log")
        logGroup.setObjectName("LogGroup")
        logLayout = QVBoxLayout(logGroup)
        logLayout.setContentsMargins(3, 3, 3, 3)
        self.logBox = QTextBrowser(logGroup)
        self.logBox.setObjectName("LogBox")
        self.logBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.logBox.setMaximumHeight(120)  # Limit height
        logLayout.addWidget(self.logBox)
        rightLayout.addWidget(logGroup)

        mainSplitter.addWidget(leftPanel)
        mainSplitter.addWidget(rightPanel)
        mainSplitter.setSizes([350, 1050])  # Initial sizing for sidebar and main area

        # Set the main splitter as the central widget's layout root
        centralLayout = QVBoxLayout(centralWidget)
        centralLayout.setContentsMargins(0, 0, 0, 0)
        centralLayout.addWidget(mainSplitter)

    def _createOperationPages(self):
        """Creates the QWidget pages for the operations QStackedWidget."""

        # Helper function to create a scrollable page
        def create_scrollable_page(content_layout_filler_func):
            page_container = QWidget()
            page_main_layout = QVBoxLayout(page_container)
            page_main_layout.setContentsMargins(
                0, 0, 0, 0
            )  # No margins for the container itself

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setObjectName("OperationScrollArea")
            # scroll_area.setStyleSheet("QScrollArea#OperationScrollArea { border: 0px; }") # Optional: remove border

            scroll_content_widget = (
                QWidget()
            )  # This widget will contain the actual parameters
            # content_layout will be set by the filler function
            # Example: content_layout = QVBoxLayout(scroll_content_widget)

            content_layout_filler_func(
                scroll_content_widget
            )  # Populate the scroll_content_widget

            scroll_area.setWidget(scroll_content_widget)
            page_main_layout.addWidget(scroll_area)
            # page_container.setLayout(page_main_layout) # Already set by QVBoxLayout(page_container)
            return page_container

        # --- Conversion Page ---
        def fill_conversion_page(parent_widget):
            conversion_layout = QVBoxLayout(
                parent_widget
            )  # Use parent_widget for layout
            conversion_layout.setContentsMargins(5, 5, 5, 5)
            conversion_layout.setSpacing(8)

            self.rgbToGrayButton = QPushButton(
                QIcon("icons/conversion.png"), "RGB -> Grayscale"
            )  # Added Icon
            self.rgbToGrayButton.setObjectName("OperationButton")
            self.rgbToGrayButton.setToolTip("Convert the current image to grayscale.")
            conversion_layout.addWidget(self.rgbToGrayButton)

            self.rgbToHsvButton = QPushButton(
                QIcon("icons/conversion.png"), "RGB -> HSV"
            )  # Added Icon
            self.rgbToHsvButton.setObjectName("OperationButton")
            self.rgbToHsvButton.setToolTip(
                "Convert the current image from RGB to HSV color space."
            )
            conversion_layout.addWidget(self.rgbToHsvButton)

            # Binary Threshold with Material Design 3 styling
            thresholdGroup = QGroupBox("Binary Threshold")
            thresholdGroup.setObjectName("MaterialCardContainer")
            thresholdLayout = QVBoxLayout()
            thresholdLayout.setContentsMargins(12, 12, 12, 12)
            thresholdLayout.setSpacing(10)

            # Threshold control widget
            threshold_widget = QWidget()
            threshold_widget.setObjectName("ThresholdWidget")
            threshold_layout = QVBoxLayout(threshold_widget)
            threshold_layout.setContentsMargins(0, 0, 0, 0)
            threshold_layout.setSpacing(8)

            # Threshold label
            label_layout = QHBoxLayout()
            threshold_label = QLabel("Threshold:")
            threshold_label.setObjectName("ParameterLabel")
            label_layout.addWidget(threshold_label)
            label_layout.addStretch()
            threshold_layout.addLayout(label_layout)

            # Slider and value row
            slider_value_layout = QHBoxLayout()

            # Minus button
            minus_btn = QPushButton("-")
            minus_btn.setObjectName("CircleButton")
            minus_btn.setFixedSize(36, 36)
            minus_btn.setToolTip("Decrease threshold value")
            slider_value_layout.addWidget(minus_btn)

            # Create the threshold value display and slider
            value_slider_widget = QWidget()
            value_slider_layout = QVBoxLayout(value_slider_widget)
            value_slider_layout.setContentsMargins(0, 0, 0, 0)
            value_slider_layout.setSpacing(2)

            # Value display
            self.binaryThresholdSpinBox = QDoubleSpinBox()
            self.binaryThresholdSpinBox.setRange(0.0, 1.0)
            self.binaryThresholdSpinBox.setValue(0.5)
            self.binaryThresholdSpinBox.setSingleStep(0.01)
            self.binaryThresholdSpinBox.setDecimals(2)
            self.binaryThresholdSpinBox.setAlignment(Qt.AlignCenter)
            self.binaryThresholdSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            self.binaryThresholdSpinBox.setObjectName("MaterialValueDisplay")
            value_slider_layout.addWidget(self.binaryThresholdSpinBox)

            # Slider
            self.binaryThresholdSlider = QSlider(Qt.Horizontal)
            self.binaryThresholdSlider.setRange(0, 100)
            self.binaryThresholdSlider.setValue(50)
            self.binaryThresholdSlider.setObjectName("ThresholdSlider")
            value_slider_layout.addWidget(self.binaryThresholdSlider)

            slider_value_layout.addWidget(value_slider_widget, 1)

            # Plus button
            plus_btn = QPushButton("+")
            plus_btn.setObjectName("CircleButton")
            plus_btn.setFixedSize(36, 36)
            plus_btn.setToolTip("Increase threshold value")
            slider_value_layout.addWidget(plus_btn)

            threshold_layout.addLayout(slider_value_layout)

            # Connect the buttons to threshold adjustment
            minus_btn.clicked.connect(
                lambda: self._decrementSpinnerValue(self.binaryThresholdSpinBox, 0.01)
            )
            plus_btn.clicked.connect(
                lambda: self._incrementSpinnerValue(self.binaryThresholdSpinBox, 0.01)
            )

            # Add the threshold widget to the group layout
            thresholdLayout.addWidget(threshold_widget)

            # Invert checkbox
            self.binaryThresholdInvertCheckbox = QCheckBox("Invert Output")
            self.binaryThresholdInvertCheckbox.setObjectName("MaterialCheckbox")
            thresholdLayout.addWidget(self.binaryThresholdInvertCheckbox)

            thresholdGroup.setLayout(thresholdLayout)

            # Apply button
            self.applyBinaryThresholdButton = QPushButton("Apply Binary Threshold")
            self.applyBinaryThresholdButton.setObjectName("OperationButton")
            self.applyBinaryThresholdButton.setToolTip(
                "Apply binary thresholding to the image using the specified threshold."
            )

            conversion_layout.addWidget(thresholdGroup)
            conversion_layout.addWidget(self.applyBinaryThresholdButton)
            conversion_layout.addStretch()

        conversionPageContainer = create_scrollable_page(fill_conversion_page)
        self.operationsStack.addWidget(conversionPageContainer)

        # --- Segmentation Page ---
        def fill_segmentation_page(parent_widget):
            segmentation_layout = QVBoxLayout(parent_widget)
            segmentation_layout.setContentsMargins(5, 5, 5, 5)
            segmentation_layout.setSpacing(8)

            # Multi-Otsu
            otsu_classes_widget = self._create_material_value_adjuster_widget(
                label_text="Number of Classes:",
                min_val=2,
                max_val=10,
                default_val=3,
                step_val=1,
                is_float=False,
                target_spinbox_attr="otsuClassesSpinBox",
                target_slider_attr="otsuClassesSlider",
            )
            segmentation_layout.addWidget(otsu_classes_widget)

            self.multiOtsuButton = QPushButton(
                QIcon("icons/segmentation.png"), "Apply Multi-Otsu"
            )  # Added Icon
            self.multiOtsuButton.setObjectName("OperationButton")
            self.multiOtsuButton.setToolTip(
                "Segment the image into multiple regions using Multi-Otsu thresholding."
            )
            segmentation_layout.addWidget(self.multiOtsuButton)

            # Chan-Vese Segmentation
            chanvese_iter_widget = self._create_material_value_adjuster_widget(
                label_text="Max Iterations:",
                min_val=50,
                max_val=1000,
                default_val=200,
                step_val=50,
                is_float=False,
                target_spinbox_attr="chanveseIterSpinBox",
                target_slider_attr="chanveseIterSlider",
            )
            segmentation_layout.addWidget(chanvese_iter_widget)

            chanvese_tol_widget = self._create_material_value_adjuster_widget(
                label_text="Tolerance:",
                min_val=0.0001,
                max_val=0.01,
                default_val=0.001,
                step_val=0.0001,
                is_float=True,
                decimals=4,
                target_spinbox_attr="chanveseTolSpinBox",
                target_slider_attr="chanveseTolSlider",
            )
            segmentation_layout.addWidget(chanvese_tol_widget)

            self.chanVeseButton = QPushButton(
                QIcon("icons/segmentation.png"), "Apply Chan-Vese"
            )  # Added Icon
            self.chanVeseButton.setObjectName("OperationButton")
            self.chanVeseButton.setToolTip(
                "Segment the image using the Chan-Vese active contour model."
            )
            segmentation_layout.addWidget(self.chanVeseButton)

            # Morphological Snakes
            morph_iter_widget = self._create_material_value_adjuster_widget(
                label_text="Iterations:",
                min_val=10,
                max_val=300,
                default_val=50,
                step_val=10,
                is_float=False,
                target_spinbox_attr="morphIterSpinBox",
                target_slider_attr="morphIterSlider",
            )
            segmentation_layout.addWidget(morph_iter_widget)

            morph_smooth_widget = self._create_material_value_adjuster_widget(
                label_text="Smoothing:",
                min_val=1,
                max_val=15,
                default_val=3,
                step_val=1,
                is_float=False,
                target_spinbox_attr="morphSmoothSpinBox",
                target_slider_attr="morphSmoothSlider",
            )
            segmentation_layout.addWidget(morph_smooth_widget)

            self.morphSnakesButton = QPushButton(
                QIcon("icons/segmentation.png"), "Apply Morph Snakes"
            )  # Added Icon
            self.morphSnakesButton.setObjectName("OperationButton")
            self.morphSnakesButton.setToolTip(
                "Segment the image using Morphological Snakes (Active Contours Without Edges)."
            )
            segmentation_layout.addWidget(self.morphSnakesButton)
            segmentation_layout.addStretch()

        segmentationPageContainer = create_scrollable_page(fill_segmentation_page)
        self.operationsStack.addWidget(segmentationPageContainer)

        # --- Edge Detection Page ---
        def fill_edge_page(parent_widget):
            edgeLayout = QVBoxLayout(parent_widget)
            edgeLayout.setContentsMargins(5, 5, 5, 5)
            edgeLayout.setSpacing(8)

            # Edge Detection Parameters
            edge_thresh_widget = self._create_material_value_adjuster_widget(
                label_text="Threshold:",
                min_val=0.0,
                max_val=1.0,
                default_val=0.1,
                step_val=0.01,
                is_float=True,
                decimals=2,
                slider_multiplier=100,
                target_spinbox_attr="edgeThresholdSpinBox",
                target_slider_attr="edgeThresholdSlider",
            )
            edgeLayout.addWidget(edge_thresh_widget)

            edge_sigma_widget = self._create_material_value_adjuster_widget(
                label_text="Sigma (Blur):",
                min_val=0.0,
                max_val=5.0,
                default_val=0.0,
                step_val=0.1,
                is_float=True,
                decimals=1,
                slider_multiplier=10,
                target_spinbox_attr="edgeSigmaSpinBox",
                target_slider_attr="edgeSigmaSlider",
            )
            edgeLayout.addWidget(edge_sigma_widget)

            # Parameter information labels
            info_widget = QWidget()
            info_layout = QVBoxLayout(info_widget)
            info_layout.setContentsMargins(12, 8, 12, 8)
            threshold_info = QLabel(
                "Threshold: Edge detection sensitivity (0.0 = auto)"
            )
            sigma_info = QLabel("Sigma: Gaussian blur strength (0.0 = no blur)")
            for info in [threshold_info, sigma_info]:
                info.setStyleSheet("color: gray; font-style: italic; font-size: 8pt;")
                info_layout.addWidget(info)
            edgeLayout.addWidget(info_widget)

            # Edge Detection Methods
            edgeMethodGroup = QGroupBox("Select Edge Detection Method")
            edgeMethodLayout = QVBoxLayout()
            self.robertsButton = QPushButton(
                QIcon("icons/edge_detection.png"), "Roberts"
            )  # Added Icon
            self.robertsButton.setObjectName("OperationButton")
            self.robertsButton.setToolTip(
                "Apply Roberts cross edge detection with the above parameters."
            )
            edgeMethodLayout.addWidget(self.robertsButton)
            self.sobelButton = QPushButton(
                QIcon("icons/edge_detection.png"), "Sobel"
            )  # Added Icon
            self.sobelButton.setObjectName("OperationButton")
            self.sobelButton.setToolTip(
                "Apply Sobel edge detection with the above parameters."
            )
            edgeMethodLayout.addWidget(self.sobelButton)
            self.scharrButton = QPushButton(
                QIcon("icons/edge_detection.png"), "Scharr"
            )  # Added Icon
            self.scharrButton.setObjectName("OperationButton")
            self.scharrButton.setToolTip(
                "Apply Scharr edge detection with the above parameters."
            )
            edgeMethodLayout.addWidget(self.scharrButton)
            self.prewittButton = QPushButton(
                QIcon("icons/edge_detection.png"), "Prewitt"
            )  # Added Icon
            self.prewittButton.setObjectName("OperationButton")
            self.prewittButton.setToolTip(
                "Apply Prewitt edge detection with the above parameters."
            )
            edgeMethodLayout.addWidget(self.prewittButton)
            edgeMethodGroup.setLayout(edgeMethodLayout)
            edgeLayout.addWidget(edgeMethodGroup)
            edgeLayout.addStretch()

        edgePageContainer = create_scrollable_page(fill_edge_page)
        self.operationsStack.addWidget(edgePageContainer)

    # Helper methods for value adjustment buttons
    def _incrementSpinnerValue(self, spinner, step_value):
        """Increments the spinner value by the given step."""
        if isinstance(spinner, QDoubleSpinBox):
            new_value = min(spinner.value() + step_value, spinner.maximum())
            spinner.setValue(new_value)
        else:  # QSpinBox
            new_value = min(spinner.value() + step_value, spinner.maximum())
            spinner.setValue(new_value)

    def _decrementSpinnerValue(self, spinner, step_value):
        """Decrements the spinner value by the given step."""
        if isinstance(spinner, QDoubleSpinBox):
            new_value = max(spinner.value() - step_value, spinner.minimum())
            spinner.setValue(new_value)
        else:  # QSpinBox
            new_value = max(spinner.value() - step_value, spinner.minimum())
            spinner.setValue(new_value)

    def _create_material_value_adjuster_widget(
        self,
        label_text,
        min_val,
        max_val,
        default_val,
        step_val,
        is_float=True,
        decimals=2,
        slider_multiplier=100,  # For float->int slider
        target_spinbox_attr=None,  # Attribute name to store the spinbox
        target_slider_attr=None,  # Attribute name to store the slider
    ):
        """Creates a Material Design 3 styled parameter widget with label, slider and +/- buttons."""
        # Main container
        container = QWidget()
        container.setObjectName("MaterialCardContainer")
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Parameter widget
        param_widget = QWidget()
        param_widget.setObjectName("ThresholdWidget")
        param_layout = QVBoxLayout(param_widget)
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setSpacing(8)

        # Label
        label_layout = QHBoxLayout()
        param_label = QLabel(label_text)
        param_label.setObjectName("ParameterLabel")
        label_layout.addWidget(param_label)
        label_layout.addStretch()
        param_layout.addLayout(label_layout)

        # Slider and value row
        slider_value_layout = QHBoxLayout()

        # Minus button
        minus_btn = QPushButton("-")
        minus_btn.setObjectName("CircleButton")
        minus_btn.setFixedSize(36, 36)
        minus_btn.setToolTip(f"Decrease {label_text.lower()}")
        slider_value_layout.addWidget(minus_btn)

        # Value and slider container
        value_slider_widget = QWidget()
        value_slider_layout = QVBoxLayout(value_slider_widget)
        value_slider_layout.setContentsMargins(0, 0, 0, 0)
        value_slider_layout.setSpacing(2)

        # Value display
        spin_box = QDoubleSpinBox() if is_float else QSpinBox()
        spin_box.setRange(min_val, max_val)
        spin_box.setValue(default_val)
        if is_float:
            spin_box.setSingleStep(step_val)
            spin_box.setDecimals(decimals)
        else:  # QSpinBox
            spin_box.setSingleStep(int(step_val))

        spin_box.setAlignment(Qt.AlignCenter)
        spin_box.setButtonSymbols(QAbstractSpinBox.NoButtons)
        spin_box.setObjectName("MaterialValueDisplay")
        value_slider_layout.addWidget(spin_box)

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setObjectName("ThresholdSlider")

        actual_min_val = min_val
        actual_max_val = max_val
        if min_val > max_val:  # Ensure min_val <= max_val for slider range logic
            actual_min_val, actual_max_val = max_val, min_val

        if is_float:
            slider.setRange(
                int(round(actual_min_val * slider_multiplier)),
                int(round(actual_max_val * slider_multiplier)),
            )
            slider.setValue(int(round(default_val * slider_multiplier)))
            slider.valueChanged.connect(
                lambda val, s=spin_box, mult=slider_multiplier: s.setValue(val / mult)
            )

            # Correctly define min_s_val and max_s_val using slider_multiplier
            min_s_val = int(round(actual_min_val * slider_multiplier))
            max_s_val = int(round(actual_max_val * slider_multiplier))
            spin_box.valueChanged.connect(
                lambda val, s=slider, mult=slider_multiplier, min_s_local=min_s_val, max_s_local=max_s_val: s.setValue(
                    max(min_s_local, min(max_s_local, int(round(val * mult))))
                )
            )
        else:  # Integer
            slider.setRange(int(actual_min_val), int(actual_max_val))
            slider.setValue(int(default_val))
            slider.valueChanged.connect(spin_box.setValue)
            spin_box.valueChanged.connect(slider.setValue)

        value_slider_layout.addWidget(slider)
        slider_value_layout.addWidget(value_slider_widget, 1)

        # Plus button
        plus_btn = QPushButton("+")
        plus_btn.setObjectName("CircleButton")
        plus_btn.setFixedSize(36, 36)
        plus_btn.setToolTip(f"Increase {label_text.lower()}")
        slider_value_layout.addWidget(plus_btn)

        # Connect buttons
        minus_btn.clicked.connect(
            lambda: self._decrementSpinnerValue(spin_box, step_val)
        )
        plus_btn.clicked.connect(
            lambda: self._incrementSpinnerValue(spin_box, step_val)
        )

        param_layout.addLayout(slider_value_layout)
        main_layout.addWidget(param_widget)

        # Store references if attribute names are provided
        if target_spinbox_attr:
            setattr(self, target_spinbox_attr, spin_box)
        if target_slider_attr:
            setattr(self, target_slider_attr, slider)

        return container

    def _create_value_adjuster_widget(
        self,
        min_val,
        max_val,
        default_val,
        step_val,
        is_float=True,
        decimals=2,
        slider_multiplier=100,  # For float->int slider
        target_spinbox_attr=None,  # Attribute name to store the spinbox
        target_slider_attr=None,  # Attribute name to store the slider
    ):
        widget = QWidget()
        h_layout = QHBoxLayout(widget)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(4)

        minus_btn = QPushButton("-")
        minus_btn.setObjectName("SpinnerButton")

        spin_box = QDoubleSpinBox() if is_float else QSpinBox()
        spin_box.setRange(min_val, max_val)
        spin_box.setValue(default_val)
        if is_float:
            spin_box.setSingleStep(step_val)
            spin_box.setDecimals(decimals)
        else:  # QSpinBox
            spin_box.setSingleStep(int(step_val))

        plus_btn = QPushButton("+")
        plus_btn.setObjectName("SpinnerButton")

        minus_btn.clicked.connect(
            lambda: self._decrementSpinnerValue(spin_box, step_val)
        )
        plus_btn.clicked.connect(
            lambda: self._incrementSpinnerValue(spin_box, step_val)
        )

        h_layout.addWidget(minus_btn)
        h_layout.addWidget(spin_box, 1)
        h_layout.addWidget(plus_btn)

        if target_spinbox_attr:
            setattr(self, target_spinbox_attr, spin_box)

        v_layout_container = QVBoxLayout()
        v_layout_container.setContentsMargins(0, 0, 0, 0)
        v_layout_container.setSpacing(2)
        v_layout_container.addWidget(widget)

        if target_slider_attr:
            slider = QSlider(Qt.Horizontal)
            actual_min_val = min_val
            actual_max_val = max_val
            if min_val > max_val:  # Ensure min_val <= max_val for slider range logic
                actual_min_val, actual_max_val = max_val, min_val

            if is_float:
                slider.setRange(
                    int(round(actual_min_val * slider_multiplier)),
                    int(round(actual_max_val * slider_multiplier)),
                )
                slider.setValue(int(round(default_val * slider_multiplier)))
                slider.valueChanged.connect(
                    lambda val, s=spin_box, mult=slider_multiplier: s.setValue(
                        val / mult
                    )
                )

                # Correctly define min_s_val and max_s_val using slider_multiplier from the outer scope
                min_s_val = int(round(actual_min_val * slider_multiplier))
                max_s_val = int(round(actual_max_val * slider_multiplier))
                spin_box.valueChanged.connect(
                    lambda val, s=slider, mult=slider_multiplier, min_s_local=min_s_val, max_s_local=max_s_val: s.setValue(
                        max(min_s_local, min(max_s_local, int(round(val * mult))))
                    )
                )
            else:  # Integer
                slider.setRange(int(actual_min_val), int(actual_max_val))
                slider.setValue(int(default_val))
                slider.valueChanged.connect(spin_box.setValue)
                spin_box.valueChanged.connect(slider.setValue)

            setattr(self, target_slider_attr, slider)
            v_layout_container.addWidget(slider)

        container_widget = QWidget()
        container_widget.setLayout(v_layout_container)
        return container_widget

    def _createStatusBar(self):
        """Creates the status bar."""
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)
        self.statusBar.setObjectName("MainStatusBar")

    def _createProgressPopup(self):
        """Creates or resets the progress popup dialog."""
        if self.progress_popup is None:
            self.progress_popup = ProgressPopup(self)
        else:
            self.progress_popup.reset()
        return self.progress_popup

    def _applyStyles(self):
        """Applies QSS styles for a modern look based on M3."""
        primary = "#6750A4"  # Purple
        on_primary = "#FFFFFF"
        primary_container = "#EADDFF"
        on_primary_container = "#21005D"
        secondary = "#9C27B0"  # Deeper purple
        on_secondary = "#FFFFFF"
        secondary_container = "#E8DEF8"
        on_secondary_container = "#1D192B"
        tertiary = "#E91E63"  # Pink
        on_tertiary = "#FFFFFF"
        tertiary_container = "#FFD8E4"
        on_tertiary_container = "#31111D"
        error = "#B3261E"
        on_error = "#FFFFFF"
        error_container = "#F9DEDC"
        on_error_container = "#410E0B"
        background = "#FFFBFE"
        on_background = "#1C1B1F"
        surface = "#FFFBFE"
        on_surface = "#1C1B1F"
        surface_variant = "#E7E0EC"
        on_surface_variant = "#49454F"
        outline = "#79747E"
        hover_primary = "#EADDFF"
        hover_secondary = "#E8DEF8"
        selected_primary_container = "#D0BCFF"
        disabled_bg = "#E0E0E0"
        disabled_fg = "#A0A0A0"

        styleSheet = f"""
            QMainWindow {{
                background-color: {background};
                font-family: "Segoe UI", Arial, sans-serif;
            }}
            QWidget#LeftPanel {{
                background-color: {surface};
                border-right: 1px solid {outline};
            }}
            QMenuBar#MainMenuBar {{
                background-color: {surface};
                border-bottom: 1px solid {outline};
                color: {on_surface};
                padding: 2px;
            }}
            QMenuBar#MainMenuBar::item {{
                padding: 5px 10px;
                background-color: transparent;
                color: {on_surface_variant};
            }}
            QMenuBar#MainMenuBar::item:selected {{
                background-color: {primary_container};
                color: {on_primary_container};
            }}
            QMenu {{
                background-color: {surface} !important;
                border: 1px solid {outline};
                padding: 5px;
                color: {on_surface};
            }}
            QMenu::item {{
                padding: 6px 20px;
                border-radius: 3px;
                background-color: {surface} !important;
                color: {on_surface};
            }}
            QMenu::item:selected {{
                background-color: {primary_container} !important;
                color: {on_primary_container};
            }}
            QMenu::item:disabled {{
                color: {disabled_fg};
            }}
            QMenu::separator {{
                height: 1px;
                background: {outline};
                margin: 4px 0px;
            }}
            QToolBar#MainToolBar {{
                background-color: {surface};
                border-bottom: 1px solid {outline};
                padding: 3px;
                spacing: 4px;
            }}
            QToolButton {{
                background-color: transparent;
                border: 1px solid transparent;
                padding: 5px;
                margin: 1px;
                border-radius: 4px;
                color: {on_surface_variant};
            }}
            QToolButton:hover {{
                background-color: {secondary_container};
                color: {on_secondary_container};
                border: 1px solid {secondary_container};
            }}
            QToolButton:pressed {{
                background-color: {selected_primary_container};
                color: {on_primary_container};
            }}
            QToolButton:disabled {{
                color: {disabled_fg};
                background-color: transparent;
                border: 1px solid transparent;
            }}
            QPushButton {{
                background-color: {primary};
                color: {on_primary};
                border: 0px;
                padding: 8px 12px;
                border-radius: 16px;
                font-size: 9pt;
                min-height: 28px;
                text-align: center;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {primary_container};
                color: {on_primary_container};
            }}
            QPushButton:pressed {{
                background-color: {selected_primary_container};
                color: {on_primary_container};
            }}
            QPushButton:checked {{
                background-color: {primary};
                color: {on_primary};
            }}
            QPushButton:disabled {{
                background-color: {disabled_bg};
                color: {disabled_fg};
            }}
            QPushButton#OperationButton {{
                background-color: {secondary};
                color: {on_secondary};
                text-align: left;
                padding-left: 15px;
                border-radius: 4px;
            }}
            QPushButton#OperationButton:hover {{
                background-color: {secondary_container};
                color: {on_secondary_container};
            }}
            QPushButton#OperationButton:pressed {{
                background-color: {secondary_container};
                color: {on_secondary_container};
            }}
            QPushButton#OperationButton:checked {{
                background-color: {secondary_container};
                color: {on_secondary_container};
                border: 1px solid {secondary};
            }}
            QPushButton#UtilityButton {{
                background-color: {tertiary_container};
                color: {on_tertiary_container};
                border-radius: 4px;
            }}
            QPushButton#UtilityButton:hover {{
                background-color: {tertiary};
                color: {on_tertiary};
            }}
            QPushButton#UtilityButton:pressed {{
                background-color: {tertiary};
                color: {on_tertiary};
            }}
            QPushButton#UtilityButton:disabled {{
                background-color: {disabled_bg};
                color: {disabled_fg};
            }}
            QPushButton#SpinnerButton {{
                background-color: {tertiary};
                color: {on_tertiary};
                min-width: 24px;
                max-width: 24px;
                min-height: 24px;
                max-height: 24px;
                padding: 0px;
                font-weight: bold;
                font-size: 14px;
                border-radius: 12px;
            }}
            QStatusBar#MainStatusBar {{
                background-color: {surface_variant};
                border-top: 1px solid {outline};
                color: {on_surface_variant};
                padding: 3px;
            }}
            QLabel#PanelTitleLabel {{
                font-weight: bold;
                font-size: 11pt;
                padding: 6px 4px;
                color: {primary};
                border-bottom: 1px solid {outline};
                margin-bottom: 5px;
            }}
            QWidget#ImagePanel {{
                background-color: {surface};
                border: 1px solid {outline};
                border-radius: 4px;
            }}
            QLabel#PixmapLabel {{
                background-color: {surface_variant};
                border: 1px dashed {outline};
                color: {on_surface_variant};
                border-radius: 3px;
                min-height: 200px;
                padding: 10px;
                font-style: normal;
                text-align: center;
            }}
            QSplitter::handle {{
                background-color: {outline};
            }}
            QSplitter::handle:horizontal {{
                width: 1px;
                margin: 0px 2px;
            }}
            QSplitter::handle:vertical {{
                height: 1px;
                margin: 2px 0px;
            }}
            QSplitter::handle:hover {{
                background-color: {primary};
            }}
            QGroupBox {{ /* General GroupBox Style */
                font-size: 9pt;
                font-weight: normal;
                color: {on_surface_variant};
                border: 1px solid {outline};
                border-radius: 4px;
                margin-top: 15px; /* Space for title */
                background-color: {surface};
                padding: 10px; /* Internal padding */
                padding-top: 15px; /* More padding at top */
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                margin-left: 10px; /* Indent title */
                margin-top: -1px; /* Pull title up slightly */
                padding: 2px 5px; /* Padding around title */
                background-color: {surface}; /* Match background */
                border-radius: 3px;
                color: {on_surface_variant};
                font-weight: bold;
            }}
            QGroupBox#LogGroup {{ /* Specific Log Group Style Overrides */
                margin-top: 20px;
                padding: 5px;
                padding-top: 10px;
            }}
            QGroupBox#LogGroup::title {{
                margin-left: 12px;
                margin-top: -6px;
                padding: 4px 8px;
                background-color: {surface_variant};
                color: {on_surface_variant};
            }}
            QTextBrowser#LogBox {{
                background-color: {surface};
                border: 0px;
                color: {on_surface};
                font-family: Consolas, Courier New, monospace;
                font-size: 9pt;
                padding: 5px;
            }}
            QListWidget#CategoryList {{
                border: 1px solid {outline};
                background-color: {surface};
                padding: 1px;      /* Minimal padding for the list widget itself */
                border-radius: 4px;
                outline: 0;
            }}
            QListWidget#CategoryList::item {{
                padding: 4px 8px;  /* Reduced padding for items */
                margin: 0px;       /* No margin between items */
                border-radius: 3px;
                color: {on_surface};
            }}
            QListWidget#CategoryList::item:hover {{
                background-color: {hover_secondary};
                color: {on_secondary_container};
            }}
            QListWidget#CategoryList::item:selected {{
                background-color: {secondary_container};
                color: {on_secondary_container};
                font-weight: bold;
                border-left: 3px solid {secondary};
                padding-left: 7px;
            }}
            QWidget#OperationsStack {{
                background-color: {surface};
                border: 0px;
                border-radius: 0px;
                padding: 5px;
            }}

            /* --- Material Design 3 Slider --- */
            QSlider::groove:horizontal {{
                border-radius: 2px; /* Slightly rounded track ends */
                height: 4px;        /* M3 typical track height */
                background: {surface_variant}; /* Inactive track color */
                margin: 0px;
            }}
            QSlider::sub-page:horizontal {{ /* Active part of the track */
                background: {primary};
                border-radius: 2px;
                height: 4px;
            }}
            QSlider::add-page:horizontal {{ /* Inactive part of the track */
                background: {surface_variant};
                border-radius: 2px;
                height: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {primary}; /* Handle color */
                border: 2px solid {surface}; /* Border to make it pop from the track */
                width: 20px;   /* M3 standard handle size */
                height: 20px;
                margin: -8px 0; /* Vertically center 20px handle on 4px track */
                border-radius: 10px; /* Circular handle */
            }}
            QSlider::handle:horizontal:hover {{
                background: {primary_container}; /* Lighter primary for hover */
                border: 2px solid {primary_container};
            }}
            QSlider::handle:horizontal:pressed {{
                background: {selected_primary_container}; /* More prominent for pressed */
                border: 2px solid {selected_primary_container};
            }}
            /* --- End Material Design 3 Slider --- */

            QSpinBox, QDoubleSpinBox {{
                padding: 6px 8px;
                border: 1px solid {outline};
                border-radius: 12px;
                background-color: {surface};
                color: {on_surface};
                min-width: 70px;
                font-weight: 500;
                selection-background-color: {primary_container};
            }}
            QSpinBox:hover, QDoubleSpinBox:hover {{
                border: 1px solid {primary};
                background-color: {surface_variant};
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 2px solid {primary};
                padding: 5px 7px;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 24px;
                height: 12px;
                border: 0px;
                border-top-right-radius: 8px;
                background-color: {surface_variant};
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 24px;
                height: 12px;
                border: 0px;
                border-bottom-right-radius: 8px;
                background-color: {surface_variant};
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: {primary_container};
            }}
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
                background-color: {primary};
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                width: 12px;
                height: 6px;
                background: {on_surface_variant};
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                width: 12px;
                height: 6px;
                background: {on_surface_variant};
            }}
            QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover,
            QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
                background: {on_primary_container};
            }}
            
            /* --- Material Design 3 Binary Threshold Components --- */
            QDoubleSpinBox#MaterialValueDisplay {{
                background-color: transparent;
                border: none;
                font-size: 16px;
                font-weight: 500;
                color: {on_primary_container};
                padding: 4px;
                text-align: center;
            }}
            
            QWidget#ThresholdWidget {{
                background-color: {surface};
                border-radius: 8px;
                padding: 8px;
            }}
            
            QLabel#ParameterLabel {{
                font-size: 14px;
                font-weight: 500;
                color: {on_surface_variant};
                margin-bottom: 4px;
            }}
            
            QPushButton#CircleButton {{
                background-color: {tertiary};
                color: {on_tertiary};
                border-radius: 18px;
                font-size: 16px;
                font-weight: bold;
                padding: 0px;
            }}
            
            QPushButton#CircleButton:hover {{
                background-color: {secondary_container};
                color: {on_secondary_container};
            }}
            
            QPushButton#CircleButton:pressed {{
                background-color: {secondary};
                color: {on_secondary};
            }}
            
            QSlider#ThresholdSlider::groove:horizontal {{
                border-radius: 3px;
                height: 6px;
                background: {surface_variant};
            }}
            
            QSlider#ThresholdSlider::sub-page:horizontal {{
                background: {primary};
                border-radius: 3px;
                height: 6px;
            }}
            
            QSlider#ThresholdSlider::add-page:horizontal {{
                background: {surface_variant};
                border-radius: 3px;
                height: 6px;
            }}
            
            QSlider#ThresholdSlider::handle:horizontal {{
                background: {secondary};
                border: 2px solid {surface};
                width: 26px;
                height: 26px;
                margin: -10px 0;
                border-radius: 13px;
            }}
            
            QSlider#ThresholdSlider::handle:horizontal:hover {{
                background: {primary_container};
                border: 2px solid {primary};
            }}
            
            QSlider#ThresholdSlider::handle:horizontal:pressed {{
                background: {selected_primary_container};
                border: 2px solid {primary};
            }}
            
            QWidget#MaterialCardContainer {{
                background-color: {surface};
                border-radius: 12px;
                border: 1px solid {outline};
            }}
            
            QCheckBox#MaterialCheckbox {{
                font-size: 14px;
                color: {on_surface_variant};
                spacing: 8px;
                margin-top: 8px;
            }}
            
            QCheckBox#MaterialCheckbox::indicator {{
                width: 20px;
                height: 20px;
                border: 2px solid {outline};
                border-radius: 2px;
            }}
            
            QCheckBox#MaterialCheckbox::indicator:checked {{
                background-color: {primary};
                border: 2px solid {primary};
            }}
            
            QCheckBox#MaterialCheckbox::indicator:hover {{
                border: 2px solid {primary};
            }}
            /* --- End Material Design 3 Binary Threshold Components --- */
            
            QCheckBox {{
                spacing: 5px;
                color: {on_surface};
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {outline};
                border-radius: 3px;
                background-color: {surface};
            }}
            QCheckBox::indicator:checked {{
                background-color: {primary};
                border: 1px solid {primary};
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {primary};
            }}
            QCheckBox:disabled {{
                color: {disabled_fg};
            }}
            QCheckBox::indicator:disabled {{
                background-color: {disabled_bg};
                border: 1px solid {disabled_fg};
            }}
            QFormLayout {{
                horizontal-spacing: 10px;
                vertical-spacing: 8px;
            }}
            QLabel {{
                color: {on_surface};
                padding-top: 4px;
            }}
        """
        self.setStyleSheet(styleSheet)

    def _connectActions(self):
        """Connect actions and signals to slots."""
        # File menu connections are now handled by FileActionsHandler
        if self.openSourceButton:
            self.openSourceButton.clicked.connect(self.file_handler.open_source)
        if self.saveOutputButton:
            self.saveOutputButton.clicked.connect(self.file_handler.save_output)
        if self.saveAsOutputButton:
            self.saveAsOutputButton.clicked.connect(self.file_handler.save_output_as)
        if self.exportSourceButton:
            self.exportSourceButton.clicked.connect(self.file_handler.export_source)
        if self.exportOutputButton:
            self.exportOutputButton.clicked.connect(self.file_handler.export_output)
            
        # Edit menu eylemlerini edit_handler'a bağla
        if self.edit_handler:
            # Edit menüsündeki eylemler - edit_handler içinde oluşturulmuş olmalı
            pass  # ileride doğrudan edit_handler'a bağlantı yapılabilir
        
        # Panel butonları için bağlantılar
        self.clearSourcePanelButton.clicked.connect(self.edit_handler.clear_source)
        self.clearOutputPanelButton.clicked.connect(self.edit_handler.clear_output)
        self.outputUndoButton.clicked.connect(self._undo)
        self.outputRedoButton.clicked.connect(self._redo)
        # Source için undo/redo butonları şimdilik bağlanmadı (devre dışı olduğu için)

        # Connect Edit Toolbar buttons to edit_handler methods
        if self.undoButton:
            self.undoButton.clicked.connect(self._undo)
        if self.redoButton:
            self.redoButton.clicked.connect(self._redo)
        if self.clearSourceButton:
            self.clearSourceButton.clicked.connect(self.edit_handler.clear_source)
        if self.clearOutputButton:
            self.clearOutputButton.clicked.connect(self.edit_handler.clear_output)
            
        # Çıkış eylemleri
        if hasattr(self.file_handler, "exitAction"):
            self.file_handler.exitAction.triggered.connect(self.close)

        # Conversion menu - Yeni image_operations'a bağlayalım
        self.rgbToGrayAction.triggered.connect(self.image_operations.apply_grayscale)
        self.rgbToHsvAction.triggered.connect(lambda: self.image_operations.apply_hsv())
        # Binary Threshold Apply button - bunu da güncelleyelim
        if hasattr(self, "applyBinaryThresholdButton") and self.applyBinaryThresholdButton:
            self.applyBinaryThresholdButton.clicked.connect(
                lambda: self.image_operations.apply_binary_threshold(
                    self.binaryThresholdSpinBox.value(),
                    self.binaryThresholdInvertCheckbox.isChecked()
                )
            )

        # Segmentation menu - Yeni image_operations'a bağlayalım
        self.multiOtsuAction.triggered.connect(
            lambda: self.image_operations.apply_multi_otsu(
                self.otsuClassesSpinBox.value() if hasattr(self, "otsuClassesSpinBox") else 3
            )
        )
        self.chanVeseAction.triggered.connect(
            lambda: self.image_operations.apply_chan_vese(
                max_iter=self.chanveseIterSpinBox.value() if hasattr(self, "chanveseIterSpinBox") else 200,
                tol=self.chanveseTolSpinBox.value() if hasattr(self, "chanveseTolSpinBox") else 0.001
            )
        )
        self.morphSnakesAction.triggered.connect(
            lambda: self.image_operations.apply_morph_snakes(
                iterations=self.morphIterSpinBox.value() if hasattr(self, "morphIterSpinBox") else 50,
                smoothing=self.morphSmoothSpinBox.value() if hasattr(self, "morphSmoothSpinBox") else 3
            )
        )

        # Edge Detection menu - Yeni image_operations'a bağlayalım
        self.robertsAction.triggered.connect(
            lambda: self.image_operations.apply_roberts()
        )
        self.sobelAction.triggered.connect(
            lambda: self.image_operations.apply_sobel(
                **self._get_edge_detection_params()
            )
        )
        self.scharrAction.triggered.connect(
            lambda: self.image_operations.apply_scharr(
                **self._get_edge_detection_params()
            )
        )
        self.prewittAction.triggered.connect(
            lambda: self.image_operations.apply_prewitt(
                **self._get_edge_detection_params()
            )
        )

        self.categoryList.currentRowChanged.connect(
            self.operationsStack.setCurrentIndex
        )

        # Connect parameter changes
        if hasattr(self, "edgeThresholdSpinBox") and self.edgeThresholdSpinBox:
            self.edgeThresholdSpinBox.valueChanged.connect(self._edge_params_changed)
        else:
            print("DEBUG: edgeThresholdSpinBox not found during _connectActions!")

        if hasattr(self, "edgeSigmaSpinBox") and self.edgeSigmaSpinBox:
            self.edgeSigmaSpinBox.valueChanged.connect(self._edge_params_changed)
        else:
            print("DEBUG: edgeSigmaSpinBox not found during _connectActions!")

        if hasattr(self, "edgeThresholdSlider") and self.edgeThresholdSlider:
            self.edgeThresholdSlider.valueChanged.connect(
                self._updateThresholdFromSlider
            )
        else:
            print("DEBUG: edgeThresholdSlider not found during _connectActions!")

        # edgeThresholdSpinBox's valueChanged is already connected to _edge_params_changed.
        if (
            hasattr(self, "edgeThresholdSpinBox")
            and self.edgeThresholdSpinBox
            and hasattr(self, "edgeThresholdSlider")
            and self.edgeThresholdSlider
        ):
            self.edgeThresholdSpinBox.valueChanged.connect(self._updateThresholdSlider)
        else:
            if (
                not hasattr(self, "edgeThresholdSpinBox")
                or not self.edgeThresholdSpinBox
            ):
                print(
                    "DEBUG: edgeThresholdSpinBox (for slider update) not found during _connectActions!"
                )
            if not hasattr(self, "edgeThresholdSlider") or not self.edgeThresholdSlider:
                print(
                    "DEBUG: edgeThresholdSlider (for spinbox update) not found during _connectActions!"
                )

        if hasattr(self, "edgeSigmaSlider") and self.edgeSigmaSlider:
            self.edgeSigmaSlider.valueChanged.connect(self._updateSigmaFromSlider)
        else:
            print("DEBUG: edgeSigmaSlider not found during _connectActions!")

        if (
            hasattr(self, "edgeSigmaSpinBox")
            and self.edgeSigmaSpinBox
            and hasattr(self, "edgeSigmaSlider")
            and self.edgeSigmaSlider
        ):
            self.edgeSigmaSpinBox.valueChanged.connect(self._updateSigmaSlider)
        else:
            if not hasattr(self, "edgeSigmaSpinBox") or not self.edgeSigmaSpinBox:
                print(
                    "DEBUG: edgeSigmaSpinBox (for slider update) not found during _connectActions!"
                )
            if not hasattr(self, "edgeSigmaSlider") or not self.edgeSigmaSlider:
                print(
                    "DEBUG: edgeSigmaSlider (for spinbox update) not found during _connectActions!"
                )

        # Connect Binary Threshold controls
        if hasattr(self, "binaryThresholdSlider") and self.binaryThresholdSlider:
            self.binaryThresholdSlider.valueChanged.connect(
                self._updateBinaryThresholdFromSlider
            )
        else:
            print("DEBUG: binaryThresholdSlider not found during _connectActions!")
        if hasattr(self, "binaryThresholdSpinBox") and self.binaryThresholdSpinBox:
            self.binaryThresholdSpinBox.valueChanged.connect(
                self._updateBinaryThresholdSlider
            )
        else:
            print("DEBUG: binaryThresholdSpinBox not found during _connectActions!")

        # --- Conversion Page PushButtons ---
        if hasattr(self, "rgbToGrayButton") and self.rgbToGrayButton:
            self.rgbToGrayButton.clicked.connect(self.image_operations.apply_grayscale)
        if hasattr(self, "rgbToHsvButton") and self.rgbToHsvButton:
            self.rgbToHsvButton.clicked.connect(lambda: self.image_operations.apply_hsv())

        # Segmentation Page PushButtons
        if hasattr(self, "multiOtsuButton") and self.multiOtsuButton:
            self.multiOtsuButton.clicked.connect(lambda: self.image_operations.apply_multi_otsu(
                self.otsuClassesSpinBox.value() if hasattr(self, "otsuClassesSpinBox") else 3
            ))
        if hasattr(self, "chanVeseButton") and self.chanVeseButton:
            self.chanVeseButton.clicked.connect(lambda: self.image_operations.apply_chan_vese(
                max_iter=self.chanveseIterSpinBox.value() if hasattr(self, "chanveseIterSpinBox") else 200,
                tol=self.chanveseTolSpinBox.value() if hasattr(self, "chanveseTolSpinBox") else 0.001
            ))
        if hasattr(self, "morphSnakesButton") and self.morphSnakesButton:
            self.morphSnakesButton.clicked.connect(lambda: self.image_operations.apply_morph_snakes(
                iterations=self.morphIterSpinBox.value() if hasattr(self, "morphIterSpinBox") else 50,
                smoothing=self.morphSmoothSpinBox.value() if hasattr(self, "morphSmoothSpinBox") else 3
            ))

        # Edge Detection Page PushButtons
        if hasattr(self, "robertsButton") and self.robertsButton:
            self.robertsButton.clicked.connect(lambda: self.image_operations.apply_roberts())
        if hasattr(self, "sobelButton") and self.sobelButton:
            self.sobelButton.clicked.connect(lambda: self.image_operations.apply_sobel(
                **self._get_edge_detection_params()
            ))
        if hasattr(self, "scharrButton") and self.scharrButton:
            self.scharrButton.clicked.connect(lambda: self.image_operations.apply_scharr(
                **self._get_edge_detection_params()
            ))
        if hasattr(self, "prewittButton") and self.prewittButton:
            self.prewittButton.clicked.connect(lambda: self.image_operations.apply_prewitt(
                **self._get_edge_detection_params()
            ))
        # --- YENİ EKLENECEK BAĞLANTILAR SONU ---

    def _runOperation(self, operation, is_redo=False):
        """Runs an operation in a separate thread using operation_handler."""
        self.operation_handler.run_operation(operation, is_redo)

    def _applyBinaryThreshold(self):
        """Applies binary thresholding using the controls on the Conversion page."""
        threshold = self.binaryThresholdSpinBox.value()
        invert = self.binaryThresholdInvertCheckbox.isChecked()
        self.image_operations.apply_binary_threshold(threshold, invert)

    def _updateBinaryThresholdFromSlider(self, value):
        """Update binary threshold spinbox from slider value"""
        self.binaryThresholdSpinBox.blockSignals(True)
        self.binaryThresholdSpinBox.setValue(value / 100.0)
        self.binaryThresholdSpinBox.blockSignals(False)

    def _updateBinaryThresholdSlider(self, value):
        """Update binary threshold slider from spinbox value"""
        self.binaryThresholdSlider.blockSignals(True)
        self.binaryThresholdSlider.setValue(int(value * 100))
        self.binaryThresholdSlider.blockSignals(False)

    def _undo(self):
        """Delegates undo operation to EditActionsHandler."""
        if self.edit_handler:
            self.edit_handler.undo()

    def _redo(self):
        """Delegates redo operation to EditActionsHandler."""
        if self.edit_handler:
            self.edit_handler.redo()

    def _showNoImageWarning(self):
        """Shows a warning message when no image is loaded."""
        QMessageBox.warning(
            self, "No Image Loaded", "Please load an image before applying operations."
        )

    def _updateProgress(self, progress):
        """Updates the progress bar in the progress popup."""
        self.progress_popup.updateProgress(progress)

    def _handleOperationComplete(self, result, operation, error, is_redo=False):
        """Handles the completion of an operation using operation_handler."""
        self.operation_handler.handle_operation_complete(result, operation, error, is_redo)

    def _setInitialState(self):
        """Sets the initial state of the UI components."""
        # Call activate with False since no source image is loaded initially
        # Explicitly disable non-open actions/buttons initially
        self.file_handler.update_state(source_loaded=False, output_exists=False)
        self.edit_handler.update_state(source_loaded=False, output_exists=False)
        self._activateUIComponents(False)

    def _activateUIComponents(self, source_loaded=None):
        """Activates or deactivates UI components based on application state."""
        if source_loaded is None:
            source_loaded = self.current_source_image is not None

        output_exists = self.current_output_image is not None
        # Get undo/redo state from handler
        # can_undo = self.edit_handler.can_undo() # Not needed directly here anymore
        # can_redo = self.edit_handler.can_redo() # Not needed directly here anymore

        # File Menu / Toolbar - Let the handler manage menu actions
        self.file_handler.update_state(source_loaded, output_exists)

        # Update File Toolbar buttons based on handler state or direct state
        # Open button is always enabled implicitly
        if self.saveOutputButton:  # Mirrors saveOutputAction logic
            self.saveOutputButton.setEnabled(
                output_exists and bool(self.sourceFilePath)
            )
        if self.saveAsOutputButton:  # Mirrors saveAsOutputAction logic
            self.saveAsOutputButton.setEnabled(output_exists)
        if self.exportSourceButton:  # Mirrors exportSourceAction logic
            self.exportSourceButton.setEnabled(source_loaded)
        if self.exportOutputButton:  # Mirrors exportOutputAction logic
            self.exportOutputButton.setEnabled(output_exists)

        # Edit Menu - Let the handler manage the actions state
        self.edit_handler.update_state(source_loaded, output_exists)

        # Manage Edit Toolbar and Panel button states using handler's info
        if self.clearSourceButton:
            self.clearSourceButton.setEnabled(source_loaded)
        if self.clearOutputButton:
            self.clearOutputButton.setEnabled(output_exists)
        if self.undoButton:
            self.undoButton.setEnabled(self.edit_handler.can_undo())
        if self.redoButton:
            self.redoButton.setEnabled(self.edit_handler.can_redo())

        # Panel Clear Buttons
        if self.clearSourcePanelButton:
            self.clearSourcePanelButton.setEnabled(source_loaded)
        if self.clearOutputPanelButton:
            self.clearOutputPanelButton.setEnabled(output_exists)
            
        # Panel Undo/Redo Buttons
        if hasattr(self, "outputUndoButton"):
            self.outputUndoButton.setEnabled(self.edit_handler.can_undo())
        if hasattr(self, "outputRedoButton"):
            self.outputRedoButton.setEnabled(self.edit_handler.can_redo())
        
        # Source panelinden undo/redo butonları kaldırıldı
        
        # Operations Panel
        self.categoryList.setEnabled(source_loaded)
        self.operationsStack.setEnabled(source_loaded)

        # Enable/disable all operation buttons based on source_loaded
        buttons_to_toggle = [
            self.rgbToGrayButton,
            self.rgbToHsvButton,
            self.applyBinaryThresholdButton,
            self.multiOtsuButton,
            self.chanVeseButton,
            self.morphSnakesButton,
            self.robertsButton,
            self.sobelButton,
            self.scharrButton,
            self.prewittButton,
        ]
        for button in buttons_to_toggle:
            if button:
                button.setEnabled(source_loaded)

        self.conversionMenu.setEnabled(source_loaded)
        self.segmentationMenu.setEnabled(source_loaded)
        self.edgeDetectionMenu.setEnabled(source_loaded)

        # Update status bar or log if needed
        if source_loaded and self.categoryList.currentRow() < 0:
            if self.categoryList.count() > 0:
                self.categoryList.setCurrentRow(
                    0
                )  # Select first category if none selected

    def _edge_params_changed(self):
        # This method seems misplaced now, was likely related to edge processor
        # We might need a general param changed handler if we want live preview
        self.statusBar.showMessage(
            f"Edge Params: Threshold={self.edgeThresholdSpinBox.value():.2f}, Sigma={self.edgeSigmaSpinBox.value():.1f}",
            3000,
        )
        pass  # Keep it simple for now

    def _applyGrayscale(self):
        """Applies grayscale conversion."""
        if self.current_source_image is None:
            self._showNoImageWarning()
            return
        op = GrayscaleOperation()
        self._runOperation(op)

    def _applyHsv(self):
        """Applies RGB to HSV conversion with strong visible effect."""
        if self.current_source_image is None:
            self._showNoImageWarning()
            return

        source_img = self.current_source_image
        if source_img.ndim != 3 or source_img.shape[2] < 3:
            self._logMessage(
                f"WARNING: Source image is not compatible with HSV conversion. Shape: {source_img.shape}, Dtype: {source_img.dtype}",
                "warning",
            )
            QMessageBox.warning(
                self, "Input Error", "HSV conversion requires a 3-channel RGB image."
            )
            return

        self._logMessage(
            f"Applying HSV conversion to source image. Shape: {source_img.shape}, Dtype: {source_img.dtype}"
        )
        self._logMessage(
            f"Channels: {source_img.shape[2] if source_img.ndim == 3 else 1}"
        )

        # Daha güçlü bir efekt için parametreler:
        op = HsvOperation(hue_shift=0.5, saturation_scale=2.0, value_scale=1.5)
        self._runOperation(op)

    def _applyMultiOtsu(self):
        """Creates and runs the Multi-Otsu operation."""
        try:
            op = MultiOtsuOperation(classes=self.otsuClassesSpinBox.value())
            self._runOperation(op)
        except Exception as e:
            error_msg = f"Failed to start Multi-Otsu operation: {e}"
            self._logMessage(error_msg, "error")
            QMessageBox.critical(self, "Operation Error", error_msg)

    def _applyChanVese(self):
        """Creates and runs the Chan-Vese operation."""
        if self.current_source_image is None:
            self._showNoImageWarning()
            return

        try:
            # Get values from UI controls
            max_iter = self.chanveseIterSpinBox.value()
            tol = self.chanveseTolSpinBox.value()

            self._logMessage(
                f"Applying Chan-Vese with max_iter={max_iter}, tol={tol}", "info"
            )

            op = ChanVeseOperation(
                max_iter=max_iter, tol=tol, mu=0.25, lambda1=1.0, lambda2=1.0, dt=0.5
            )

            self._runOperation(op)
        except Exception as e:
            error_msg = f"Failed to start Chan-Vese operation: {e}"
            self._logMessage(error_msg, "error")
            QMessageBox.critical(self, "Operation Error", error_msg)

    def _applyMorphSnakes(self):
        """Creates and runs the Morphological Snakes operation."""
        if self.current_source_image is None:
            self._showNoImageWarning()
            return

        try:
            # Get parameter values from UI controls
            iterations = self.morphIterSpinBox.value()
            smoothing = self.morphSmoothSpinBox.value()

            self._logMessage(
                f"Applying MorphSnakes with iterations={iterations}, smoothing={smoothing}",
                "info",
            )

            op = MorphSnakesOperation(
                iterations=iterations, smoothing=smoothing, lambda1=1.0, lambda2=1.0
            )

            self._runOperation(op)
        except Exception as e:
            error_msg = f"Failed to start Morphological Snakes operation: {e}"
            self._logMessage(error_msg, "error")
            QMessageBox.critical(self, "Operation Error", error_msg)

    def _get_edge_detection_params(self):
        """Helper to get current edge detection parameters from UI."""
        threshold = self.edgeThresholdSpinBox.value()
        sigma = self.edgeSigmaSpinBox.value()
        # Use None for threshold if value is 0.0, indicating auto-threshold
        return self.image_operations.get_edge_detection_params(threshold, sigma)

    def _applyRoberts(self):
        """Applies Roberts edge detection."""
        if self.current_source_image is None:
            self._showNoImageWarning()
            return

        self._logMessage(
            "Attempting Roberts with fixed parameters (auto threshold, no blur).",
            "info",
        )
        # Sabit, en basit parametrelerle dene
        op = RobertsOperation(threshold=None, sigma=0.0)
        # params = self._get_edge_detection_params() # GUI'den parametre almayı geçici olarak devre dışı bırak
        # op = RobertsOperation(threshold=params['threshold'], sigma=params['sigma'])

        self._runOperation(op)

    def _applySobel(self):
        """Applies Sobel edge detection."""
        if self.current_source_image is None:
            self._showNoImageWarning()
            return
        params = self._get_edge_detection_params()
        op = SobelOperation(threshold=params["threshold"], sigma=params["sigma"])
        self._runOperation(op)

    def _applyScharr(self):
        """Applies Scharr edge detection."""
        if self.current_source_image is None:
            self._showNoImageWarning()
            return
        params = self._get_edge_detection_params()
        op = ScharrOperation(threshold=params["threshold"], sigma=params["sigma"])
        self._runOperation(op)

    def _applyPrewitt(self):
        """Applies Prewitt edge detection."""
        if self.current_source_image is None:
            self._showNoImageWarning()
            return
        params = self._get_edge_detection_params()
        op = PrewittOperation(threshold=params["threshold"], sigma=params["sigma"])
        self._runOperation(op)

    # --- Edge Detection Parameter Sync Methods ---

    def _updateThresholdFromSlider(self, value):
        """Update threshold spinbox from slider value (0-100 -> 0.0-1.0)."""
        self.edgeThresholdSpinBox.blockSignals(True)
        self.edgeThresholdSpinBox.setValue(value / 100.0)
        self.edgeThresholdSpinBox.blockSignals(False)
        self._edge_params_changed()  # Update status bar or trigger preview

    def _updateThresholdSlider(self, value):
        """Update threshold slider from spinbox value (0.0-1.0 -> 0-100)."""
        self.edgeThresholdSlider.blockSignals(True)
        self.edgeThresholdSlider.setValue(int(value * 100))
        self.edgeThresholdSlider.blockSignals(False)
        # No need to call _edge_params_changed here, spinbox triggers it

    def _updateSigmaFromSlider(self, value):
        """Update sigma spinbox from slider value (0-50 -> 0.0-5.0)."""
        self.edgeSigmaSpinBox.blockSignals(True)
        self.edgeSigmaSpinBox.setValue(value / 10.0)
        self.edgeSigmaSpinBox.blockSignals(False)
        self._edge_params_changed()  # Update status bar or trigger preview

    def _updateSigmaSlider(self, value):
        """Update sigma slider from spinbox value (0.0-5.0 -> 0-50)."""
        self.edgeSigmaSlider.blockSignals(True)
        self.edgeSigmaSlider.setValue(int(value * 10))
        self.edgeSigmaSlider.blockSignals(False)
        # No need to call _edge_params_changed here, spinbox triggers it

    def initUI(self):
        self.setWindowTitle("Parameter Adjustment")

    # --- Image Display ---

    def _updateImageDisplay(self, label: QLabel, image_data: Union[np.ndarray, None]):
        """Updates a QLabel with a NumPy image array."""
        if image_data is None:
            # Determine placeholder text based on label
            placeholder = "No Image Data"
            if label == self.sourcePixmapLabel:
                placeholder = "Drop or Open Image/Video"
            elif label == self.outputPixmapLabel:
                placeholder = "Processing Result Area"
            label.setText(placeholder)
            label.setPixmap(QPixmap())  # Need QPixmap import
            return

        try:
            img_display = image_data
            # Ensure image is uint8 for QPixmap
            if img_display.dtype != np.uint8:
                # Check if it's a float image in [0, 1]
                if (
                    np.issubdtype(img_display.dtype, np.floating)
                    and img_display.max() <= 1.0
                    and img_display.min() >= 0.0
                ):
                    img_display = (img_display * 255).astype(np.uint8)
                else:
                    # Attempt conversion for other types (like bool, int, or float outside [0,1])
                    # img_as_ubyte handles scaling for floats outside [0,1] and converts other types
                    self._logMessage(
                        f"Converting image from {img_display.dtype} to uint8 for display.",
                        "info",
                    )
                    img_display = img_as_ubyte(img_display)

            # If still not uint8 after conversion attempts, log error
            if img_display.dtype != np.uint8:
                raise TypeError(
                    f"Could not convert image to uint8 for display. Current dtype: {img_display.dtype}"
                )

            height, width = img_display.shape[:2]
            bytes_per_line = width
            format = QImage.Format_Grayscale8  # Default format

            if img_display.ndim == 3:
                channels = img_display.shape[2]
                if channels == 4:
                    format = QImage.Format_RGBA8888
                    bytes_per_line = 4 * width
                elif channels == 3:
                    format = QImage.Format_RGB888
                    bytes_per_line = 3 * width
                else:
                    # Fallback: try displaying the first channel as grayscale for unsupported channel counts
                    self._logMessage(
                        f"Displaying first channel of {channels}-channel image as grayscale.",
                        "warning",
                    )
                    img_display = img_display[:, :, 0]
                    format = QImage.Format_Grayscale8
                    bytes_per_line = width  # Recalculate bytes per line
                    # Ensure the single channel is contiguous in memory
                    if not img_display.flags["C_CONTIGUOUS"]:
                        img_display = np.ascontiguousarray(img_display)

            elif img_display.ndim != 2:
                self._logMessage(
                    f"Cannot display image with {img_display.ndim} dimensions.", "error"
                )
                label.setText("Display Error")
                label.setPixmap(QPixmap())
                return
            else:  # ndim == 2, ensure it's contiguous
                format = QImage.Format_Grayscale8
                bytes_per_line = width
                if not img_display.flags["C_CONTIGUOUS"]:
                    img_display = np.ascontiguousarray(img_display)

            # Create QImage directly from numpy array data
            qimage = QImage(img_display.data, width, height, bytes_per_line, format)

            # QImage from data might share memory; create a copy for safety
            qimage_copy = qimage.copy()

            # Important: For RGB images, check if byte order needs swapping (BGR vs RGB)
            # QImage.Format_RGB888 assumes RGB order. If your numpy array is BGR (e.g., from OpenCV *without conversion*),
            # you might need qimage_copy = qimage_copy.rgbSwapped()
            # Since we mostly use skimage which prefers RGB, swapping is usually NOT needed here.

            pixmap = QPixmap.fromImage(qimage_copy)

            # Scale pixmap to fit label while keeping aspect ratio
            # Use label.size() available at the moment of update
            scaled_pixmap = pixmap.scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            label.setText("")  # Clear placeholder text

        except Exception as e:
            import traceback

            self._logMessage(
                f"Error displaying image: {e}\n{traceback.format_exc()}", "error"
            )
            label.setText("Display Error")
            label.setPixmap(QPixmap())

    def resizeEvent(self, event):
        """Handle window resize to re-scale images."""
        super().resizeEvent(event)
        # Rescale images displayed in labels when window size changes
        # Check if labels exist before trying to update
        if self.sourcePixmapLabel and self.current_source_image is not None:
            self.operation_handler.update_image_display(self.sourcePixmapLabel, self.current_source_image)
        if self.outputPixmapLabel and self.current_output_image is not None:
            self.operation_handler.update_image_display(self.outputPixmapLabel, self.current_output_image)
