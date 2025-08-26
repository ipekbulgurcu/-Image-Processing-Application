"""
Edge Detection Operations Module
------------------------------
This module implements various edge detection operations, all inheriting from
the BaseEdgeDetectionOperation class.

Inheritance Hierarchy:
AbstractOperation
└── BaseEdgeDetectionOperation
    ├── RobertsOperation
    ├── SobelOperation
    ├── ScharrOperation
    └── PrewittOperation
"""

from abc import ABC, abstractmethod

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage import filters, img_as_ubyte

from dialog_base import BaseDialog  # Import base dialog class
from operations_base import (
    AbstractOperation,
    BaseEdgeDetectionOperation,
    ProgressCallback,
)

# =============================================================================
# EDGE DETECTION OPERATIONS - Inherit from BaseEdgeDetectionOperation 
# =============================================================================

class RobertsOperation(BaseEdgeDetectionOperation):
    """
    Applies Roberts edge detection filter.
    
    The Roberts operator performs a simple, quick to compute, 2-D spatial gradient 
    measurement on an image. It highlights regions of high spatial frequency which 
    often correspond to edges.
    """
    
    def get_operation_name(self) -> str:
        return "Roberts Edge Detection"

    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        # Apply the Roberts filter directly
        return filters.roberts(image)


class SobelOperation(BaseEdgeDetectionOperation):
    """
    Applies Sobel edge detection filter.
    
    The Sobel filter emphasizes edges by computing an approximation of the gradient
    of the image intensity function.
    """
    
    def get_operation_name(self) -> str:
        return "Sobel Edge Detection"

    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        # Apply the Sobel filter directly
        return filters.sobel(image)


class ScharrOperation(BaseEdgeDetectionOperation):
    """
    Applies Scharr edge detection filter.
    
    The Scharr filter is similar to Sobel but uses different kernel coefficients
    that make it more accurate (less rotational asymmetry).
    """
    
    def get_operation_name(self) -> str:
        return "Scharr Edge Detection"

    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        # Apply the Scharr filter directly
        return filters.scharr(image)


class PrewittOperation(BaseEdgeDetectionOperation):
    """
    Applies Prewitt edge detection filter.
    
    The Prewitt filter emphasizes edges by using a discrete differentiation operator,
    computing an approximation of the gradient of the image intensity function.
    """
    
    def get_operation_name(self) -> str:
        return "Prewitt Edge Detection"

    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        # Apply the Prewitt filter directly
        return filters.prewitt(image)


# =============================================================================
# DIALOG CLASSES
# =============================================================================

class EdgeDetectionDialog(BaseDialog):
    """
    Dialog for setting parameters for edge detection operations.
    Inherits from BaseDialog.
    """
    
    def __init__(self, parent=None, operation_name=""):
        # Pass the title to the base class constructor
        super().__init__(parent, window_title=f"{operation_name} Parameters")
        self.operation_name = operation_name
        # Base class calls _add_parameter_widgets(), _add_buttons(), _position_dialog()

    def _add_parameter_widgets(self):
        # Add widgets specific to this dialog to self.param_layout (provided by base class)
        if "Canny" in self.operation_name:
            # Canny-specific parameters
            self.sigma_spin = self._create_float_parameter(
                self.param_layout, "Sigma:", 0.1, 10.0, 1.0, 0.1
            )
            self.low_thresh_spin = self._create_float_parameter(
                self.param_layout, "Low Threshold:", 0.0, 1.0, 0.1, 0.01
            )
            self.high_thresh_spin = self._create_float_parameter(
                self.param_layout, "High Threshold:", 0.0, 1.0, 0.2, 0.01
            )

            # Info labels - using helper from base class
            self._add_info_label(
                self.param_layout, "Sigma: Standard deviation of the Gaussian filter."
            )
            self._add_info_label(
                self.param_layout, "Low Threshold: Lower threshold for hysteresis."
            )
            self._add_info_label(
                self.param_layout, "High Threshold: Upper threshold for hysteresis."
            )
        else:
            # Common parameters for Roberts, Sobel, Scharr, Prewitt
            self.threshold_spin = self._create_float_parameter(
                self.param_layout, "Threshold:", 0.0, 1.0, 0.1, 0.01
            )
            self.sigma_spin = self._create_float_parameter(
                self.param_layout, "Sigma (Blur):", 0.0, 5.0, 0.0, 0.1
            )

            # Info labels - using helper from base class
            self._add_info_label(
                self.param_layout, "Threshold: Gradient threshold (0.0 = auto)."
            )
            self._add_info_label(
                self.param_layout,
                "Sigma: Gaussian blur applied before detection (0.0 = no blur).",
            )

    # _create_int_parameter and _create_float_parameter are now inherited from BaseDialog
    # _add_buttons is handled by BaseDialog
    # _position_dialog (via showEvent or init) is handled by BaseDialog

    def get_parameters(self):
        # Implement the abstract method from BaseDialog
        params = {}
        # Safely access widgets, as they might not exist depending on operation_name
        if hasattr(self, "sigma_spin"):
            params["sigma"] = self.sigma_spin.value()
        if hasattr(self, "low_thresh_spin"):
            params["low_threshold"] = self.low_thresh_spin.value()
        if hasattr(self, "high_thresh_spin"):
            params["high_threshold"] = self.high_thresh_spin.value()
        if hasattr(self, "threshold_spin"):
            params["threshold"] = self.threshold_spin.value()

        return params

    # showEvent logic is now in BaseDialog._position_dialog
