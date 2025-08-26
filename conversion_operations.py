"""
Conversion Operations Module
---------------------------
This module implements various image conversion operations, all inheriting from
the BaseConversionOperation class.

Inheritance Hierarchy:
AbstractOperation
└── BaseConversionOperation
    ├── GrayscaleOperation
    ├── HsvOperation 
    ├── BinaryThresholdOperation
    └── AdaptiveThresholdOperation
"""

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
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
from skimage import color, util

from dialog_base import BaseDialog
from operations_base import AbstractOperation, BaseConversionOperation, ProgressCallback

# =============================================================================
# CONVERSION OPERATIONS - Inherit from BaseConversionOperation
# =============================================================================

class GrayscaleOperation(BaseConversionOperation):
    """Converts an RGB image to grayscale with adjustable brightness and contrast."""

    def __init__(self, brightness=0.0, contrast=1.0):
        """
        Initialize with brightness and contrast parameters.

        Parameters:
        -----------
        brightness : float
            Value between -1.0 and 1.0 controlling brightness adjustment
            0.0 = no change, positive values brighten, negative values darken
        contrast : float
            Value between 0.0 and 3.0 controlling contrast adjustment
            1.0 = no change, values > 1.0 increase contrast, values < 1.0 decrease contrast
        """
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast

    def _validate_input(self, image_data: np.ndarray):
        super()._validate_input(image_data)
        if image_data.ndim != 3 or image_data.shape[2] < 3:
            raise ValueError("Input must be an RGB image (3 channels).")

    def _apply_impl(
        self, image_data: np.ndarray, progress_callback: ProgressCallback
    ) -> np.ndarray:
        """Applies grayscale conversion with brightness and contrast adjustments."""
        self._report_progress(progress_callback, 30, "Converting to grayscale...")
        grayscale_image = color.rgb2gray(image_data)

        # Apply adjustments if parameters are not at default values
        if self.brightness != 0.0 or self.contrast != 1.0:
            self._report_progress(
                progress_callback, 60, "Applying brightness and contrast..."
            )
            grayscale_float = grayscale_image.astype(np.float32)
            if self.brightness != 0.0:
                grayscale_float = grayscale_float + self.brightness
                grayscale_float = np.clip(grayscale_float, 0, 1.0)
            if self.contrast != 1.0:
                mean_value = 0.5
                grayscale_float = (
                    grayscale_float - mean_value
                ) * self.contrast + mean_value
                grayscale_float = np.clip(grayscale_float, 0, 1.0)
            grayscale_image = grayscale_float

        self._report_progress(progress_callback, 80, "Formatting output...")
        output_image = util.img_as_ubyte(grayscale_image)
        return output_image

    def get_operation_name(self) -> str:
        return "RGB to Grayscale"


class HsvOperation(BaseConversionOperation):
    """Converts an RGB image to HSV colorspace with adjustable Hue, Saturation, and Value."""

    def __init__(self, hue_shift=0.0, saturation_scale=1.0, value_scale=1.0):
        """
        Initialize with HSV adjustment parameters.

        Parameters:
        -----------
        hue_shift : float
            Value between -0.5 and 0.5 controlling hue shift (circular)
            0.0 = no change
        saturation_scale : float
            Value between 0.0 and 2.0 controlling saturation scaling
            1.0 = no change, higher values increase saturation
        value_scale : float
            Value between 0.0 and 2.0 controlling value (brightness) scaling
            1.0 = no change, higher values increase brightness
        """
        super().__init__()
        self.hue_shift = hue_shift
        self.saturation_scale = saturation_scale
        self.value_scale = value_scale

    def _validate_input(self, image_data: np.ndarray):
        super()._validate_input(image_data)
        if image_data.ndim != 3 or image_data.shape[2] < 3:
            raise ValueError("HSV conversion requires an RGB image (3 channels).")

    def _apply_impl(
        self, image_data: np.ndarray, progress_callback: ProgressCallback
    ) -> np.ndarray:
        self._report_progress(progress_callback, 30, "Converting to HSV...")
        image_float = util.img_as_float(image_data)

        if progress_callback:
            progress_callback(
                35,
                f"Debug - Image shape: {image_float.shape}, dtype: {image_float.dtype}, min: {image_float.min()}, max: {image_float.max()}",
            )

        hsv_image_float = color.rgb2hsv(image_float)

        if progress_callback:
            progress_callback(
                45,
                f"Debug - HSV conversion success. Shape: {hsv_image_float.shape}, min: {hsv_image_float.min()}, max: {hsv_image_float.max()}",
            )

        if (
            self.hue_shift != 0.0
            or self.saturation_scale != 1.0
            or self.value_scale != 1.0
        ):
            self._report_progress(progress_callback, 50, "Applying HSV adjustments...")
            if self.hue_shift != 0.0:
                hsv_image_float[:, :, 0] = (
                    hsv_image_float[:, :, 0] + self.hue_shift
                ) % 1.0
            if self.saturation_scale != 1.0:
                hsv_image_float[:, :, 1] = np.clip(
                    hsv_image_float[:, :, 1] * self.saturation_scale, 0, 1.0
                )
            if self.value_scale != 1.0:
                hsv_image_float[:, :, 2] = np.clip(
                    hsv_image_float[:, :, 2] * self.value_scale, 0, 1.0
                )

        self._report_progress(
            progress_callback, 70, "Converting back to RGB for display..."
        )
        output_image_float = color.hsv2rgb(hsv_image_float)
        output_image_float = np.clip(output_image_float, 0, 1.0)
        output_image = util.img_as_ubyte(output_image_float)
        return output_image

    def get_operation_name(self) -> str:
        return "RGB to HSV"


class BinaryThresholdOperation(BaseConversionOperation):
    """Converts an image to binary (black and white) using thresholding."""

    def __init__(self, threshold=0.5, invert=False):
        """
        Initialize with threshold value.

        Parameters:
        -----------
        threshold : float
            Value between 0.0 and 1.0 determining threshold level
        invert : bool
            If True, inverts the result (white becomes black and vice versa)
        """
        super().__init__()
        self.threshold = threshold
        self.invert = invert

    def _apply_impl(
        self, image_data: np.ndarray, progress_callback: ProgressCallback
    ) -> np.ndarray:
        self._report_progress(progress_callback, 30, "Preparing grayscale image...")
        gray_image = self._prepare_grayscale(image_data, progress_callback)

        self._report_progress(
            progress_callback, 60, f"Applying threshold at {self.threshold}..."
        )
        binary_image = gray_image > self.threshold

        if self.invert:
            self._report_progress(progress_callback, 80, "Inverting image...")
            binary_image = ~binary_image

        output_image = util.img_as_ubyte(binary_image)
        return output_image

    def get_operation_name(self) -> str:
        return "Binary Threshold"


class AdaptiveThresholdOperation(BaseConversionOperation):
    """Applies adaptive thresholding to an image."""

    def __init__(self, block_size=35, constant=0.0):
        """
        Initialize with block size and constant.

        Parameters:
        -----------
        block_size : int
            Size of local neighborhood for adaptive thresholding (must be odd)
        constant : float
            Constant subtracted from weighted mean (-0.2 to 0.2 typical range)
        """
        super().__init__()
        # Ensure block_size is odd
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.constant = constant

    def _apply_impl(
        self, image_data: np.ndarray, progress_callback: ProgressCallback
    ) -> np.ndarray:
        self._report_progress(progress_callback, 20, "Preparing grayscale image...")
        gray_image = self._prepare_grayscale(image_data, progress_callback)

        self._report_progress(progress_callback, 40, "Computing adaptive threshold...")
        from skimage.filters import threshold_local

        adaptive_thresh = threshold_local(
            gray_image, self.block_size, offset=self.constant
        )

        self._report_progress(progress_callback, 80, "Applying threshold...")
        binary_image = gray_image > adaptive_thresh

        output_image = util.img_as_ubyte(binary_image)
        return output_image

    def get_operation_name(self) -> str:
        return "Adaptive Threshold"


# =============================================================================
# DIALOG CLASSES
# =============================================================================

class ConversionDialog(BaseDialog):
    """
    Dialog for setting parameters for conversion operations.
    Inherits from BaseDialog.
    """
    
    def __init__(self, parent=None, operation_name=""):
        super().__init__(parent, window_title=f"{operation_name} Parameters")
        self.operation_name = operation_name
        # BaseDialog._add_parameter_widgets() is called by super().__init__
        # BaseDialog._add_buttons() is called by super().__init__
        # BaseDialog._position_dialog() is called by super().__init__

    def _add_parameter_widgets(self):
        if "Grayscale" in self.operation_name:
            self.brightness_spin = self._create_float_parameter(
                self.param_layout, "Brightness:", -1.0, 1.0, 0.0, 0.05
            )
            self.contrast_spin = self._create_float_parameter(
                self.param_layout, "Contrast:", 0.0, 3.0, 1.0, 0.05
            )
            self._add_info_label(
                self.param_layout, "Brightness: -1.0 (darker) to 1.0 (brighter)."
            )
            self._add_info_label(
                self.param_layout, "Contrast: 0.0 (none) to 3.0 (high)."
            )

        elif "HSV" in self.operation_name:
            self.hue_shift_spin = self._create_float_parameter(
                self.param_layout, "Hue Shift:", -0.5, 0.5, 0.0, 0.01
            )
            self.saturation_scale_spin = self._create_float_parameter(
                self.param_layout, "Saturation Scale:", 0.0, 2.0, 1.0, 0.05
            )
            self.value_scale_spin = self._create_float_parameter(
                self.param_layout, "Value Scale:", 0.0, 2.0, 1.0, 0.05
            )
            self._add_info_label(
                self.param_layout, "Hue Shift: -0.5 to 0.5 (circular)."
            )
            self._add_info_label(
                self.param_layout,
                "Saturation Scale: 0.0 (desaturated) to 2.0 (oversaturated).",
            )
            self._add_info_label(
                self.param_layout, "Value Scale: 0.0 (black) to 2.0 (brighter)."
            )

        elif "Binary Threshold" in self.operation_name:
            self.threshold_spin = self._create_float_parameter(
                self.param_layout, "Threshold:", 0.0, 1.0, 0.5, 0.01
            )

            # Invert Checkbox
            self.invert_checkbox = QCheckBox("Invert Output")
            self.invert_checkbox.setChecked(False)
            self.invert_checkbox.stateChanged.connect(self.parameter_changed.emit)
            h_layout = QHBoxLayout()
            h_layout.addStretch()
            h_layout.addWidget(self.invert_checkbox)
            h_layout.addStretch()
            self.param_layout.addLayout(h_layout)
            self._add_info_label(
                self.param_layout, "Threshold: 0.0 (black) to 1.0 (white)."
            )
            self._add_info_label(
                self.param_layout, "Invert: Swaps black and white in the output."
            )

        elif "Adaptive Threshold" in self.operation_name:
            # Block size must be odd
            self.block_size_spin = self._create_int_parameter(
                self.param_layout, "Block Size (odd):", 3, 255, 35
            )
            # Ensure it's odd, or connect to a validator if BaseDialog supports it
            # For now, we'll rely on the operation to adjust if it's even.
            # Or, we can adjust it here:
            # self.block_size_spin.valueChanged.connect(self._ensure_odd_block_size)

            self.constant_c_spin = self._create_float_parameter(
                self.param_layout, "Constant (C):", -0.5, 0.5, 0.0, 0.01
            )  # Adjusted range for C
            self._add_info_label(
                self.param_layout,
                "Block Size: Size of the pixel neighborhood (must be odd).",
            )
            self._add_info_label(
                self.param_layout,
                "Constant (C): Subtracted from the mean or weighted mean.",
            )
        else:
            # Default message if operation_name is not recognized
            self._add_info_label(
                self.param_layout,
                f"No parameters configurable for {self.operation_name}.",
            )

    # Optional: Helper to ensure block_size is odd for Adaptive Threshold
    # def _ensure_odd_block_size(self, value):
    #     if hasattr(self, 'block_size_spin'):
    #         if value % 2 == 0:
    #             # Find the closest odd number. If value is 4, go to 3 or 5.
    #             # Going up is generally safer for block sizes.
    #             self.block_size_spin.setValue(value + 1 if value + 1 <= self.block_size_spin.maximum() else value -1)

    def get_parameters(self) -> dict:
        params = {}
        if "Grayscale" in self.operation_name:
            if hasattr(self, "brightness_spin"):
                params["brightness"] = self.brightness_spin.value()
            if hasattr(self, "contrast_spin"):
                params["contrast"] = self.contrast_spin.value()
        elif "HSV" in self.operation_name:
            if hasattr(self, "hue_shift_spin"):
                params["hue_shift"] = self.hue_shift_spin.value()
            if hasattr(self, "saturation_scale_spin"):
                params["saturation_scale"] = self.saturation_scale_spin.value()
            if hasattr(self, "value_scale_spin"):
                params["value_scale"] = self.value_scale_spin.value()
        elif "Binary Threshold" in self.operation_name:
            if hasattr(self, "threshold_spin"):
                params["threshold"] = self.threshold_spin.value()
            if hasattr(self, "invert_checkbox"):
                params["invert"] = self.invert_checkbox.isChecked()
        elif "Adaptive Threshold" in self.operation_name:
            if hasattr(self, "block_size_spin"):
                bs_val = self.block_size_spin.value()
                params["block_size"] = (
                    bs_val if bs_val % 2 == 1 else bs_val + 1
                )  # Ensure odd
            if hasattr(self, "constant_c_spin"):
                params["constant"] = self.constant_c_spin.value()
        return params
