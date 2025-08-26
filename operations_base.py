"""
Operations Base Module
---------------------
This module defines the base classes for the inheritance hierarchy of image operations.
The structure follows:

AbstractOperation (ABC)
  ├── BaseConversionOperation
  ├── BaseEdgeDetectionOperation
  └── BaseSegmentationOperation

Each concrete operation class inherits from one of the three base operation classes.
"""

import traceback  # Add traceback for error handling in apply
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import numpy as np
from skimage import color

# Type hint for progress callback
ProgressCallback = Optional[Callable[[int, str], None]]


class AbstractOperation(ABC):
    """
    Abstract base class for all image processing operations.
    This is the top-level class in the operation inheritance hierarchy.
    
    All operation classes must inherit from this or one of its subclasses.
    """

    def __init__(self):
        """Initializes the operation."""
        self._original_image_data = None

    @abstractmethod
    def _apply_impl(
        self, image_data: np.ndarray, progress_callback: ProgressCallback
    ) -> np.ndarray:
        """
        Core implementation of the operation by subclasses.
        This method assumes input validation and undo storage have been handled.

        Args:
            image_data: The potentially pre-processed input image as a NumPy array.
            progress_callback: Function to report progress.

        Returns:
            The processed image as a NumPy array.
        """
        pass

    def apply(
        self, image_data: np.ndarray, progress_callback: ProgressCallback = None
    ) -> np.ndarray:
        """
        Applies the operation with error handling, input validation, and undo storage.
        Calls the subclass's _apply_impl for the core logic.
        """
        op_name = self.get_operation_name()  # Get name once
        try:
            self._report_progress(progress_callback, 0, f"Starting {op_name}...")

            # Store original for undo FIRST
            self._store_for_undo(image_data)

            # Validate input (can be overridden by subclasses for more specific checks)
            self._validate_input(image_data)
            self._report_progress(
                progress_callback, 10, "Input validated."
            )  # Generic validation step

            # --- Core Operation ---
            # Subclass performs its specific logic, including any necessary preprocessing
            result = self._apply_impl(image_data, progress_callback)
            # ---------------------

            self._report_progress(progress_callback, 100, f"{op_name} complete.")
            return result

        except Exception as e:
            error_msg = f"Error in {op_name}: {e}"
            print(error_msg)
            print(traceback.format_exc())
            self._report_progress(progress_callback, 100, f"Error: {str(e)}")
            raise  # Re-raise the original exception

    def undo(self) -> Union[np.ndarray, None]:
        """Returns the original image data stored before apply."""
        return self._original_image_data

    def _store_for_undo(self, image_data: np.ndarray):
        """Stores a copy of the image data for undo operation."""
        # Ensure we have valid data before copying
        if image_data is not None and isinstance(image_data, np.ndarray):
            self._original_image_data = image_data.copy()
        else:
            # Log a warning or handle as appropriate if image_data is invalid here
            print("Warning: Attempted to store invalid data for undo.")
            self._original_image_data = None

    def _report_progress(
        self, callback: ProgressCallback, percentage: int, message: str
    ):
        """Safely calls the progress callback if it exists."""
        if callback:
            # Clamp percentage between 0 and 100
            percentage = max(0, min(percentage, 100))
            try:
                callback(percentage, message)
            except Exception as e:
                print(f"Error in progress callback: {e}")

    def _validate_input(self, image_data: np.ndarray):
        """
        Basic input validation. Subclasses can override this for more specific checks
        *after* calling super()._validate_input(image_data).
        """
        if not isinstance(image_data, np.ndarray):
            raise TypeError("Input image_data must be a NumPy array.")
        if image_data.size == 0:
            raise ValueError("Input image is empty.")
        if image_data.ndim < 2 or image_data.ndim > 3:
            # Allow only 2D (grayscale) or 3D (color) arrays for common operations
            raise ValueError(
                f"Input image must be 2D or 3D, but got {image_data.ndim} dimensions."
            )
        # Add more checks if needed, e.g., check for specific dtypes if required by most operations

    def _prepare_grayscale(
        self, image_data: np.ndarray, progress_callback: ProgressCallback = None
    ) -> np.ndarray:
        """
        Converts image to grayscale if needed and normalizes to float [0, 1].
        (Kept here as a utility, but not called directly by the base apply method).
        """
        prep_image = image_data  # Start with original
        try:
            # Handle RGBA images - remove alpha channel
            if prep_image.ndim == 3 and prep_image.shape[2] == 4:
                self._report_progress(
                    progress_callback, 15, "Converting RGBA to RGB..."
                )  # Adjust %
                prep_image = prep_image[:, :, :3]

            # Convert RGB to grayscale or use existing grayscale
            if prep_image.ndim == 3 and prep_image.shape[2] == 3:
                self._report_progress(
                    progress_callback, 20, "Converting RGB to grayscale..."
                )  # Adjust %
                prep_image = color.rgb2gray(prep_image)
            elif prep_image.ndim == 2:
                self._report_progress(
                    progress_callback, 20, "Image already grayscale..."
                )  # Adjust %
                pass  # Already grayscale
            else:  # Handle unexpected dimensions after RGBA->RGB conversion
                raise ValueError(
                    f"Cannot prepare grayscale for image with shape {prep_image.shape}"
                )

            # Normalize to float in range [0, 1] if not already float
            if not np.issubdtype(prep_image.dtype, np.floating):
                self._report_progress(
                    progress_callback, 25, "Converting to float [0, 1]..."
                )  # Adjust %
                max_val = 255.0 if prep_image.dtype == np.uint8 else np.max(prep_image)
                if max_val > 0:
                    prep_image = prep_image.astype(np.float64) / max_val
                else:
                    prep_image = prep_image.astype(np.float64)  # Avoid division by zero
            # Ensure float image is clipped to [0, 1]
            elif np.min(prep_image) < 0.0 or np.max(prep_image) > 1.0:
                self._report_progress(
                    progress_callback, 25, "Clipping float image to [0, 1]..."
                )  # Adjust %
                prep_image = np.clip(prep_image, 0.0, 1.0)

            return prep_image

        except Exception as e:
            print(f"Error in _prepare_grayscale: {e}")
            self._report_progress(
                progress_callback, 30, f"Error preparing grayscale image: {str(e)}"
            )
            raise

    def get_operation_name(self) -> str:
        """Returns the name of the operation (defaulting to class name)."""
        return self.__class__.__name__.replace(
            "Operation", ""
        )  # Provide a cleaner default name

    def uses_original_source(self) -> bool:
        """Indicates if the operation should always use the original source image.
        Defaults to False, meaning it uses the previous operation's output.
        Subclasses that need the original source should override this to return True.
        """
        return False


# ==============================================================================
# LEVEL 2 INHERITANCE: Specialized Base Operation Types
# ==============================================================================

class BaseEdgeDetectionOperation(AbstractOperation):
    """
    Abstract base class for edge detection operations.
    
    All edge detection operations should inherit from this class.
    Provides common features for edge detection:
    - Threshold parameter for edge detection
    - Gaussian blur via sigma parameter
    - Common preprocessing for grayscale conversion
    """

    def __init__(self, threshold: float = 0.1, sigma: float = 0.0):
        """
        Initializes the edge detection operation.

        Args:
            threshold: Threshold value for edge detection. If <= 0 (or None), auto-thresholding might be used by the filter.
            sigma: Standard deviation for Gaussian blur applied before edge detection. If 0, no blur.
        """
        super().__init__()
        # Input validation
        if threshold is not None and not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a number or None")
        if not isinstance(sigma, (int, float)) or sigma < 0:
            raise ValueError("sigma must be a non-negative number")

        self.threshold = (
            float(threshold) if threshold is not None else None
        )  # Store None if threshold is None or 0
        self.sigma = float(sigma)

    @abstractmethod
    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        """Applies the specific edge detection filter (e.g., Sobel, Roberts)."""
        pass

    def _apply_impl(
        self, image_data: np.ndarray, progress_callback: ProgressCallback
    ) -> np.ndarray:
        """Applies preprocessing, Gaussian blur, and the specific edge filter."""
        # Base apply handles try/except, progress start/end, undo, validate

        # Prepare image (grayscale, float [0, 1]) using the base utility
        self._report_progress(progress_callback, 20, "Preparing grayscale image...")
        prepared_image = self._prepare_grayscale(image_data, progress_callback)

        # Apply Gaussian blur if sigma is greater than 0
        if self.sigma > 0:
            self._report_progress(
                progress_callback, 40, f"Applying Gaussian blur (sigma={self.sigma})..."
            )
            from skimage import filters

            prepared_image = filters.gaussian(
                prepared_image, sigma=self.sigma, channel_axis=None
            )

        op_name = self.get_operation_name()  # Get name for logging within impl
        self._report_progress(progress_callback, 60, f"Applying {op_name} filter...")

        # Apply the specific filter implemented by the subclass
        edge_image = self._apply_filter(prepared_image)

        # Apply thresholding only if a threshold is provided AND filter output is not already boolean
        if self.threshold is not None and edge_image.dtype != bool:
            self._report_progress(
                progress_callback, 80, f"Applying threshold ({self.threshold})..."
            )
            edge_image = edge_image > self.threshold

        # Convert to uint8 for display
        self._report_progress(progress_callback, 90, "Formatting output...")
        from skimage import util

        output = util.img_as_ubyte(edge_image)

        return output


class BaseSegmentationOperation(AbstractOperation):
    """
    Abstract base class for segmentation operations.
    
    All segmentation operations should inherit from this class.
    These operations typically work on the original source image
    and often involve grayscale preparation.
    """

    def uses_original_source(self) -> bool:
        """
        Segmentation operations generally use the original source image.
        """
        return True

    # _prepare_grayscale is already available in AbstractOperation
    # Subclasses will call it as needed within their _apply_impl


class BaseConversionOperation(AbstractOperation):
    """
    Abstract base class for conversion operations.
    
    All conversion operations should inherit from this class.
    Behavior regarding source image (original vs. previous output)
    and grayscale preparation can vary among conversion types.
    
    Subclasses should override 'uses_original_source' and call
    '_prepare_grayscale' as needed within their '_apply_impl'.
    """

    def uses_original_source(self) -> bool:
        """
        Conversion operations generally use the original source image by default.
        """
        return True
