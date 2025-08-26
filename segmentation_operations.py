"""
Segmentation Operations Module
----------------------------
This module implements various image segmentation operations, all inheriting from
the BaseSegmentationOperation class.

Inheritance Hierarchy:
AbstractOperation
└── BaseSegmentationOperation
    ├── MultiOtsuOperation
    ├── ChanVeseOperation
    └── MorphSnakesOperation
"""

#!/usr/bin/env python3
import os  # For saving debug images
import time  # Zaman hesaplaması için

import matplotlib.pyplot as plt  # For saving debug images without GUI toolkit issues
import numpy as np
from skimage import color, filters, measure, segmentation, util

from operations_base import (
    AbstractOperation,
    BaseSegmentationOperation,
    ProgressCallback,
)


# =============================================================================
# SEGMENTATION OPERATIONS - Inherit from BaseSegmentationOperation
# =============================================================================

class MultiOtsuOperation(BaseSegmentationOperation):
    """
    Applies Multi-Otsu thresholding segmentation.
    
    Multi-Otsu finds optimal thresholds to separate an image into multiple
    classes by maximizing the variance between classes.
    """
    
    def __init__(self, classes: int = 3):
        super().__init__()
        self.classes = max(2, min(classes, 5))  # Ensure classes are between 2 and 5

    def get_operation_name(self) -> str:
        return "Multi-Otsu Segmentation"

    def _apply_impl(
        self, image_data: np.ndarray, progress_callback: ProgressCallback
    ) -> np.ndarray:
        self._report_progress(progress_callback, 20, "Preparing grayscale image...")
        prepared_image = self._prepare_grayscale(image_data, progress_callback)

        self._report_progress(progress_callback, 40, "Computing Otsu thresholds...")
        thresholds = filters.threshold_multiotsu(prepared_image, classes=self.classes)

        self._report_progress(progress_callback, 60, "Applying segmentation...")
        regions = np.digitize(prepared_image, bins=thresholds)

        self._report_progress(progress_callback, 80, "Formatting output...")
        output = util.img_as_ubyte(regions / (self.classes - 1))

        return output


class ChanVeseOperation(BaseSegmentationOperation):
    """
    Applies Chan-Vese active contour segmentation.
    
    This is an energy-minimization segmentation method that can detect objects
    without well-defined boundaries. It's particularly useful for medical imaging.
    """
    
    def __init__(
        self,
        max_iter: int = 200,
        tol: float = 1e-3,
        mu: float = 0.25,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        dt: float = 0.5,
    ):
        super().__init__()
        # Validate parameters
        self.max_iter = max(10, int(max_iter))  # At least 10 iterations
        self.tol = max(1e-6, float(tol))  # Reasonable tolerance minimum
        self.mu = float(mu)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.dt = float(dt)

    def get_operation_name(self) -> str:
        return "Chan-Vese Segmentation"

    def _apply_impl(
        self, image_data: np.ndarray, progress_callback: ProgressCallback
    ) -> np.ndarray:
        self._report_progress(progress_callback, 5, "ChanVese _apply_impl started.")

        try:
            # Ensure debug directory exists
            debug_dir = "temp_debug_images"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)

            print("\n--- CHANVESE DEBUG START ---")
            print(
                f"CHANVESE_DEBUG: Input image_data - shape: {image_data.shape}, dtype: {image_data.dtype}, min: {np.min(image_data):.2f}, max: {np.max(image_data):.2f}"
            )
            plt.imsave(
                os.path.join(debug_dir, "chanvese_0_input.png"), image_data, cmap="gray"
            )

            self._report_progress(progress_callback, 20, "Preparing grayscale image...")
            prepared_image = self._prepare_grayscale(image_data, progress_callback)
            print(
                f"CHANVESE_DEBUG: Prepared grayscale image - shape: {prepared_image.shape}, dtype: {prepared_image.dtype}, min: {np.min(prepared_image):.2f}, max: {np.max(prepared_image):.2f}"
            )
            if np.all(prepared_image == prepared_image[0, 0]):
                print("CHANVESE_DEBUG: Prepared image is uniform!")
            plt.imsave(
                os.path.join(debug_dir, "chanvese_1_prepared_gray.png"),
                prepared_image,
                cmap="gray",
            )

            # Start progress indication
            self._report_progress(progress_callback, 40, "Starting Chan-Vese segmentation...")

            # Convert image to float type
            prepared_image_float = util.img_as_float(prepared_image).astype(np.float64)
            print(
                f"CHANVESE_DEBUG: Prepared image as float - shape: {prepared_image_float.shape}, dtype: {prepared_image_float.dtype}, min: {np.min(prepared_image_float):.2f}, max: {np.max(prepared_image_float):.2f}"
            )
            plt.imsave(
                os.path.join(debug_dir, "chanvese_2_prepared_float.png"),
                prepared_image_float,
                cmap="gray",
            )

            # Set up a thread to update progress based on expected iterations
            import threading
            import time

            # Record start time
            start_time = time.time()

            # Flag to stop the progress thread
            stop_progress_thread = threading.Event()

            # Create a thread to simulate progress updates during iterations
            def progress_reporter_thread():
                iter_count = 0
                total_iters = self.max_iter
                sleep_time = 0.1  # Sleep 100ms between updates
                
                # Calculate the expected time per iteration (assume each iteration takes equal time)
                # Based on typical Chan-Vese performance
                expected_total_time = total_iters * 0.05  # Estimate 50ms per iteration
                
                while not stop_progress_thread.is_set() and iter_count < total_iters:
                    # Calculate progress percentage (from 40% to 90%)
                    progress = 40 + int((iter_count / total_iters) * 50)
                    
                    # Update progress bar with iteration count
                    self._report_progress(
                        progress_callback, 
                        progress, 
                        f"Chan-Vese iteration {iter_count}/{total_iters}..."
                    )
                    
                    # Sleep for a short time
                    time.sleep(sleep_time)
                    
                    # Increment iteration counter
                    iter_count += 1
                    
                    # If we're taking longer than expected, slow down the updates
                    elapsed = time.time() - start_time
                    if elapsed > (iter_count * sleep_time * 2):
                        sleep_time = 0.2  # Slow down if taking longer than expected

            # Start the progress update thread
            progress_thread = threading.Thread(target=progress_reporter_thread)
            progress_thread.daemon = True
            progress_thread.start()

            try:
                # Run Chan-Vese algorithm
                segmented_image, phi, energies = segmentation.chan_vese(
                    prepared_image_float,
                    max_num_iter=self.max_iter,
                    tol=self.tol,
                    mu=self.mu,
                    lambda1=self.lambda1,
                    lambda2=self.lambda2,
                    dt=self.dt,
                    init_level_set="checkerboard",
                    extended_output=True
                )
                
                # Stop the progress thread
                stop_progress_thread.set()
                if progress_thread.is_alive():
                    progress_thread.join(timeout=1.0)
                
                # Calculate total elapsed time
                elapsed_time = time.time() - start_time
                
                # Show completion information
                self._report_progress(
                    progress_callback, 
                    95, 
                    f"Chan-Vese completed: {len(energies)} iterations in {elapsed_time:.2f}s"
                )
                
                print(
                    f"CHANVESE_DEBUG: Chan-Vese algorithm completed with {len(energies)} iterations in {elapsed_time:.2f}s. Final energy: {energies[-1]:.6f}"
                )
                print(
                    f"CHANVESE_DEBUG: Raw segmented_image from chan_vese - shape: {segmented_image.shape}, dtype: {segmented_image.dtype}, min: {np.min(segmented_image)}, max: {np.max(segmented_image)}"
                )
                print(
                    f"CHANVESE_DEBUG: Unique values in raw segmented_image: {np.unique(segmented_image)}"
                )
                
                # Format the segmented image as uint8
                result_image = util.img_as_ubyte(segmented_image)
                print(
                    f"CHANVESE_DEBUG: Final result_image (ubyte) - shape: {result_image.shape}, dtype: {result_image.dtype}, min: {np.min(result_image)}, max: {np.max(result_image)}"
                )
                
                # Final progress update
                self._report_progress(progress_callback, 100, "Chan-Vese segmentation complete.")
                
                return result_image
            except Exception as e:
                # Stop the progress thread if still running
                stop_progress_thread.set()
                if progress_thread.is_alive():
                    progress_thread.join(timeout=1.0)
                
                print(f"CHANVESE_DEBUG: Error in Chan-Vese algorithm: {e}")
                import traceback
                print(traceback.format_exc())
                self._report_progress(progress_callback, 100, f"Error: {e}")
                raise

        except Exception as e:
            print(f"CHANVESE_DEBUG: EXCEPTION in _apply_impl: {e}")
            import traceback

            print(traceback.format_exc())
            self._report_progress(progress_callback, 100, f"Error: {e}")
            raise


class MorphSnakesOperation(BaseSegmentationOperation):
    """
    Applies Morphological Snakes (active contour) segmentation.
    
    This algorithm uses morphological operations to implement an active contour based
    on the Chan-Vese approach but with improved numerical stability.
    """
    
    def __init__(
        self,
        iterations: int = 35,
        smoothing: int = 1,
        lambda1: float = 1,
        lambda2: float = 1,
    ):
        super().__init__()
        # Ensure parameters are valid
        self.iterations = max(1, int(iterations))  # At least 1 iteration
        self.smoothing = max(1, int(smoothing))  # At least 1 smoothing step
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)

    def get_operation_name(self) -> str:
        return "Morphological Snakes (ACWE)"

    def _apply_impl(
        self, image_data: np.ndarray, progress_callback: ProgressCallback
    ) -> np.ndarray:
        self._report_progress(progress_callback, 5, "MorphSnakes _apply_impl started.")

        try:
            # Ensure debug directory exists
            debug_dir = "temp_debug_images"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)

            print("\n--- MORPHSNAKES DEBUG START ---")
            print(
                f"MORPHSNAKES_DEBUG: Input image_data - shape: {image_data.shape}, dtype: {image_data.dtype}, min: {np.min(image_data):.2f}, max: {np.max(image_data):.2f}"
            )
            plt.imsave(
                os.path.join(debug_dir, "morphsnakes_0_input.png"),
                image_data,
                cmap="gray",
            )

            self._report_progress(progress_callback, 20, "Preparing grayscale image...")
            prepared_image = self._prepare_grayscale(image_data, progress_callback)
            print(
                f"MORPHSNAKES_DEBUG: Prepared grayscale image - shape: {prepared_image.shape}, dtype: {prepared_image.dtype}, min: {np.min(prepared_image):.2f}, max: {np.max(prepared_image):.2f}"
            )
            if np.all(prepared_image == prepared_image[0, 0]):
                print("MORPHSNAKES_DEBUG: Prepared image is uniform!")
            plt.imsave(
                os.path.join(debug_dir, "morphsnakes_1_prepared_gray.png"),
                prepared_image,
                cmap="gray",
            )

            prepared_image_float = util.img_as_float(prepared_image)
            print(
                f"MORPHSNAKES_DEBUG: Prepared image as float - shape: {prepared_image_float.shape}, dtype: {prepared_image_float.dtype}, min: {np.min(prepared_image_float):.2f}, max: {np.max(prepared_image_float):.2f}"
            )
            plt.imsave(
                os.path.join(debug_dir, "morphsnakes_2_prepared_float.png"),
                prepared_image_float,
                cmap="gray",
            )

            self._report_progress(
                progress_callback,
                40,
                f"Applying Morphological Snakes with iterations={self.iterations}, smoothing={self.smoothing}...",
            )

            # İterasyon sayısını parçalara bölerek ilerleme raporlama
            step_size = max(1, self.iterations // 20)  # En az 20 adım rapor et
            progress_increment = (100 - 40) / (self.iterations // step_size)  # 40'tan 100'e kadar
            
            def progress_callback_wrapper(levelset):
                # İterasyon numarasını takip etmek için static bir değer kullanacağız
                if not hasattr(progress_callback_wrapper, "iter_num"):
                    progress_callback_wrapper.iter_num = 0
                
                iter_num = progress_callback_wrapper.iter_num
                
                if iter_num % step_size == 0 or iter_num == self.iterations - 1:
                    progress = 40 + int((iter_num / self.iterations) * 60)
                    self._report_progress(
                        progress_callback, 
                        progress, 
                        f"ACWE iterasyon {iter_num}/{self.iterations}..."
                    )
                
                progress_callback_wrapper.iter_num += 1
                return progress_callback_wrapper.iter_num < self.iterations  # Devam et
            
            segmented_image = segmentation.morphological_chan_vese(
                prepared_image_float,
                num_iter=self.iterations,
                init_level_set="checkerboard",
                smoothing=self.smoothing,
                lambda1=self.lambda1,
                lambda2=self.lambda2,
                iter_callback=progress_callback_wrapper  # İlerleme geri çağrısını ekle
            )

            print(
                f"MORPHSNAKES_DEBUG: Raw segmented_image from morph_chan_vese - shape: {segmented_image.shape}, dtype: {segmented_image.dtype}, min: {np.min(segmented_image)}, max: {np.max(segmented_image)}"
            )
            print(
                f"MORPHSNAKES_DEBUG: Unique values in raw segmented_image: {np.unique(segmented_image)}"
            )
            plt.imsave(
                os.path.join(debug_dir, "morphsnakes_4_segmented_ls_boolean.png"),
                segmented_image.astype(np.uint8) * 255,
                cmap="gray",
            )

            # Explicitly convert to boolean ensure correct scaling by img_as_ubyte
            segmented_ls_bool = segmented_image.astype(bool)
            print(
                f"MORPHSNAKES_DEBUG: segmented_ls_bool - shape: {segmented_ls_bool.shape}, dtype: {segmented_ls_bool.dtype}, unique: {np.unique(segmented_ls_bool)}"
            )

            result_image = util.img_as_ubyte(segmented_ls_bool)
            print(
                f"MORPHSNAKES_DEBUG: Final result_image (ubyte) - shape: {result_image.shape}, dtype: {result_image.dtype}, min: {np.min(result_image)}, max: {np.max(result_image)}"
            )

            # Check if the result is all black or all white
            if np.all(result_image == 0):
                print("MORPHSNAKES_DEBUG: Final image is all black!")
            if np.all(result_image == 255):
                print("MORPHSNAKES_DEBUG: Final image is all white!")

            plt.imsave(
                os.path.join(debug_dir, "morphsnakes_5_final_ubyte.png"),
                result_image,
                cmap="gray",
            )
            print("--- MORPHSNAKES DEBUG END ---\n")

            self._report_progress(
                progress_callback, 100, "Morphological Snakes segmentation complete."
            )
            return result_image
        except Exception as e:
            print(f"MORPHSNAKES_DEBUG: EXCEPTION in _apply_impl: {e}")
            import traceback

            print(traceback.format_exc())
            self._report_progress(progress_callback, 100, f"Error: {e}")
            raise


from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
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

from dialog_base import BaseDialog


# =============================================================================
# DIALOG CLASSES
# =============================================================================

class SegmentationDialog(BaseDialog):
    """
    Dialog for setting parameters for segmentation operations.
    Inherits from BaseDialog.
    """
    
    def __init__(self, parent=None, operation_name=""):
        super().__init__(parent, window_title=f"{operation_name} Parameters")
        self.operation_name = operation_name
        # Base class handles layout, buttons, positioning

    def _add_parameter_widgets(self):
        # Add widgets specific to this dialog to self.param_layout
        if "Otsu" in self.operation_name:
            # Multi-Otsu parameters
            self.classes_spin = self._create_int_parameter(
                self.param_layout, "Number of Classes:", 2, 5, 3
            )
            # Info label
            self._add_info_label(self.param_layout, "Number of segments (usually 2-5).")

        elif "Chan-Vese" in self.operation_name:
            # Chan-Vese parameters
            self.max_iter_spin = self._create_int_parameter(
                self.param_layout, "Max Iterations:", 1, 1000, 100
            )
            self.tol_spin = self._create_float_parameter(
                self.param_layout, "Tolerance:", 0.0001, 0.01, 0.001, 0.0001
            )
            self.mu_spin = self._create_float_parameter(
                self.param_layout, "Mu:", 0.0, 1.0, 0.25, 0.01
            )
            self.lambda1_spin = self._create_float_parameter(
                self.param_layout, "Lambda1:", 0.1, 5.0, 1.0, 0.1
            )
            self.lambda2_spin = self._create_float_parameter(
                self.param_layout, "Lambda2:", 0.1, 5.0, 1.0, 0.1
            )
            self.dt_spin = self._create_float_parameter(
                self.param_layout, "Dt:", 0.1, 2.0, 0.5, 0.1
            )
            self._add_info_label(
                self.param_layout, "Parameters for Chan-Vese segmentation."
            )

        elif "Snakes" in self.operation_name:
            # Morphological Snakes parameters
            self.iterations_spin = self._create_int_parameter(
                self.param_layout, "Iterations:", 1, 200, 35
            )
            self.smoothing_spin = self._create_int_parameter(
                self.param_layout, "Smoothing:", 1, 10, 3
            )
            self._add_info_label(
                self.param_layout, "Parameters for Morphological Snakes."
            )

    # Helper methods are inherited
    # Button creation is inherited
    # Positioning is inherited

    def get_parameters(self):
        # Implement abstract method
        params = {}
        if hasattr(self, "classes_spin"):
            params["classes"] = self.classes_spin.value()
        if hasattr(self, "max_iter_spin"):
            params["max_iter"] = self.max_iter_spin.value()
            params["tol"] = self.tol_spin.value()
            if hasattr(self, "mu_spin"):
                params["mu"] = self.mu_spin.value()
            if hasattr(self, "lambda1_spin"):
                params["lambda1"] = self.lambda1_spin.value()
            if hasattr(self, "lambda2_spin"):
                params["lambda2"] = self.lambda2_spin.value()
            if hasattr(self, "dt_spin"):
                params["dt"] = self.dt_spin.value()
        if hasattr(self, "iterations_spin"):
            params["iterations"] = self.iterations_spin.value()
            params["smoothing"] = self.smoothing_spin.value()
        return params

    # showEvent is handled by base class
    # display_preview method remains specific to this dialog, so we keep it.
    # If preview is a common need, it could be moved/generalized in BaseDialog
    def display_preview(self, image, title="Preview"):
        # This method seems useful, keep it here for now
        try:
            # import matplotlib.pyplot as plt # Already imported at the top of the file
            # from PyQt5.QtGui import QPixmap # Already imported at the top of the file
            # Create a temporary dialog for the preview
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle(title)
            preview_layout = QVBoxLayout(preview_dialog)

            # Create a figure and display the image using matplotlib
            fig, ax = plt.subplots(figsize=(5, 5))  # Adjust size as needed
            ax.imshow(image, cmap="gray")
            ax.axis("off")

            # Save to a temporary file (ensure temp dir exists)
            # import os # Already imported at the top of the file
            temp_dir = "temp"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_path = os.path.join(temp_dir, "preview.png")

            fig.savefig(temp_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)  # Close the figure to free memory

            # Create label and add image using QPixmap
            preview_label = QLabel()
            pixmap = QPixmap(temp_path)
            if pixmap.isNull():
                print(f"Error: Could not load preview image from {temp_path}")
                preview_label.setText("Error loading preview")
            else:
                preview_label.setPixmap(pixmap)
                preview_label.setScaledContents(True)  # Scale if needed
                preview_label.setMinimumSize(200, 200)  # Ensure minimum size
            preview_layout.addWidget(preview_label)

            # Add close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(preview_dialog.accept)  # Use accept
            preview_layout.addWidget(close_button)

            preview_dialog.exec_()

        except ImportError:
            print(
                "Matplotlib is required for preview. Please install it: pip install matplotlib"
            )
        except Exception as e:
            print(f"Error displaying preview: {e}")
            # Optionally show an error message dialog
