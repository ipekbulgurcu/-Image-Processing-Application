import numpy as np
from PyQt5.QtWidgets import QLabel, QMessageBox, QApplication
from PyQt5.QtCore import Qt

from typing import Union, Callable

from worker import WorkerThread
from progress_dialog import ProgressPopup
from skimage import img_as_ubyte


class OperationHandler:
    """Handles running operations and processing their results."""

    def __init__(self, main_window):
        """
        Initializes the operation handler.
        
        Args:
            main_window: The main application window.
        """
        self.main_window = main_window
        self.current_source_image = None
        self.current_output_image = None
        self.progress_popup = None
        self.worker_thread = None

    def set_images(self, source_image, output_image):
        """Sets the current source and output images."""
        self.current_source_image = source_image
        self.current_output_image = output_image

    def create_progress_popup(self):
        """Creates or resets the progress popup dialog."""
        if self.progress_popup is None:
            self.progress_popup = ProgressPopup(self.main_window)
        else:
            self.progress_popup.reset()
        return self.progress_popup

    def run_operation(self, operation, is_redo=False):
        """Runs an operation in a separate thread."""
        if self.main_window.current_source_image is None:
            self.show_no_image_warning()
            return

        # Always use the original source image for operations
        input_image_for_op = self.main_window.current_source_image.copy()
        
        if input_image_for_op is None:
            self.main_window._logMessage(
                "Cannot run operation: Input image for operation is missing.", "error"
            )
            return

        # Ensure progress popup is ready
        self.progress_popup = self.create_progress_popup()
        self.progress_popup.setWindowTitle(
            f"Processing: {operation.get_operation_name()}"
        )
        self.progress_popup.show()
        self.progress_popup.raise_()
        QApplication.processEvents()

        # Pass the is_redo flag to the handler via the thread if needed
        self.worker_thread = WorkerThread(
            operation, input_image_for_op, self.progress_popup.update_progress
        )
        # Connect with a lambda to pass the is_redo flag to the handler
        self.worker_thread.operation_complete.connect(
            lambda result, op, error: self.handle_operation_complete(
                result, op, error, is_redo
            )
        )
        self.worker_thread.start()

    def handle_operation_complete(self, result, operation, error, is_redo=False):
        """Handles the completion of an operation."""
        try:
            # Close progress popup first
            if self.progress_popup and self.progress_popup.isVisible():
                self.progress_popup.close()

            if error:
                self.main_window._logMessage(
                    f"Operation '{operation.get_operation_name()}' failed: {error}",
                    "error",
                )
                # Show error message to user
                QMessageBox.critical(
                    self.main_window,
                    "Operation Failed",
                    f"Operation '{operation.get_operation_name()}' failed:\n{error}",
                )
            elif result is not None:
                op_name = operation.get_operation_name()
                
                # Yeni undo mantığı - önceki durumu kaydet (redo değilse)
                if not is_redo:
                    # Yeni görüntü ve başlık uygulanmadan önceki mevcut durumu kaydet
                    previous_image_data = self.main_window.current_output_image.copy() if self.main_window.current_output_image is not None else None
                    previous_title = self.main_window.outputTitleLabel.text() if self.main_window.current_output_image is not None else "Output"
                    
                    # Eski durumu undo_stack'e ekle
                    self.main_window.edit_handler.add_to_undo_stack(previous_image_data, previous_title)
                    
                    # Redo stack'i temizle (eğer yeni bir operasyonsa)
                    self.main_window.edit_handler.clear_redo()
                
                # Sonucu uygula
                self.main_window._logMessage(
                    f"Operation '{op_name}' completed successfully.",
                    "success",
                )
                self.current_output_image = result  # Update handler's output image
                self.main_window.current_output_image = result # Update MainWindow's output image
                
                # Update display and title
                self.update_image_display(
                    self.main_window.outputPixmapLabel, self.current_output_image
                )
                self.main_window.outputTitleLabel.setText(f"Output ({op_name})")
                
                # Eski add_to_undo(operation) çağrısı kaldırıldı
                # self.main_window.edit_handler.add_to_undo(operation) 

            else:
                # This case (result is None, no error) might indicate an issue
                self.main_window._logMessage(
                    f"Operation '{operation.get_operation_name()}' returned no result and no error. Output not changed.",
                    "warning",
                )

            self.main_window._activateUIComponents()  # Update UI components
            self.worker_thread = None  # Clear thread reference

        except Exception as e:
            # Catch any unexpected errors in the handler itself
            self.main_window._logMessage(
                f"Error in operation completion handler: {str(e)}", "error"
            )
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"An unexpected error occurred while completing the operation:\n{str(e)}",
            )
            self.worker_thread = None  # Ensure thread is cleared even on error

    def show_no_image_warning(self):
        """Shows a warning message when no image is loaded."""
        QMessageBox.warning(
            self.main_window, "No Image Loaded", "Please load an image before applying operations."
        )

    def update_image_display(self, label: QLabel, image_data: Union[np.ndarray, None]):
        """Updates a QLabel with a NumPy image array."""
        from PyQt5.QtGui import QImage, QPixmap
        
        if image_data is None:
            # Determine placeholder text based on label
            placeholder = "No Image Data"
            if label == self.main_window.sourcePixmapLabel:
                placeholder = "Drop or Open Image/Video"
            elif label == self.main_window.outputPixmapLabel:
                placeholder = "Processing Result Area"
            label.setText(placeholder)
            label.setPixmap(QPixmap())
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
                    # Attempt conversion for other types
                    self.main_window._logMessage(
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
                    self.main_window._logMessage(
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
                self.main_window._logMessage(
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

            pixmap = QPixmap.fromImage(qimage_copy)

            # Scale pixmap to fit label while keeping aspect ratio
            scaled_pixmap = pixmap.scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            label.setText("")  # Clear placeholder text

        except Exception as e:
            import traceback

            self.main_window._logMessage(
                f"Error displaying image: {e}\n{traceback.format_exc()}", "error"
            )
            label.setText("Display Error")
            label.setPixmap(QPixmap()) 