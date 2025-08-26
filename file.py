# import sys # Removed unused sys import
import os

# Type hinting for MainWindow without circular import
from typing import TYPE_CHECKING

import imageio  # Use imageio for wider format support
import numpy as np
from PyQt5.QtGui import QIcon, QKeySequence

# Removed QStyle from import, as _get_std_icon is removed
from PyQt5.QtWidgets import QAction, QApplication, QFileDialog, QMenu, QMessageBox
from skimage.util import img_as_ubyte  # Add back the removed import

from handler_base import BaseActionsHandler  # Import new base class

if TYPE_CHECKING:
    from gui import MainWindow

# Assuming MainWindow has attributes:
# current_source_image, current_output_image, sourceFilePath, outputFilePath,
# edit_handler, style(), _logMessage(), _updateImageDisplay(), _activateUIComponents(), close()


class FileActionsHandler(BaseActionsHandler):
    """Handles the creation, connection, state, and logic for File menu actions."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self._create_actions()
        self._connect_actions()

    def _create_actions(self):
        """Creates the QAction objects for the File menu."""
        mw = self.main_window  # Alias

        # Use icons from icons folder
        self.openSourceAction = QAction(QIcon("icons/folder.png"), "Open Source...", mw)
        self.openSourceAction.setShortcut(QKeySequence.Open)  # Ctrl+O
        self.openSourceAction.setStatusTip("Open an image file (.jpg, .png, etc.)")
        self.openSourceAction.setToolTip("Open an image file")

        self.saveOutputAction = QAction(
            QIcon("icons/save_output.png"), "Save Output", mw
        )
        self.saveOutputAction.setShortcut(QKeySequence.Save)  # Ctrl+S
        self.saveOutputAction.setStatusTip(
            "Save the output image to the original source file location"
        )
        self.saveOutputAction.setToolTip("Save output to original source path (Ctrl+S)")

        self.saveAsOutputAction = QAction(
            QIcon("icons/save_as.png"), "Save Output As...", mw
        )
        self.saveAsOutputAction.setShortcut(QKeySequence.SaveAs)  # Ctrl+Shift+S
        self.saveAsOutputAction.setStatusTip("Save the output image to a new file")
        self.saveAsOutputAction.setToolTip("Save output to a new file (Ctrl+Shift+S)")

        # Export Menu
        self.exportMenu = QMenu("Export", mw)
        self.exportMenu.setIcon(
            QIcon("icons/export_output.png")
        )  # Corrected icon for submenu

        self.exportSourceAction = QAction("Source", mw)
        self.exportSourceAction.setStatusTip(
            "Export the source image to the other format (.jpg <-> .png)"
        )
        self.exportSourceAction.setToolTip("Export source image to other format")
        self.exportMenu.addAction(self.exportSourceAction)

        self.exportOutputAction = QAction("Output", mw)
        self.exportOutputAction.setStatusTip(
            "Export the output image to the other format (.jpg <-> .png)"
        )
        self.exportOutputAction.setToolTip("Export output image to other format")
        self.exportMenu.addAction(self.exportOutputAction)

        # Exit Action
        # Using a standard icon if available or a placeholder if not
        # exit_icon = self._get_std_icon(QStyle.SP_DialogCloseButton) # This line would be removed or changed
        # For simplicity, using a custom icon path like others or no icon
        self.exitAction = QAction("Exit", mw)  # No icon for ExitAction
        self.exitAction.setShortcut(QKeySequence("Shift+F4"))  # Example shortcut
        self.exitAction.setStatusTip("Exit the application")
        self.exitAction.setToolTip("Exit the application (Shift+F4)")

    def _connect_actions(self):
        """Connects the actions to this handler's logic methods."""
        self.openSourceAction.triggered.connect(self.open_source)
        self.saveOutputAction.triggered.connect(self.save_output)
        self.saveAsOutputAction.triggered.connect(self.save_output_as)
        self.exportSourceAction.triggered.connect(self.export_source)
        self.exportOutputAction.triggered.connect(self.export_output)
        self.exitAction.triggered.connect(self.exit_app)

    # --- Action Logic Methods ---
    def open_source(self):
        mw = self.main_window
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            mw,  # Parent
            "Select Source File",  # Caption
            "",  # Directory
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All Files (*)",  # Filter - adjusted as per requirement
            options=options,
        )
        if fileName:
            try:
                image_data = imageio.v3.imread(fileName)
                mw.sourceFilePath = fileName
                mw.current_source_image = image_data
                fileName = os.path.basename(fileName)

                status_message = f"Source file opened: {fileName} ({image_data.shape}, {image_data.dtype})"
                mw.statusBar.showMessage(status_message, 5000)
                mw._logMessage(status_message, level="success")

                mw.sourceTitleLabel.setText(f"Source: {fileName}")
                mw._updateImageDisplay(mw.sourcePixmapLabel, mw.current_source_image)

                # Clear previous output AND undo/redo history via edit_handler
                if hasattr(mw, "edit_handler") and mw.edit_handler:
                    mw.edit_handler.clear_output(log=False)
                else:
                    mw._logMessage(
                        "Warning: Edit handler not found, cannot clear output/history.",
                        "warning",
                    )
                    # Fallback: Clear output image directly if no handler
                    mw.current_output_image = None
                    mw._updateImageDisplay(
                        mw.outputPixmapLabel, mw.current_output_image
                    )
                    mw.outputTitleLabel.setText("Output")

                mw.outputFilePath = None  # Reset output path as well
                mw._activateUIComponents()  # Enable controls

            except Exception as e:
                error_msg = f"Failed to open or read image file: {fileName}\nError: {e}"
                mw._logMessage(error_msg, level="error")
                QMessageBox.critical(mw, "Error", error_msg)
                # Don't call setInitialState, just ensure UI reflects lack of image
                mw.current_source_image = None
                mw._updateImageDisplay(mw.sourcePixmapLabel, mw.current_source_image)
                mw._activateUIComponents()
        else:
            mw.statusBar.showMessage("File selection cancelled.", 3000)
            mw._logMessage("File selection cancelled.", level="warning")

    def save_output(self):
        mw = self.main_window
        if mw.current_output_image is None:
            mw._logMessage("Save failed: No output image exists.", "warning")
            return

        if not mw.sourceFilePath:
            mw._logMessage(
                "Save failed: Original source path unknown. Use 'Save As' instead.",
                "warning",
            )
            # Automatically call Save As if source path is missing
            self.save_output_as()
            return

        # Ana dizini al (C:\Users\ipekb\Desktop\lab_oop_2)
        current_directory = os.path.dirname(os.path.realpath(__file__))
        
        # Son işlem adını al
        last_op_name = "processed"
        if (
            hasattr(mw, "edit_handler")
            and mw.edit_handler
            and mw.edit_handler.undo_stack
        ):
            last_operation = mw.edit_handler.undo_stack[-1]
            # Tuple kontrolü ekle
            if isinstance(last_operation, tuple):
                last_op_name = "processed"
            else:
                last_op_name = (
                    last_operation.get_operation_name()
                    .lower()
                    .replace(" ", "_")
                    .replace("->", "_to_")
                )
        
        # Kaynak dosya adını ve uzantısını ayır
        file_base, file_ext = os.path.splitext(os.path.basename(mw.sourceFilePath))
        
        # İşlem adını içeren yeni dosya adı oluştur
        new_filename = f"{file_base}_{last_op_name}{file_ext}"
        
        # Ana dizin ile yeni dosya adını birleştir
        targetPath = os.path.join(current_directory, new_filename)

        try:
            self._save_image(mw.current_output_image, targetPath)
            mw.outputFilePath = targetPath  # Update the known output path
            mw._logMessage(
                f"Output saved to main directory: {targetPath}", "success"
            )
            mw.statusBar.showMessage(
                f"Output saved to {os.path.basename(targetPath)} (main directory)", 4000
            )
            mw._activateUIComponents()  # Update button states
        except Exception as e:
            error_msg = f"Failed to save output to {targetPath}: {e}"
            mw._logMessage(error_msg, "error")
            QMessageBox.critical(mw, "Save Error", error_msg)

    def save_output_as(self):
        mw = self.main_window
        if mw.current_output_image is None:
            mw._logMessage("No output image to save.", "warning")
            return

        # Suggest a filename
        suggested_name = "output.png"
        last_op_name = "processed"
        if (
            hasattr(mw, "edit_handler")
            and mw.edit_handler
            and mw.edit_handler.undo_stack
        ):
            last_operation = mw.edit_handler.undo_stack[-1]
            # Tuple kontrolü ekle
            if isinstance(last_operation, tuple):
                last_op_name = "processed"
            else:
                last_op_name = (
                    last_operation.get_operation_name()
                    .lower()
                    .replace(" ", "_")
                    .replace("->", "_to_")
                )

        if mw.sourceFilePath:
            base, _ = os.path.splitext(os.path.basename(mw.sourceFilePath))
            # Default to png, let user choose format
            suggested_name = f"{base}_{last_op_name}.png"
        else:
            suggested_name = f"{last_op_name}.png"

        options = QFileDialog.Options()
        # Ana dizini başlangıç yolu olarak kullan (C:\Users\ipekb\Desktop\lab_oop_2)
        current_directory = os.path.dirname(os.path.realpath(__file__))
        filePath, selected_filter = QFileDialog.getSaveFileName(
            mw,
            "Save Output As...",
            os.path.join(current_directory, suggested_name),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap Image (*.bmp);;TIFF Image (*.tif *.tiff);;All Files (*)",
            options=options,
        )

        if filePath:
            try:
                self._save_image(mw.current_output_image, filePath)
                mw.outputFilePath = filePath  # Store the new path
                mw._logMessage(f"Output saved to new file: {filePath}", "success")
                mw.statusBar.showMessage(
                    f"Output saved to {os.path.basename(filePath)}", 4000
                )
                mw._activateUIComponents()  # Update save button state
            except Exception as e:
                error_msg = f"Failed to save output to {filePath}: {e}"
                mw._logMessage(error_msg, "error")
                QMessageBox.critical(mw, "Save As Error", error_msg)
        else:
            mw._logMessage("Save As cancelled.", "info")

    def _save_image(self, image_data: np.ndarray, file_path: str):
        """Helper to save numpy array as image using imageio."""
        try:
            # Ensure image data is in uint8 format if it's a common type
            if image_data.dtype != np.uint8:
                if image_data.max() <= 1.0 and image_data.min() >= 0.0:
                    image_data = (image_data * 255).astype(np.uint8)
                else:
                    # Attempt conversion, might need adjustments based on data range
                    image_data = image_data.astype(np.uint8)
            imageio.v3.imwrite(file_path, image_data)
            self.main_window._logMessage(f"Image saved to {file_path}", "success")
            return True
        except Exception as e:
            error_msg = f"Failed to save image to {file_path}: {e}"
            self.main_window._logMessage(error_msg, "error")
            QMessageBox.critical(self.main_window, "Save Error", error_msg)
            return False

    def _export_image(self, image_data: np.ndarray, base_path: str, image_type: str):
        """Handles the logic for exporting an image to the other format."""
        mw = self.main_window
        if image_data is None:
            mw._logMessage(
                f"Cannot export {image_type}: Image data is missing.", "warning"
            )
            return
        if not base_path:
            mw._logMessage(
                f"Cannot suggest export name for {image_type}: Base path unknown.",
                "warning",
            )
            base_path = f"{image_type}_export"

        # Determine current and target extensions
        name_part, current_ext_dot = os.path.splitext(os.path.basename(base_path))
        current_ext = current_ext_dot.lower()

        target_ext = ".jpg"
        target_filter = "JPEG Image (*.jpg *.jpeg)"
        if current_ext == ".jpg" or current_ext == ".jpeg":
            target_ext = ".png"
            target_filter = "PNG Image (*.png)"
        # Add handling for other types if needed, default to jpg if unknown
        elif current_ext != ".png":
            mw._logMessage(
                f"Unknown current extension '{current_ext}', exporting as JPG.",
                "warning",
            )

        suggested_name = f"{name_part}_exported{target_ext}"
        initial_dir = os.path.dirname(base_path)

        filePath, _ = QFileDialog.getSaveFileName(
            mw,
            f"Export {image_type} As...",
            os.path.join(initial_dir, suggested_name),
            f"{target_filter};;All Files (*)",
        )

        if filePath:
            try:
                export_img_ubyte = img_as_ubyte(image_data)
                # Ensure JPEG quality setting if saving as JPG (optional)
                # quality = 95
                # io.imsave(filePath, export_img_ubyte, quality=quality if target_ext == ".jpg" else None)
                self._save_image(export_img_ubyte, filePath)
                mw._logMessage(f"{image_type} exported to: {filePath}", "success")
                mw.statusBar.showMessage(
                    f"{image_type} exported to {os.path.basename(filePath)}", 4000
                )
            except Exception as e:
                error_msg = f"Failed to export {image_type} to {filePath}: {e}"
                mw._logMessage(error_msg, "error")
                QMessageBox.critical(mw, "Export Error", error_msg)
        else:
            mw._logMessage(f"Export {image_type} cancelled.", "info")

    def export_source(self):
        self._export_image(
            self.main_window.current_source_image,
            self.main_window.sourceFilePath,
            "Source",
        )

    def export_output(self):
        self._export_image(
            self.main_window.current_output_image,
            self.main_window.outputFilePath or self.main_window.sourceFilePath,
            "Output",
        )

    def exit_app(self):
        self.main_window.close()

    # --- Action State Update ---
    def update_state(self, source_loaded: bool, output_exists: bool):
        """Updates the enabled state of the File menu actions."""
        # Open is always enabled (or handled by main window init state)
        # self.openSourceAction.setEnabled(True)

        # Save Output requires output and a known source path (as per requirement)
        self.saveOutputAction.setEnabled(
            output_exists and bool(self.main_window.sourceFilePath)
        )
        # Save As Output requires output
        self.saveAsOutputAction.setEnabled(output_exists)
        # Export Source requires source
        self.exportSourceAction.setEnabled(source_loaded)
        # Export Output requires output
        self.exportOutputAction.setEnabled(output_exists)
        # Export menu itself
        self.exportMenu.setEnabled(source_loaded or output_exists)
        # Exit is always enabled
        # self.exitAction.setEnabled(True)

    # --- Menu Integration ---
    def add_actions_to_menu(self, file_menu: QMenu):
        """Adds the created actions to the provided QMenu object."""
        file_menu.addAction(self.openSourceAction)
        file_menu.addAction(self.saveOutputAction)
        file_menu.addAction(self.saveAsOutputAction)
        file_menu.addMenu(self.exportMenu)
        file_menu.addSeparator()
        file_menu.addAction(self.exitAction)
