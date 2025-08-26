# Type hinting for MainWindow without circular import
from typing import TYPE_CHECKING

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QMessageBox, QMenu
import numpy as np

# Import necessary operations for type checking/logic if moving recompute here
from conversion_operations import (
    BinaryThresholdOperation,
    GrayscaleOperation,
    HsvOperation,
)
from edge_detection_operations import (
    PrewittOperation,
    RobertsOperation,
    ScharrOperation,
    SobelOperation,
)
from handler_base import BaseActionsHandler  # Import new base class
from segmentation_operations import (
    ChanVeseOperation,
    MorphSnakesOperation,
    MultiOtsuOperation,
)

# Assuming MainWindow has methods: _clearSource, _clearOutput, _undo, _redo
# Assuming MainWindow has attributes: undo_stack, redo_stack, style()

if TYPE_CHECKING:
    from gui import MainWindow


class EditActionsHandler(BaseActionsHandler):
    """Handles the creation, connection, state, and logic for Edit menu actions.
    Undo/Redo stack now stores (image_data, title_string) tuples.
    """

    def __init__(self, main_window):
        """
        Initializes the handler and creates the actions.

        Args:
            main_window: The MainWindow instance.
        """
        super().__init__(main_window)
        # Stacks now store tuples: (image_numpy_array_copy, title_string)
        self.undo_stack = []
        self.redo_stack = []
        self._create_actions()
        self._connect_actions()

    def _create_actions(self):
        """Creates the QAction objects for the Edit menu."""
        self.undoAction = QAction(QIcon("icons/undo.png"), "&Undo", self.main_window)
        self.undoAction.setShortcut("Ctrl+Z")
        self.undoAction.setStatusTip("Undo the last operation")
        self.undoAction.setToolTip("Undo the last operation (Ctrl+Z)")

        self.redoAction = QAction(QIcon("icons/redo.png"), "&Redo", self.main_window)
        self.redoAction.setShortcut("Ctrl+Y")
        self.redoAction.setStatusTip("Redo the last undone operation")
        self.redoAction.setToolTip("Redo the last undone operation (Ctrl+Y)")

        self.clearMenu = QMenu("&Clear", self.main_window)
        self.clearSourceAction = QAction("Clear Source Image", self.main_window)
        self.clearSourceAction.setToolTip("Clear source image and all history")
        self.clearOutputAction = QAction("Clear Output Image", self.main_window)
        self.clearOutputAction.setToolTip("Clear only the output image and history")
        self.clearMenu.addAction(self.clearSourceAction)
        self.clearMenu.addAction(self.clearOutputAction)

    def _connect_actions(self):
        """Connects the actions to this handler's logic methods."""
        self.clearSourceAction.triggered.connect(self.clear_source)
        self.clearOutputAction.triggered.connect(self.clear_output)
        self.undoAction.triggered.connect(self.undo)
        self.redoAction.triggered.connect(self.redo)

    # --- State Query Methods ---
    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    # --- Stack Management Methods ---
    def add_to_undo_stack(self, image_data, title_string):
        """Adds the given image state (a copy) and title to the undo stack."""
        # Ensure image_data is copied if it's a mutable numpy array
        # If image_data can be None (e.g. for initial "empty" output state), handle that.
        img_copy = image_data.copy() if image_data is not None else None
        self.undo_stack.append((img_copy, title_string))
        # Optional: Implement max stack size here
        # MAX_UNDO_STEPS = 20
        # while len(self.undo_stack) > MAX_UNDO_STEPS:
        #     self.undo_stack.pop(0)

    def clear_redo(self):
        if self.redo_stack:
            self.redo_stack.clear()
            self.main_window._logMessage(
                "Redo stack cleared due to new operation.", "info"
            )

    def clear_stacks(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.main_window._logMessage("Undo/Redo history cleared.", "info")

    # --- Action Logic Methods ---
    def clear_source(self):
        """Clears the source image, output image, and optionally resets history after confirmation."""
        reply = QMessageBox.question(
            self.main_window,
            "Confirm Clear",
            "Are you sure you want to clear the source image? This will also clear the output image.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            mw = self.main_window
            mw.current_source_image = None
            mw.sourceFilePath = None
            mw.current_output_image = None
            mw.outputFilePath = None
            
            # Artık geçmiş temizlenmiyor
            # self.clear_stacks() # Clears undo/redo image stacks
            
            mw._logMessage("Source image cleared. Output image also cleared.", "info")
            mw._updateImageDisplay(mw.sourcePixmapLabel, None)
            mw._updateImageDisplay(mw.outputPixmapLabel, None)
            mw.sourceTitleLabel.setText("Source")
            mw.outputTitleLabel.setText("Output")
            mw._activateUIComponents()
        else:
            self.main_window._logMessage("Clear source cancelled.", "info")

    def clear_output(self, log: bool = True):
        """Clears the output image without resetting undo/redo stacks."""
        mw = self.main_window
        if mw.current_output_image is not None: # Sadece output varsa temizle
            mw.current_output_image = None
            if log:
                mw._logMessage("Output cleared. History preserved.", "info")
            
            # Artık geçmiş temizlenmiyor
            # self.clear_stacks() # Clear image history
            
            mw._updateImageDisplay(mw.outputPixmapLabel, None)
            mw.outputTitleLabel.setText("Output")
            mw._activateUIComponents()
        elif log:
            mw._logMessage("Clear Output: No output image exists.", "warning")

    def undo(self):
        """Performs the undo operation by restoring image state from the stack."""
        mw = self.main_window
        if hasattr(mw, 'worker_thread') and mw.worker_thread and mw.worker_thread.isRunning():
            QMessageBox.warning(mw, "Meşgul", "Ana bir işlem çalışırken geri alamazsınız.")
            return
        # No recompute_thread to check anymore

        if self.can_undo():
            # State to restore from undo_stack
            image_to_restore, title_to_restore = self.undo_stack.pop()
            
            # Current state to push to redo_stack
            current_image_for_redo = mw.current_output_image.copy() if mw.current_output_image is not None else None
            current_title_for_redo = mw.outputTitleLabel.text() # Get title *before* changing it
            self.redo_stack.append((current_image_for_redo, current_title_for_redo))
            
            # Restore state
            mw.current_output_image = image_to_restore # This is already a copy or None from add_to_undo_stack
            mw.outputTitleLabel.setText(title_to_restore)
            
            mw._logMessage(f"Geri alındı. Gösterilen durum: '{title_to_restore}'", "info")
            mw.statusBar.showMessage(f"Geri alındı: {title_to_restore}", 3000)
            mw._updateImageDisplay(mw.outputPixmapLabel, mw.current_output_image)
            mw._activateUIComponents()
        else:
            mw._logMessage("Geri alınacak bir şey yok.", "warning")
            mw.statusBar.showMessage("Geri alınacak bir şey yok.", 3000)

    def redo(self):
        """Performs the redo operation by restoring image state from the stack."""
        mw = self.main_window
        if hasattr(mw, 'worker_thread') and mw.worker_thread and mw.worker_thread.isRunning():
            QMessageBox.warning(mw, "Meşgul", "Ana bir işlem çalışırken yineleyemezsiniz.")
            return
        # No recompute_thread to check anymore

        if self.can_redo():
            # State to restore from redo_stack
            image_to_restore, title_to_restore = self.redo_stack.pop()
            
            # Current state to push to undo_stack
            current_image_for_undo = mw.current_output_image.copy() if mw.current_output_image is not None else None
            current_title_for_undo = mw.outputTitleLabel.text() # Get title *before* changing it
            self.undo_stack.append((current_image_for_undo, current_title_for_undo))
            
            # Restore state
            mw.current_output_image = image_to_restore # This is already a copy or None
            mw.outputTitleLabel.setText(title_to_restore)

            mw._logMessage(f"Yinelendi. Gösterilen durum: '{title_to_restore}'", "info")
            mw.statusBar.showMessage(f"Yinelendi: {title_to_restore}", 3000)
            mw._updateImageDisplay(mw.outputPixmapLabel, mw.current_output_image)
            mw._activateUIComponents()
        else:
            mw._logMessage("Yinelenecek bir şey yok.", "warning")
            mw.statusBar.showMessage("Yinelenecek bir şey yok.", 3000)

    # Eski recompute_output_from_stack metodu ve add_to_undo(self, operation) metodu kaldırıldı.
    # _handle_recompute_complete metodu da kaldırıldı çünkü artık thread yok.

    # --- Action State Update ---
    def update_state(self, source_loaded: bool, output_exists: bool):
        """Updates the enabled state of the menu actions based on app state."""
        can_undo_flag = self.can_undo()
        can_redo_flag = self.can_redo()

        self.clearSourceAction.setEnabled(source_loaded)
        # Output_exists şimdi biraz daha karmaşık. Sadece current_output_image değil, undo_stack'in varlığı da output'u temizlenebilir yapar.
        # clearOutputAction.setEnabled(output_exists or self.can_undo()) # Eğer undo varsa output temizlenebilir.
        # Veya daha basit: Output varsa veya undo yığını doluysa temizle butonu aktif olsun.
        # Şimdilik MainWindow'daki _activateUIComponents'in output_exists'e göre yönettiğini varsayalım,
        # ama clear_output'un kendisi yığınları da temizliyor.
        # Bu metodun çağrıldığı yerlerdeki output_exists mantığı önemli.
        # self.main_window._activateUIComponents output_exists'i current_output_image'e göre belirler.
        # Belki de clearOutputAction'ın aktifliği (output_exists or bool(self.undo_stack)) olmalı.
        # Şimdilik basit tutalım:
        self.clearOutputAction.setEnabled(output_exists or bool(self.undo_stack))

        self.undoAction.setEnabled(can_undo_flag)
        self.redoAction.setEnabled(can_redo_flag)
        self.clearMenu.setEnabled(source_loaded or output_exists or bool(self.undo_stack))

    # --- Menu Integration ---
    def add_actions_to_menu(self, edit_menu: QMenu):
        """Adds the created actions to the provided QMenu object."""
        edit_menu.addMenu(self.clearMenu)
        edit_menu.addSeparator()
        edit_menu.addAction(self.undoAction)
        edit_menu.addAction(self.redoAction)
