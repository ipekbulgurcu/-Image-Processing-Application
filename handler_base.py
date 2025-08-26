from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QMenu


# Conditional import to avoid circular dependencies
if TYPE_CHECKING:
    from gui import MainWindow


class BaseActionsHandler(ABC):
    """Base class for handling menu actions related to the main window."""

    def __init__(self, main_window: "MainWindow"):
        """Initializes the handler with a reference to the main window."""
        if main_window is None:
            raise ValueError("main_window cannot be None")
        self.main_window = main_window
        self._create_actions()  # Call internal method to create specific actions

    @abstractmethod
    def _create_actions(self):
        """Subclasses must implement this to create their specific QAction objects."""
        pass

    @abstractmethod
    def add_actions_to_menu(self, menu: QMenu):
        """Subclasses must implement this to add their created actions to the provided menu."""
        pass

    @abstractmethod
    def update_state(self, source_loaded: bool, output_exists: bool):
        """Subclasses must implement this to enable/disable their actions based on app state."""
        pass
