import abc

from PyQt5.QtCore import Qt, pyqtSignal
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

# Resolve metaclass conflict between QDialog's metaclass and ABCMeta
qt_meta = type(QDialog)


class BaseDialog(QDialog):
    """Base class for parameter input dialogs."""

    parameter_changed = pyqtSignal()

    def __init__(self, parent=None, window_title="Parameters", min_width=350):
        super().__init__(parent)
        self.setWindowTitle(window_title)
        self.setModal(True)
        self.setMinimumWidth(min_width)

        self.main_layout = QVBoxLayout(self)
        self.param_layout = (
            QVBoxLayout()
        )  # Layout for parameter widgets added by subclasses
        self.main_layout.addLayout(self.param_layout)

        # --- Abstract method for subclasses to implement ---
        self._add_parameter_widgets()

        # --- Common Buttons ---
        self._add_buttons()

        # --- Final Adjustments ---
        self.adjustSize()
        self._position_dialog()

    @abc.abstractmethod
    def _add_parameter_widgets(self):
        """Subclasses must implement this to add their specific parameter widgets
        to self.param_layout."""
        pass

    @abc.abstractmethod
    def get_parameters(self) -> dict:
        """Subclasses must implement this to return a dictionary of parameters
        based on their widget values."""
        pass

    def _add_buttons(self):
        """Adds standard OK and Cancel buttons."""
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        self.main_layout.addLayout(button_layout)

    def _position_dialog(self):
        """Positions the dialog relative to the parent or screen."""
        if self.parent():
            parent_geo = self.parent().geometry()
            # Position roughly bottom-left relative to parent center
            self.move(parent_geo.left() + 50, parent_geo.bottom() - self.height() - 50)
        else:
            # Default position if no parent (e.g., bottom-left of screen)
            screen = QApplication.primaryScreen().geometry()
            self.move(screen.left() + 100, screen.bottom() - self.height() - 150)

    # --- Helper methods for creating parameter widgets (can be used by subclasses) ---
    def _create_int_parameter(
        self, layout: QVBoxLayout, label: str, min_val: int, max_val: int, default: int
    ) -> QSpinBox:
        """Creates a standard integer parameter widget (Label, Slider, SpinBox) and adds it to the layout."""
        widget = QWidget()
        h_layout = QHBoxLayout(widget)
        h_layout.addWidget(QLabel(label))

        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)

        # Connect slider and spinbox
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)

        # Emit parameter_changed signal when value changes
        spin.valueChanged.connect(self.parameter_changed.emit)

        h_layout.addWidget(slider)
        h_layout.addWidget(spin)
        layout.addWidget(widget)  # Add the combined widget to the provided layout
        return spin  # Return the spinbox for easy access to its value

    def _create_float_parameter(
        self,
        layout: QVBoxLayout,
        label: str,
        min_val: float,
        max_val: float,
        default: float,
        step: float,
        decimals: int = None,
    ) -> QDoubleSpinBox:
        """Creates a standard float parameter widget (Label, Slider, SpinBox) and adds it to the layout."""
        widget = QWidget()
        h_layout = QHBoxLayout(widget)
        h_layout.addWidget(QLabel(label))

        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setSingleStep(step)

        # Calculate decimals if not provided
        if decimals is None:
            if isinstance(step, int) or step == 0:
                decimals = 0
            elif (
                "e" in f"{step:.10e}"
            ):  # Use scientific notation to find order of magnitude
                decimals = abs(int(f"{step:.10e}".split("e")[-1]))
            else:
                decimals = len(str(float(step)).split(".")[-1])
            decimals = min(decimals, 10)  # Limit decimals for sanity
        spin.setDecimals(decimals)

        # Emit parameter_changed signal when value changes
        spin.valueChanged.connect(self.parameter_changed.emit)

        # Slider setup (handle potential division by zero)
        if abs(step) > 1e-9:
            slider = QSlider(Qt.Horizontal)
            slider_min = int(round(min_val / step))
            slider_max = int(round(max_val / step))
            slider_default = int(round(default / step))
            # Ensure min <= max
            if slider_min > slider_max:
                slider_min, slider_max = slider_max, slider_min
            slider.setRange(slider_min, slider_max)
            slider.setValue(slider_default)

            # Connect slider and spinbox
            slider.valueChanged.connect(
                lambda v, s=spin, st=step, d=decimals: s.setValue(round(v * st, d))
            )
            spin.valueChanged.connect(
                lambda v, s=slider, st=step: s.setValue(int(round(v / st)))
            )

            h_layout.addWidget(slider)
        else:  # No slider if step is too small
            h_layout.addStretch()

        h_layout.addWidget(spin)
        layout.addWidget(widget)  # Add the combined widget to the provided layout
        return spin  # Return the spinbox for easy access to its value

    def _add_info_label(self, layout: QVBoxLayout, text: str):
        """Adds a styled informational label to the layout."""
        info_label = QLabel(text)
        info_label.setStyleSheet("color: gray; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
