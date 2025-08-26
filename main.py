import sys

from PyQt5.QtWidgets import QApplication

# Assuming your main window class is MainWindow in gui.py
try:
    from gui import MainWindow
except ImportError as e:
    print(f"Error importing MainWindow from gui.py: {e}")
    print("Please ensure gui.py exists and contains the MainWindow class.")
    sys.exit(1)

if __name__ == "__main__":
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Apply a style if desired (optional, e.g., 'Fusion')
    # app.setStyle('Fusion')

    # Create and show the main window
    mainWindow = MainWindow()
    # mainWindow.show() # The show() call is already in MainWindow's __init__

    # Run the application's event loop
    sys.exit(app.exec_())
