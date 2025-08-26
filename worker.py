from PyQt5.QtCore import QThread, pyqtSignal

class WorkerThread(QThread):
    """Runs an operation in a separate thread."""

    operation_complete = pyqtSignal(object, object, object)

    def __init__(self, operation_instance, image_data, progress_callback):
        super().__init__()
        self.operation = operation_instance
        self.image_data = image_data
        self.progress_callback = progress_callback

    def run(self):
        try:
            result = self.operation.apply(self.image_data, self.progress_callback)
            self.operation_complete.emit(result, self.operation, None)
        except Exception as e:
            self.operation_complete.emit(None, self.operation, str(e)) 