from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget, QApplication

class ProgressPopup(QDialog):
    """A dialog showing operation progress."""

    def __init__(self, parent=None, title="Processing..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)  # Block interaction with main window
        self.setMinimumWidth(350)  # Biraz daha genişletelim
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
        )  # Çerçevesiz ve her zaman üstte

        # Ana layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Kenar boşluklarını kaldır

        # Stil için bir çerçeve widget
        frame = QWidget(self)
        frame.setObjectName("ProgressFrame")
        main_layout.addWidget(frame)

        layout = QVBoxLayout(frame)  # Asıl içerik bu frame içine
        layout.setContentsMargins(20, 20, 20, 20)  # İç boşluklar
        layout.setSpacing(15)  # Widget'lar arası boşluk

        self.titleLabel = QLabel(title)  # Başlık için ayrı bir label
        self.titleLabel.setObjectName("ProgressTitleLabel")
        self.titleLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.titleLabel)

        self.messageLabel = QLabel("Starting operation...")
        self.messageLabel.setObjectName("ProgressMessageLabel")
        self.messageLabel.setAlignment(Qt.AlignCenter)
        self.messageLabel.setWordWrap(True)
        layout.addWidget(self.messageLabel)

        self.progressBar = QProgressBar()
        self.progressBar.setObjectName("CustomProgressBar")
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(
            False
        )  # Yüzde metnini gizle, chunk'larla daha iyi durur
        self.progressBar.setFixedHeight(12)  # Yüksekliği ayarla
        layout.addWidget(self.progressBar)

        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.setObjectName("CancelProgressButton")
        self.cancelButton.clicked.connect(self.reject)  # reject() closes the dialog
        layout.addWidget(self.cancelButton, 0, Qt.AlignRight)  # Sağa yasla

        self._apply_styles()

    def _apply_styles(self):
        # Renkler (M3 stilinden esinlenilmiş)
        primary_color = "#6750A4"  # Mor
        surface_color = "#FFFBFE"  # Neredeyse beyaz arka plan
        on_surface_color = "#1C1B1F"  # Koyu metin
        outline_color = "#79747E"  # Kenarlık rengi
        primary_container_color = "#EADDFF"  # Açık mor (hover vb.)
        on_primary_container_color = "#21005D"
        error_color = "#B3261E"

        self.setStyleSheet(
            f"""
            QDialog#ProgressPopup {{ /* Bu ID'yi ProgressPopup'a atamamız gerekebilir veya doğrudan QDialog */
                background-color: transparent; /* Ana dialog transparan */
            }}
            QWidget#ProgressFrame {{ /* Stil uygulanan çerçeve */
                background-color: {surface_color};
                border-radius: 12px;
                border: 1px solid {outline_color};
                /* Gölge efekti siliyor, PyQt5'te desteklenmiyor */
            }}
            QLabel#ProgressTitleLabel {{
                font-size: 14pt;
                font-weight: bold;
                color: {primary_color};
                padding-bottom: 5px;
            }}
            QLabel#ProgressMessageLabel {{
                font-size: 10pt;
                color: {on_surface_color};
                min-height: 30px; /* Mesajlar için yer ayır */
            }}
            QProgressBar#CustomProgressBar {{
                border: 1px solid {outline_color};
                border-radius: 6px;
                background-color: {primary_container_color}; /* Chunk'ların olmadığı arka plan */
                text-align: center; /* Yüzde metni gösterilirse */
            }}
            QProgressBar#CustomProgressBar::chunk {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0, 
                    stop:0 {primary_color}, stop:1 {primary_color} /* Daha yumuşak bir gradient için stop:1 #8A70CC gibi */
                );
                border-radius: 5px; /* Chunk'ların kenar yuvarlaklığı */
                margin: 1px; /* Chunk ve border arasında küçük bir boşluk */
            }}
            QPushButton#CancelProgressButton {{
                background-color: transparent;
                color: {error_color};
                border: 1px solid {error_color};
                padding: 6px 12px;
                border-radius: 16px;
                font-size: 9pt;
                min-height: 28px;
                font-weight: 500;
            }}
            QPushButton#CancelProgressButton:hover {{
                background-color: {error_color};
                color: #FFFFFF;
            }}
            QPushButton#CancelProgressButton:pressed {{
                background-color: #A01C14; /* Daha koyu kırmızı */
                color: #FFFFFF;
            }}
        """
        )
        self.setObjectName("ProgressPopup")  # QDialog#ProgressPopup ID'si için

    def update_progress(self, percentage: int, message: str):
        """Updates the progress bar and message."""
        self.progressBar.setValue(percentage)
        self.messageLabel.setText(message)
        QApplication.processEvents()  # Force GUI update

    def reset(self):
        """Resets the progress bar and message."""
        self.progressBar.setValue(0)
        self.messageLabel.setText("Starting operation...")

    def closeEvent(self, event):
        """Handle window close."""
        self.reset()
        event.accept() 