# Image Processing Application

This is a Python-based image processing application that provides various tools for image manipulation, analysis, and processing. The application features a user-friendly GUI built with PyQt and incorporates several advanced image processing capabilities.

## Features

### 1. Image Conversion Operations
- Format conversion capabilities
- Image enhancement tools
- Basic image manipulation functions

### 2. Edge Detection
- Implementation of various edge detection algorithms
- Customizable parameters for edge detection
- Real-time preview of edge detection results

### 3. Segmentation Operations
- Image segmentation tools
- Region-based segmentation
- Threshold-based segmentation techniques

### 4. File Operations
- Open and save images in various formats
- Export processed images
- Batch processing capabilities
<img width="2878" height="1707" alt="Ekran görüntüsü 2025-05-14 155555" src="https://github.com/user-attachments/assets/27374296-1687-415c-957f-2b2e0c6c25b7" />

## Project Structure

```
project/
│
├── icons/                  # Application icons and UI elements
├── images/                 # Sample and test images
├── conversion_operations.py    # Image conversion functionality
├── edge_detection_operations.py# Edge detection algorithms
├── segmentation_operations.py  # Segmentation tools
├── image_operations.py     # Core image processing operations
├── gui.py                 # Main GUI implementation
├── operation_handler.py    # Operation management
└── worker.py              # Background processing worker
```

## Requirements

- Python 3.x
- PyQt5
- NumPy
- OpenCV
- scikit-image

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ipekbulgurcu/-Image-Processing-Application.git
```

2. Install the required dependencies:
```bash
pip install PyQt5 numpy opencv-python scikit-image
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Use the GUI to:
   - Load images
   - Apply various image processing operations
   - Save or export processed images
   - View results in real-time

## Features in Detail

### Image Conversion
- Format conversion between different image types
- Color space transformations
- Image enhancement and filtering

### Edge Detection
- Support for multiple edge detection algorithms
- Adjustable parameters for edge detection sensitivity
- Real-time preview of results

### Segmentation
- Region-based segmentation tools
- Threshold-based segmentation
- Custom segmentation parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

İpek Bulgurcu
