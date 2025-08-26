"""
Image Operations Module
Contains the ImageOperations class which provides a unified interface for applying
various image processing operations, organized by inheritance hierarchy.
"""

# Conversion Operations (BaseConversionOperation)
from conversion_operations import (
    GrayscaleOperation,
    HsvOperation, 
    BinaryThresholdOperation,
    AdaptiveThresholdOperation
)

# Edge Detection Operations (BaseEdgeDetectionOperation)
from edge_detection_operations import (
    RobertsOperation, 
    SobelOperation, 
    ScharrOperation, 
    PrewittOperation
)

# Segmentation Operations (BaseSegmentationOperation)
from segmentation_operations import (
    MultiOtsuOperation, 
    ChanVeseOperation, 
    MorphSnakesOperation
)


class ImageOperations:
    """
    Unified interface for applying various image processing operations.
    Operations are organized according to their inheritance hierarchy:
    - Conversion operations (BaseConversionOperation subclasses)
    - Edge detection operations (BaseEdgeDetectionOperation subclasses)
    - Segmentation operations (BaseSegmentationOperation subclasses)
    """

    def __init__(self, operation_handler):
        """
        Initializes the image operations with the given operation handler.
        
        Args:
            operation_handler: The handler for running operations.
        """
        self.operation_handler = operation_handler

    # ===== CONVERSION OPERATIONS =====
    # These operations inherit from BaseConversionOperation

    def apply_grayscale(self):
        """Applies grayscale conversion."""
        op = GrayscaleOperation()
        self.operation_handler.run_operation(op)

    def apply_hsv(self, hue_shift=0.5, saturation_scale=2.0, value_scale=1.5):
        """
        Applies RGB to HSV conversion with the given parameters.
        
        Args:
            hue_shift: Amount to shift the hue (0-1).
            saturation_scale: Factor to scale the saturation.
            value_scale: Factor to scale the value/brightness.
        """
        op = HsvOperation(
            hue_shift=hue_shift, 
            saturation_scale=saturation_scale, 
            value_scale=value_scale
        )
        self.operation_handler.run_operation(op)

    def apply_binary_threshold(self, threshold, invert=False):
        """
        Applies binary thresholding with the given parameters.
        
        Args:
            threshold: The threshold value (0-1).
            invert: Whether to invert the output.
        """
        op = BinaryThresholdOperation(threshold=threshold, invert=invert)
        self.operation_handler.run_operation(op)
        
    def apply_adaptive_threshold(self, block_size=35, constant=0.0):
        """
        Applies adaptive thresholding with the given parameters.
        
        Args:
            block_size: Size of local neighborhood for adaptive thresholding.
            constant: Constant subtracted from weighted mean.
        """
        op = AdaptiveThresholdOperation(block_size=block_size, constant=constant)
        self.operation_handler.run_operation(op)

    # ===== SEGMENTATION OPERATIONS =====
    # These operations inherit from BaseSegmentationOperation

    def apply_multi_otsu(self, classes=3):
        """
        Applies Multi-Otsu segmentation with the given parameters.
        
        Args:
            classes: Number of classes to segment the image into.
        """
        op = MultiOtsuOperation(classes=classes)
        self.operation_handler.run_operation(op)

    def apply_chan_vese(self, max_iter=200, tol=0.001, mu=0.25, lambda1=1.0, lambda2=1.0, dt=0.5):
        """
        Applies Chan-Vese segmentation with the given parameters.
        
        Args:
            max_iter: Maximum number of iterations.
            tol: Tolerance for convergence.
            mu, lambda1, lambda2, dt: Parameters for the Chan-Vese algorithm.
        """
        op = ChanVeseOperation(
            max_iter=max_iter, 
            tol=tol, 
            mu=mu, 
            lambda1=lambda1, 
            lambda2=lambda2, 
            dt=dt
        )
        self.operation_handler.run_operation(op)

    def apply_morph_snakes(self, iterations=50, smoothing=3, lambda1=1.0, lambda2=1.0):
        """
        Applies Morphological Snakes segmentation with the given parameters.
        
        Args:
            iterations: Number of iterations.
            smoothing: Smoothing parameter.
            lambda1, lambda2: Parameters for the algorithm.
        """
        op = MorphSnakesOperation(
            iterations=iterations, 
            smoothing=smoothing, 
            lambda1=lambda1, 
            lambda2=lambda2
        )
        self.operation_handler.run_operation(op)

    # ===== EDGE DETECTION OPERATIONS =====
    # These operations inherit from BaseEdgeDetectionOperation
    
    def get_edge_detection_params(self, threshold, sigma):
        """
        Helper to process edge detection parameters.
        
        Args:
            threshold: The threshold value (0-1 or None).
            sigma: Sigma for Gaussian blur.
            
        Returns:
            Dictionary with processed parameters.
        """
        # Use None for threshold if value is 0.0, indicating auto-threshold
        return {"threshold": threshold if threshold > 0.0 else None, "sigma": sigma} 

    def apply_roberts(self, threshold=None, sigma=0.0):
        """
        Applies Roberts edge detection with the given parameters.
        
        Args:
            threshold: Threshold for edge detection (None for auto).
            sigma: Sigma for Gaussian blur (0.0 for no blur).
        """
        op = RobertsOperation(threshold=threshold, sigma=sigma)
        self.operation_handler.run_operation(op)

    def apply_sobel(self, threshold=None, sigma=0.0):
        """
        Applies Sobel edge detection with the given parameters.
        
        Args:
            threshold: Threshold for edge detection (None for auto).
            sigma: Sigma for Gaussian blur (0.0 for no blur).
        """
        op = SobelOperation(threshold=threshold, sigma=sigma)
        self.operation_handler.run_operation(op)

    def apply_scharr(self, threshold=None, sigma=0.0):
        """
        Applies Scharr edge detection with the given parameters.
        
        Args:
            threshold: Threshold for edge detection (None for auto).
            sigma: Sigma for Gaussian blur (0.0 for no blur).
        """
        op = ScharrOperation(threshold=threshold, sigma=sigma)
        self.operation_handler.run_operation(op)

    def apply_prewitt(self, threshold=None, sigma=0.0):
        """
        Applies Prewitt edge detection with the given parameters.
        
        Args:
            threshold: Threshold for edge detection (None for auto).
            sigma: Sigma for Gaussian blur (0.0 for no blur).
        """
        op = PrewittOperation(threshold=threshold, sigma=sigma)
        self.operation_handler.run_operation(op) 