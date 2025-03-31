"""
Logging utilities for the PINNs package.
"""

import os
import sys
import torch
import logging
import datetime
from pathlib import Path
from typing import Optional, Union

# Try importing tensorboard for logging training metrics
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def setup_logger(
    name: str,
    log_dir: Union[str, Path],
    level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if file_output:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TensorBoardLogger:
    """TensorBoard logger for training metrics."""
    
    def __init__(self, log_dir: Union[str, Path], enabled: bool = True):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to store TensorBoard logs
            enabled: Whether TensorBoard logging is enabled
        """
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        
        if self.enabled:
            self.log_dir = Path(log_dir) / "tensorboard"
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None
            
            if enabled and not TENSORBOARD_AVAILABLE:
                print("TensorBoard not available. Install with: pip install tensorboard")
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value to TensorBoard.
        
        Args:
            tag: Data identifier
            value: Value to log
            step: Global step value
        """
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        """
        Log multiple scalars to TensorBoard.
        
        Args:
            main_tag: Parent name for the tags
            tag_scalar_dict: Dictionary of tag names and values
            step: Global step value
        """
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_figure(self, tag: str, figure, step: int) -> None:
        """
        Log a matplotlib figure to TensorBoard.
        
        Args:
            tag: Data identifier
            figure: Matplotlib figure to log
            step: Global step value
        """
        if self.enabled and self.writer:
            self.writer.add_figure(tag, figure, step)
    
    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self.enabled and self.writer:
            self.writer.close()