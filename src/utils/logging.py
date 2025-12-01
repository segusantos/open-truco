"""Logging utilities for Truco training experiments."""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


def setup_logger(
    name: str = "truco",
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a simple logger for console output.
    
    Args:
        name: Logger name.
        level: Logging level.
        
    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """Log metrics dictionary in a readable format.
    
    Args:
        logger: Logger to use.
        metrics: Dictionary of metrics.
        step: Optional step/iteration number.
        prefix: Optional prefix for log message.
    """
    parts = []
    
    if step is not None:
        parts.append(f"Step {step}")
    
    if prefix:
        parts.append(prefix)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")
    
    logger.info(" | ".join(parts))


class MetricsLogger:
    """Logger that saves metrics to JSON files."""
    
    def __init__(
        self,
        output_dir: Path,
        experiment_name: Optional[str] = None,
    ):
        """Initialize metrics logger.
        
        Args:
            output_dir: Directory for output files.
            experiment_name: Optional experiment name.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.metrics_file = self.output_dir / f"{experiment_name}_metrics.jsonl"
        self.history: list = []
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to file.
        
        Args:
            metrics: Dictionary of metrics.
            step: Optional step number.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        
        if step is not None:
            record["step"] = step
        
        self.history.append(record)
        
        # Append to JSONL file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def save_summary(self, summary: Dict[str, Any]) -> Path:
        """Save experiment summary.
        
        Args:
            summary: Summary dictionary.
            
        Returns:
            Path to summary file.
        """
        summary_file = self.output_dir / f"{self.experiment_name}_summary.json"
        
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary_file
