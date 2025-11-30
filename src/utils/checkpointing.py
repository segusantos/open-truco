"""Checkpointing utilities for Truco training experiments."""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging

import torch


logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint: Dict[str, Any],
    path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a training checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary (model state, optimizer, etc.).
        path: Path to save checkpoint.
        metadata: Optional metadata to save alongside.
        
    Returns:
        Path to saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata if provided
    if metadata:
        checkpoint["metadata"] = metadata
    
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")
    
    # Save metadata separately for easy access
    if metadata:
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    
    return path


def load_checkpoint(
    path: Path,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint.
    
    Args:
        path: Path to checkpoint file.
        device: Optional device to map tensors to.
        
    Returns:
        Checkpoint dictionary.
    """
    path = Path(path)
    
    if device:
        checkpoint = torch.load(path, map_location=device)
    else:
        checkpoint = torch.load(path)
    
    logger.info(f"Loaded checkpoint from {path}")
    
    return checkpoint


def find_latest_checkpoint(
    checkpoint_dir: Path,
    pattern: str = "checkpoint_*.pt",
) -> Optional[Path]:
    """Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory to search.
        pattern: Glob pattern for checkpoint files.
        
    Returns:
        Path to latest checkpoint, or None if not found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return checkpoints[0]


class CheckpointManager:
    """Manages checkpoint saving with rotation."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_to_keep: int = 5,
        prefix: str = "checkpoint",
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints.
            max_to_keep: Maximum number of checkpoints to keep.
            prefix: Prefix for checkpoint filenames.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.prefix = prefix
        self.checkpoints: list = []
    
    def save(
        self,
        checkpoint: Dict[str, Any],
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save checkpoint and manage rotation.
        
        Args:
            checkpoint: Checkpoint dictionary.
            step: Training step/iteration.
            metadata: Optional metadata.
            
        Returns:
            Path to saved checkpoint.
        """
        filename = f"{self.prefix}_step_{step}.pt"
        path = self.checkpoint_dir / filename
        
        save_checkpoint(checkpoint, path, metadata)
        self.checkpoints.append(path)
        
        # Rotate old checkpoints
        while len(self.checkpoints) > self.max_to_keep:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
                # Also remove metadata file if exists
                meta_path = old_path.with_suffix(".json")
                if meta_path.exists():
                    meta_path.unlink()
                logger.debug(f"Removed old checkpoint: {old_path}")
        
        return path
    
    def load_latest(self, device: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint.
        
        Args:
            device: Optional device to map tensors to.
            
        Returns:
            Checkpoint dictionary or None.
        """
        latest = find_latest_checkpoint(
            self.checkpoint_dir,
            f"{self.prefix}_*.pt",
        )
        
        if latest:
            return load_checkpoint(latest, device)
        
        return None
