"""
Checkpoint manager for saving and resuming RL training sessions seamlessly.
"""

import os
import glob
import json
import torch
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("CheckpointManager")


class CheckpointManager:
    """
    Manages saving and restoring training state (model weights, optimizer state, epoch count, history).
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        self.history_path = os.path.join(self.checkpoint_dir, "training_history.json")

    def has_checkpoint(self) -> bool:
        return os.path.exists(self.latest_path)

    def save_checkpoint(
        self,
        epoch: int,
        model_state_dict: Dict[str, Any],
        optimizer_state_dict: Dict[str, Any],
        history: list,
        best_profit: float,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Saves both latest and epoch-specific checkpoint files."""
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "history": history,
            "best_profit": best_profit,
            "config": config_dict or {},
        }

        # Save latest
        torch.save(checkpoint_data, self.latest_path)

        # Save per-epoch checkpoint
        epoch_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        torch.save(checkpoint_data, epoch_path)

        # Save history JSON for easy inspection
        try:
            with open(self.history_path, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write history JSON: {e}")

        logger.info(f"Saved checkpoint for epoch {epoch} to {epoch_path}")
        return epoch_path

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[Dict[str, Any]]:
        """
        Loads checkpoint from path or latest_checkpoint.pt.
        """
        if checkpoint_path is None or checkpoint_path.lower() == "latest":
            checkpoint_path = self.latest_path

        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return None

        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint
