#!/usr/bin/env python3
"""
Main training entry point for Reinforcement Learning Crypto Trading System.
Supports Multi-GPU training, CPU parallel Qwen 8B news engine, 100% offline execution,
per-epoch PNG visualization, and resumable training.

Usage:
  python train_rl.py --epochs 50 --num_workers 4 --initial_balance 2000
  python train_rl.py --resume
"""

import os
import sys
import argparse
import logging
import torch

from src.config import TradingConfig
from src.multi_gpu_trainer import MultiGPUTrainer


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Multi-Asset Crypto RL Trading Agent with Qwen 8B News Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs (1 epoch = 1 trading day / 1440 mins)")
    parser.add_argument("--initial_balance", type=float, default=2000.0, help="Starting simulated balance in USD")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel worker agents (auto-detected if None)")
    parser.add_argument("--gpus", type=str, default="auto", help="GPU device IDs (e.g. '0,1' or 'auto' or 'cpu')")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for PPO Actor-Critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for PPO updates")
    parser.add_argument("--mini_epochs", type=int, default=4, help="PPO mini-epochs per rollout")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if available")
    parser.add_argument("--checkpoint", type=str, default="latest", help="Specific checkpoint file to load when resuming")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing minute-by-minute CSV datasets")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory containing Qwen model files")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory to save per-epoch PNG charts")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    # Build Configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = TradingConfig(
        initial_balance=args.initial_balance,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        mini_epochs=args.mini_epochs,
        total_epochs=args.epochs,
        data_dir=os.path.join(base_dir, args.data_dir),
        models_dir=os.path.join(base_dir, args.models_dir),
        checkpoints_dir=os.path.join(base_dir, args.checkpoints_dir),
        plots_dir=os.path.join(base_dir, args.plots_dir),
    )

    # Parse GPU IDs
    gpu_ids = None
    if args.gpus.lower() == "cpu":
        gpu_ids = []
    elif args.gpus.lower() != "auto":
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip().isdigit()]

    # Instantiate Trainer
    trainer = MultiGPUTrainer(
        config=config,
        num_workers=args.num_workers,
        gpu_ids=gpu_ids,
    )

    # Check resumption
    if args.resume:
        chk_path = args.checkpoint if args.checkpoint.endswith(".pt") else None
        trainer.resume_if_requested(resume=True, checkpoint_path=chk_path)

    # Run Training
    trainer.train(total_epochs=args.epochs)


if __name__ == "__main__":
    main()
