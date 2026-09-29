"""
Multi-GPU and Multi-Worker Parallel Training Orchestrator.
Coordinates distributed / parallel agent rollouts, Qwen 8B news processing,
per-epoch PNG generation, and checkpointing.
"""

import os
import time
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .config import TradingConfig
from .data_loader import CryptoDataLoader
from .qwen_engine import QwenNewsEngine
from .crypto_env import CryptoTradingEnv
from .rl_agent import PPOAgent
from .visualizer import TradingVisualizer
from .checkpoint import CheckpointManager

logger = logging.getLogger("Trainer")


def run_single_worker_rollout(
    worker_id: int,
    epoch: int,
    agent: PPOAgent,
    env: CryptoTradingEnv,
    visualizer: TradingVisualizer,
    config: TradingConfig,
) -> Dict[str, Any]:
    """
    Executes a single 1440-minute day trading episode for a worker agent.
    Collects rollout transitions, triggers PPO update, and generates per-epoch PNG.
    """
    obs, info = env.reset()
    done = False
    step_count = 0

    while not done:
        action, log_prob, value = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        agent.buffer.add(
            state=obs,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
        )

        obs = next_obs
        step_count += 1
        info = step_info

    # Final bootstrap value for GAE
    _, _, last_value = agent.select_action(obs, deterministic=False)

    # Perform PPO optimization update
    update_metrics = agent.update(
        last_value=last_value,
        batch_size=config.batch_size,
        mini_epochs=config.mini_epochs,
    )

    # Save per-epoch, per-worker PNG plot
    png_path = visualizer.plot_epoch(
        epoch=epoch,
        worker_id=worker_id,
        history=env.history,
        info=info,
    )

    result = {
        "epoch": epoch,
        "worker_id": worker_id,
        "symbol": str(info.get("symbol", "CRYPTO")),
        "date": str(info.get("date", "")),
        "initial_balance": float(env.initial_balance),
        "final_value": float(info.get("portfolio_value", 2000.0)),
        "net_profit": float(info.get("net_profit", 0.0)),
        "pnl_pct": float(info.get("pnl_pct", 0.0)),
        "bench_pnl_pct": float(info.get("bench_pnl_pct", 0.0)),
        "total_trades": int(info.get("total_trades", 0)),
        "win_rate": float(info.get("win_rate", 0.0)),
        "policy_loss": update_metrics.get("policy_loss", 0.0),
        "value_loss": update_metrics.get("value_loss", 0.0),
        "entropy": update_metrics.get("entropy", 0.0),
        "plot_path": png_path,
    }
    return result


class MultiGPUTrainer:
    """
    Orchestrates multi-GPU / multi-worker parallel training.
    """

    def __init__(
        self,
        config: TradingConfig,
        num_workers: Optional[int] = None,
        gpu_ids: Optional[List[int]] = None,
    ):
        self.config = config
        self.data_loader = CryptoDataLoader(config.data_dir)
        self.qwen_engine = QwenNewsEngine(config.data_dir, config.models_dir)
        self.visualizer = TradingVisualizer(config.plots_dir)
        self.checkpoint_manager = CheckpointManager(config.checkpoints_dir)

        # Device & Multi-GPU Detection
        self.gpu_ids = gpu_ids
        if self.gpu_ids is None:
            if torch.cuda.is_available():
                self.gpu_ids = list(range(torch.cuda.device_count()))
            else:
                self.gpu_ids = []

        # Determine number of parallel workers
        if num_workers is not None:
            self.num_workers = max(1, num_workers)
        elif len(self.gpu_ids) > 0:
            self.num_workers = len(self.gpu_ids) * config.workers_per_gpu
        else:
            # CPU multi-worker fallback
            self.num_workers = min(4, max(1, os.cpu_count() or 1))

        # Setup Worker Environments and Agents
        self.workers: List[Dict[str, Any]] = []
        self._init_workers()

        self.epoch_history: List[Dict[str, Any]] = []
        self.best_profit: float = -float("inf")
        self.start_epoch: int = 1

    def _init_workers(self):
        logger.info(f"Initializing {self.num_workers} parallel RL trading workers...")
        for i in range(self.num_workers):
            # Assign device
            if self.gpu_ids:
                gpu_idx = self.gpu_ids[i % len(self.gpu_ids)]
                device = torch.device(f"cuda:{gpu_idx}")
            else:
                device = torch.device("cpu")

            env = CryptoTradingEnv(
                data_loader=self.data_loader,
                qwen_engine=self.qwen_engine,
                config=self.config,
                seed=self.config.seed + i * 100,
            )

            agent = PPOAgent(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                lr=self.config.learning_rate,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_epsilon=self.config.clip_epsilon,
                entropy_coef=self.config.entropy_coef,
                value_loss_coef=self.config.value_loss_coef,
                max_grad_norm=self.config.max_grad_norm,
                device=device,
            )

            self.workers.append({
                "worker_id": i,
                "device": device,
                "env": env,
                "agent": agent,
            })
            logger.info(f"  Worker #{i} initialized on {device}")

    def resume_if_requested(self, resume: bool, checkpoint_path: Optional[str] = None):
        """Loads state from previous checkpoint if resume is True."""
        if not resume:
            return

        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        if checkpoint is not None:
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_profit = checkpoint.get("best_profit", -float("inf"))
            self.epoch_history = checkpoint.get("history", [])

            # Restore model weights to all workers
            for w in self.workers:
                w["agent"].actor_critic.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    try:
                        w["agent"].optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    except Exception:
                        pass

            logger.info(f"Resumed successfully! Next training epoch: {self.start_epoch}")
        else:
            logger.info("No checkpoint found to resume. Starting fresh from Epoch 1.")

    def train(self, total_epochs: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Executes the main training loop across all epochs and parallel workers.
        """
        if total_epochs is None:
            total_epochs = self.config.total_epochs

        print("=" * 80)
        print("🚀 STARTING CRYPTO RL PARALLEL TRAINING")
        print(f"• Total Epochs:        {total_epochs} (Starting at {self.start_epoch})")
        print(f"• Parallel Workers:    {self.num_workers}")
        print(f"• Available Devices:   {[w['device'] for w in self.workers]}")
        print(f"• Initial Balance:     ${self.config.initial_balance:,.2f}")
        print(f"• Episode Duration:    1 Trading Day (1,440 1-Minute Steps)")
        print(f"• Assets Traded:       {self.config.symbols}")
        print(f"• Qwen 8B Integration: Active (CPU Parallel News Engine)")
        print(f"• Plot Output Dir:     {self.config.plots_dir}")
        print(f"• Checkpoint Dir:      {self.config.checkpoints_dir}")
        print("=" * 80)

        for epoch in range(self.start_epoch, total_epochs + 1):
            epoch_start_time = time.time()
            epoch_worker_results = []

            # Run parallel worker rollouts
            if self.num_workers > 1:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [
                        executor.submit(
                            run_single_worker_rollout,
                            w["worker_id"],
                            epoch,
                            w["agent"],
                            w["env"],
                            self.visualizer,
                            self.config,
                        )
                        for w in self.workers
                    ]
                    for f in futures:
                        epoch_worker_results.append(f.result())
            else:
                w = self.workers[0]
                res = run_single_worker_rollout(
                    w["worker_id"],
                    epoch,
                    w["agent"],
                    w["env"],
                    self.visualizer,
                    self.config,
                )
                epoch_worker_results.append(res)

            # Sync weights from Worker 0 to other workers if multi-GPU
            if self.num_workers > 1:
                master_state = self.workers[0]["agent"].actor_critic.state_dict()
                for i in range(1, self.num_workers):
                    self.workers[i]["agent"].actor_critic.load_state_dict(master_state)

            # Aggregate epoch metrics
            avg_profit = np.mean([r["net_profit"] for r in epoch_worker_results])
            avg_return = np.mean([r["pnl_pct"] for r in epoch_worker_results])
            avg_win_rate = np.mean([r["win_rate"] for r in epoch_worker_results]) * 100.0
            total_trades_all = sum([r["total_trades"] for r in epoch_worker_results])
            epoch_duration = time.time() - epoch_start_time

            for r in epoch_worker_results:
                self.epoch_history.append(r)

            if avg_profit > self.best_profit:
                self.best_profit = avg_profit

            # Save checkpoint after EVERY epoch
            master_worker = self.workers[0]
            self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model_state_dict=master_worker["agent"].actor_critic.state_dict(),
                optimizer_state_dict=master_worker["agent"].optimizer.state_dict(),
                history=self.epoch_history,
                best_profit=self.best_profit,
                config_dict=self.config.__dict__,
            )

            # Save global summary chart
            self.visualizer.save_summary_dashboard(self.epoch_history)

            # Console Progress Print
            assets_str = ", ".join([f"{r['symbol']}(${r['net_profit']:+.1f})" for r in epoch_worker_results])
            profit_symbol = "🟢" if avg_profit >= 0 else "🔴"
            print(
                f"[{epoch:04d}/{total_epochs:04d}] {profit_symbol} "
                f"Avg PnL: ${avg_profit:+7.2f} ({avg_return:+6.2f}%) | "
                f"Assets: [{assets_str}] | "
                f"Trades: {total_trades_all:2d} (Win: {avg_win_rate:4.1f}%) | "
                f"Saved {len(epoch_worker_results)} PNGs | "
                f"Time: {epoch_duration:.2f}s"
            )

        print("=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"• Total Epochs Completed: {total_epochs}")
        print(f"• Best Average Net Profit: ${self.best_profit:+,.2f}")
        print(f"• All epoch plots saved in: {self.config.plots_dir}")
        print(f"• Checkpoints saved in:   {self.config.checkpoints_dir}")
        print("=" * 80)

        return self.epoch_history
