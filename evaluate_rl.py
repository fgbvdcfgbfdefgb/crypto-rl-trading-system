#!/usr/bin/env python3
"""
Evaluation and backtesting script for trained Crypto RL Trading Agent.
Evaluates the checkpoint across all assets (BTC, ETH, LTC) and all available days.

Usage:
  python evaluate_rl.py --checkpoint checkpoints/latest_checkpoint.pt
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import TradingConfig
from src.data_loader import CryptoDataLoader
from src.qwen_engine import QwenNewsEngine
from src.crypto_env import CryptoTradingEnv
from src.rl_agent import PPOAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Crypto RL Agent")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest_checkpoint.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--initial_balance", type=float, default=2000.0, help="Simulated initial balance in USD")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Path to models directory")
    parser.add_argument("--output_plot", type=str, default="plots/evaluation_backtest.png", help="Path to save backtest summary PNG")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, args.data_dir)
    models_dir = os.path.join(base_dir, args.models_dir)
    chk_path = os.path.join(base_dir, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    output_plot = os.path.join(base_dir, args.output_plot) if not os.path.isabs(args.output_plot) else args.output_plot

    if not os.path.exists(chk_path):
        print(f"Error: Checkpoint file not found: {chk_path}")
        print("Please train a model first with: python train_rl.py")
        return

    print("=" * 80)
    print("📊 STARTING OFFLINE EVALUATION & BACKTEST")
    print(f"• Checkpoint:       {chk_path}")
    print(f"• Initial Balance:  ${args.initial_balance:,.2f}")
    print(f"• Data Directory:   {data_dir}")
    print("=" * 80)

    config = TradingConfig(initial_balance=args.initial_balance)
    data_loader = CryptoDataLoader(data_dir)
    qwen_engine = QwenNewsEngine(data_dir, models_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        device=device,
    )

    try:
        checkpoint = torch.load(chk_path, map_location=device, weights_only=False)
    except Exception:
        checkpoint = torch.load(chk_path, map_location=device)
    agent.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    agent.actor_critic.eval()
    print(f"Loaded checkpoint from Epoch {checkpoint.get('epoch', 'Unknown')}")

    results = []
    symbols = data_loader.get_symbols()

    for symbol in symbols:
        num_days = data_loader.get_num_days(symbol)
        print(f"\n--- Backtesting {symbol}/USDT ({num_days} full days) ---")

        for day_idx in range(num_days):
            env = CryptoTradingEnv(
                data_loader=data_loader,
                qwen_engine=qwen_engine,
                config=config,
                symbol=symbol,
                day_idx=day_idx,
            )

            obs, info = env.reset()
            done = False

            while not done:
                action, _, _ = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                info = step_info

            net_profit = info.get("net_profit", 0.0)
            pnl_pct = info.get("pnl_pct", 0.0)
            bench_pnl_pct = info.get("bench_pnl_pct", 0.0)
            win_rate = info.get("win_rate", 0.0) * 100.0
            total_trades = info.get("total_trades", 0)

            results.append({
                "symbol": symbol,
                "day_idx": day_idx,
                "date": info.get("date", ""),
                "final_val": info.get("portfolio_value", 2000.0),
                "net_profit": net_profit,
                "pnl_pct": pnl_pct,
                "bench_pnl_pct": bench_pnl_pct,
                "trades": total_trades,
                "win_rate": win_rate,
                "qwen_sentiment": info.get("qwen_sentiment", 0.0),
            })

            profit_sign = "+" if net_profit >= 0 else ""
            print(f"  Day {day_idx:02d} ({info.get('date')}): RL PnL = {profit_sign}${net_profit:6.2f} ({profit_sign}{pnl_pct:5.2f}%) | Bench = {bench_pnl_pct:+5.2f}% | Trades: {total_trades:2d}")

    # Summary Statistics
    df_res = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("📈 OVERALL BACKTEST PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"• Total Evaluation Sessions: {len(df_res)}")
    print(f"• Total Cumulative Profit:   ${df_res['net_profit'].sum():+,.2f}")
    print(f"• Mean Profit per Day:       ${df_res['net_profit'].mean():+,.2f} ({df_res['pnl_pct'].mean():+.2f}%)")
    print(f"• Mean Benchmark per Day:    {df_res['bench_pnl_pct'].mean():+.2f}%")
    print(f"• Profitable Days:           {len(df_res[df_res['net_profit'] > 0])} / {len(df_res)} ({len(df_res[df_res['net_profit'] > 0]) / len(df_res) * 100.0:.1f}%)")
    print(f"• Total Trades Executed:     {df_res['trades'].sum()}")
    print("=" * 80)

    # Plot Backtest Visual
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=120)

    # Asset comparison
    asset_means = df_res.groupby("symbol")[["pnl_pct", "bench_pnl_pct"]].mean()
    x = np.arange(len(asset_means))
    width = 0.35
    ax1.bar(x - width / 2, asset_means["pnl_pct"], width, label="RL Agent Return (%)", color="#6200EA")
    ax1.bar(x + width / 2, asset_means["bench_pnl_pct"], width, label="Buy & Hold Return (%)", color="#FF6D00")
    ax1.set_xticks(x)
    ax1.set_xticklabels(asset_means.index)
    ax1.set_title("Mean Daily Return %: RL Agent vs Buy & Hold", fontweight="bold")
    ax1.set_ylabel("Return (%)")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Cumulative PnL Trajectory
    cum_profit = df_res["net_profit"].cumsum()
    ax2.plot(range(len(df_res)), cum_profit, color="#00C853", linewidth=2.0, marker="o", markersize=3)
    ax2.set_title("Cumulative Backtest Profit Trajectory ($)", fontweight="bold")
    ax2.set_xlabel("Evaluation Session Index")
    ax2.set_ylabel("Cumulative Net Profit ($)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Crypto RL Multi-Asset Trading System: Backtest Evaluation Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_plot, bbox_inches="tight", dpi=120)
    plt.close(fig)

    print(f"Backtest summary plot saved to: {output_plot}")


if __name__ == "__main__":
    main()
