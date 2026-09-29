# 🤖 Reinforcement Learning Crypto Trading System (Multi-Asset & Multi-GPU)
### *Powered by Proximal Policy Optimization (PPO) & Parallel CPU Qwen 8B News Intelligence Engine*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Environment-008080.svg)](https://gymnasium.farama.org/)
[![Qwen 8B](https://img.shields.io/badge/LLM-Qwen2.5--8B-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-8B-Instruct)
[![Multi-GPU](https://img.shields.io/badge/Hardware-Multi--GPU%20%2F%20CPU%20Parallel-green.svg)]()
[![Offline Ready](https://img.shields.io/badge/Environment-100%25%20Offline%20Ready-success.svg)]()

---

## 📌 Executive Summary

This repository contains a **100% offline-ready, multi-GPU parallel Reinforcement Learning trading system** engineered to trade major cryptocurrencies (**Bitcoin, Ethereum, Litecoin**) at 1-minute resolution. 

The architecture pairs an **Actor-Critic PPO Agent** with a **CPU-Parallel Qwen 8B News Engine** that extracts daily macro and fundamental sentiment signals from breaking crypto news, directly conditioning the trading policy.

---

## 🏗️ System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │           1-Minute Crypto Datasets           │
                       │     (BTC/USDT, ETH/USDT, LTC/USDT CSVs)      │
                       └──────────────────────┬───────────────────────┘
                                              │
    ┌─────────────────────────┐               │ 18 Market Features
    │  Daily Market News DB   │               ▼
    │ (daily_crypto_news.json)│       ┌────────────────────────────────┐
    └───────────┬─────────────┘       │       Gymnasium Trading        │
                │                     │          Environment           │
                ▼                     │    • Initial Cash: $2,000      │
    ┌─────────────────────────┐       │    • 1 Epoch = 1 Day (1440m)   │
    │  Qwen 8B News Engine    │──────▶│    • Maker/Taker Fees & Slip   │
    │  (CPU Parallel Threads) │ 8 Dim │    • Real-time PnL & Drawdown  │
    └─────────────────────────┘ Vector└───────────────┬────────────────┘
                                                      │ 31-Dim State
                                                      ▼
                                      ┌────────────────────────────────┐
                                      │      PPO Actor-Critic Agent    │
                                      │  (Multi-GPU / Parallel Worker) │
                                      └───────────────┬────────────────┘
                                                      │
                                   ┌──────────────────┴──────────────────┐
                                   ▼                                     ▼
                      ┌────────────────────────┐           ┌────────────────────────┐
                      │    Per-Epoch Plots     │           │   Model Checkpoints    │
                      │ (plots/epoch_X_W_Y.png)│           │(checkpoints/latest.pt) │
                      └────────────────────────┘           └────────────────────────┘
```

---

## 🌟 Key Features

1. **Multi-Asset Random Simulation**:
   - Randomly selects **BTC/USDT**, **ETH/USDT**, or **LTC/USDT** and a trading day per epoch.
   - 1 Epoch = 1 Full Trading Day = **1,440 continuous 1-minute steps**.
   - Starting Simulated Capital: **$2,000.00 USD**.

2. **Qwen 8B Market News Engine (CPU Parallel)**:
   - Evaluates daily market events, regulatory decisions (SEC/ETF), Federal Reserve rate signals, and network upgrades.
   - Outputs:
     - `sentiment_score`: Normalized sentiment $[-1.0, +1.0]$
     - `impact_score`: Expected market volatility scale $[0.0, 1.0]$
     - `direction_bias`: Directional conviction $\{-1, 0, +1\}$
     - `semantic_embedding`: 8-dimensional institutional/macro vector.
   - Dual-mode architecture: Runs full local `Qwen2.5-8B-Instruct` model weights when present, or instantly uses pre-computed offline neural feature caches without network calls.

3. **Multi-GPU Parallel Training**:
   - Auto-detects all available NVIDIA GPUs (`cuda:0`, `cuda:1`, ...) and spins up parallel agent workers.
   - Graceful fallback to multi-threaded CPU workers when running on CPU-only machines.

4. **Detailed Per-Epoch Visualizations**:
   - After **EVERY epoch**, renders a 4-panel, high-resolution PNG chart for **each running worker**:
     - **Panel 1**: 1-minute asset price curve + 9/21-EMA + Green (BUY) & Red (SELL) trade markers.
     - **Panel 2**: Portfolio net worth curve ($) vs Buy & Hold Benchmark ($) starting at $2,000.
     - **Panel 3**: Technical Indicator subplots (RSI 14 & MACD 12,26,9 with color histogram).
     - **Panel 4**: Action frequency bar chart (HOLD/BUY/SELL) + Session summary card with Qwen 8B news headline.
   - Automatically maintains `plots/training_summary.png` across all epochs.

5. **Resumable Checkpointing**:
   - Saves checkpoint files after every epoch (`checkpoints/latest_checkpoint.pt` and `checkpoints/checkpoint_epoch_XXXX.pt`).
   - Resume training seamlessly from interruptions with `--resume`.

---

## 📂 Repository Structure

```
crypto-rl-trading-system/
├── README.md                           # Documentation & user manual
├── requirements.txt                    # Python package dependencies
├── train_rl.py                         # Main multi-GPU / CPU training script
├── evaluate_rl.py                      # Offline evaluation and backtesting script
├── download_qwen.py                    # Utility to download full Qwen 8B weights
├── src/
│   ├── __init__.py
│   ├── config.py                       # Hyperparameters, trading rules, paths
│   ├── crypto_env.py                   # Gymnasium trading environment ($2000, 1440 mins)
│   ├── qwen_engine.py                  # CPU parallel Qwen 8B news & sentiment engine
│   ├── rl_agent.py                     # Actor-Critic PPO Neural Network (PyTorch)
│   ├── multi_gpu_trainer.py            # Multi-GPU orchestrator & parallel rollouts
│   ├── visualizer.py                   # Per-epoch 4-panel PNG chart generator
│   ├── checkpoint.py                   # Checkpoint serialization & restore manager
│   └── data_loader.py                  # 1-minute dataset parser & feature pipeline
├── data/
│   ├── BTC_USDT_1m.csv                 # 1-minute BTC OHLCV + indicators (20,160 bars)
│   ├── ETH_USDT_1m.csv                 # 1-minute ETH OHLCV + indicators (20,160 bars)
│   ├── LTC_USDT_1m.csv                 # 1-minute LTC OHLCV + indicators (20,160 bars)
│   ├── daily_crypto_news.json          # Daily market news headlines
│   └── qwen_sentiment_cache.json       # Pre-computed Qwen 8B sentiment embeddings
├── models/
│   └── qwen_8b_model_metadata.json     # Qwen 8B specifications
├── checkpoints/                        # Model checkpoints (.pt)
└── plots/                              # Generated per-epoch PNG charts
```

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/fgbvdcfgbfdefgb/crypto-rl-trading-system.git
cd crypto-rl-trading-system

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Training (Multi-GPU / Multi-Worker)

```bash
# Train on all available GPUs/CPUs for 50 epochs (Starting Balance: $2,000)
python train_rl.py --epochs 50 --initial_balance 2000

# Train with 4 parallel worker agents
python train_rl.py --epochs 100 --num_workers 4

# Train on specific GPUs (e.g., GPU 0 and GPU 1)
python train_rl.py --epochs 50 --gpus 0,1
```

### 3. Resuming Training

If your session is stopped or times out, resume immediately from the latest checkpoint:

```bash
python train_rl.py --epochs 100 --resume
```

### 4. Running Offline Evaluation & Backtesting

Run a complete backtest on all 14 days across BTC, ETH, and LTC:

```bash
python evaluate_rl.py --checkpoint checkpoints/latest_checkpoint.pt
```

---

## ❄️ Snowflake Free Trial / Air-Gapped Setup

When running in an isolated environment (such as a Snowflake VM or air-gapped compute container where internet access is restricted after the initial sync):

1. **Sync repository once**:
   Clone or sync the repository into your Snowflake workspace.
2. **Execute directly**:
   ```bash
   python train_rl.py --epochs 30 --num_workers 2
   ```
   All datasets (`BTC_USDT_1m.csv`, `ETH_USDT_1m.csv`, `LTC_USDT_1m.csv`), daily news headlines (`daily_crypto_news.json`), and pre-computed Qwen 8B sentiment features are bundled locally in `data/`, so **zero outbound network calls** are required during training.

---

## ⚙️ CLI Parameter Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--epochs` | `int` | `50` | Total training epochs (1 epoch = 1 trading day = 1,440 mins) |
| `--initial_balance` | `float` | `2000.0` | Initial simulated trading capital in USD |
| `--num_workers` | `int` | `auto` | Number of parallel worker agents |
| `--gpus` | `str` | `auto` | GPU IDs (e.g. `0,1` or `auto` or `cpu`) |
| `--lr` | `float` | `0.0003` | Learning rate for PPO optimizer |
| `--batch_size` | `int` | `64` | Mini-batch size for PPO updates |
| `--mini_epochs` | `int` | `4` | PPO optimization epochs per rollout |
| `--resume` | `flag` | `False` | Resume training from latest checkpoint |
| `--checkpoint` | `str` | `latest` | Specific `.pt` checkpoint file to load |
| `--data_dir` | `str` | `data` | Directory containing CSV datasets |
| `--plots_dir` | `str` | `plots` | Output directory for per-epoch PNG charts |
| `--checkpoints_dir`| `str` | `checkpoints` | Output directory for saved model weights |

---

## 📊 Sample Output Plot

Each epoch generates a rich 4-panel diagnostic graphic saved in `plots/epoch_XXXX_worker_Y.png`:

- **Panel 1**: Price line with moving averages and trade execution triangles (▲ Green = BUY, ▼ Red = SELL).
- **Panel 2**: Portfolio Value ($) curve starting from $2,000 vs Buy & Hold Benchmark.
- **Panel 3**: Technical Oscillators (RSI with 30/70 bands & MACD with color histogram).
- **Panel 4**: Action distribution horizontal bar chart & Qwen 8B news analysis briefing card.

---

## 📄 License

MIT License. Designed and prepared for autonomous algorithmic trading research.
