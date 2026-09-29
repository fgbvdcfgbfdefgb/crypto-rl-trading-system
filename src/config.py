"""
Configuration module for Reinforcement Learning Crypto Trading System.
Contains default hyperparameters, environment settings, and data paths.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class TradingConfig:
    # Portfolio & Simulation settings
    initial_balance: float = 2000.0  # Starting cash balance ($2,000 as requested)
    episode_steps: int = 1440  # 1 day = 1440 1-minute steps
    maker_fee: float = 0.0010  # 0.10% maker fee (standard Binance/Coinbase)
    taker_fee: float = 0.0010  # 0.10% taker fee
    slippage: float = 0.0005  # 0.05% slippage estimation
    
    # Supported cryptocurrencies
    symbols: List[str] = field(default_factory=lambda: ["BTC", "ETH", "LTC"])
    
    # State / Observation Dimensions
    # 18 Technical Market Features + 5 Account State Features + 8 Qwen 8B News Features = 31 total
    market_feature_dim: int = 18
    account_feature_dim: int = 5
    qwen_feature_dim: int = 8
    state_dim: int = 31
    
    # Action Space: 0: HOLD, 1: BUY / ALLOCATE CASH, 2: SELL / CLOSE TO CASH
    action_dim: int = 3
    
    # RL Hyperparameters (PPO)
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # Generalized Advantage Estimation lambda
    clip_epsilon: float = 0.20  # PPO surrogate clipping threshold
    entropy_coef: float = 0.015  # Entropy bonus for exploration
    value_loss_coef: float = 0.50  # Value function loss coefficient
    max_grad_norm: float = 0.50  # Gradient clipping norm
    batch_size: int = 64
    mini_epochs: int = 4  # PPO update epochs per rollout
    
    # Training Loop
    total_epochs: int = 50  # Total training epochs
    workers_per_gpu: int = 1  # Parallel workers per GPU
    seed: int = 42
    
    # Reward shaping
    sharpe_reward_scale: float = 1.0
    pnl_reward_scale: float = 100.0
    churn_penalty_scale: float = 0.05
    drawdown_penalty_scale: float = 0.20
    
    # File Paths
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    models_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    checkpoints_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    plots_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    news_file: str = "daily_crypto_news.json"
    sentiment_cache_file: str = "qwen_sentiment_cache.json"
