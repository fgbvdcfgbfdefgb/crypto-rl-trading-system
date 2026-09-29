"""
Gymnasium-compatible Cryptocurrency Trading Environment.
Simulates realistic 1-minute order execution, fees, slippage, and Qwen 8B news state integration.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from .config import TradingConfig
from .data_loader import CryptoDataLoader
from .qwen_engine import QwenNewsEngine


class CryptoTradingEnv(gym.Env):
    """
    Simulates a 1-day (1440 1-minute steps) cryptocurrency trading session with $2,000 starting cash.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        data_loader: CryptoDataLoader,
        qwen_engine: QwenNewsEngine,
        config: Optional[TradingConfig] = None,
        symbol: Optional[str] = None,
        day_idx: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.config = config or TradingConfig()
        self.data_loader = data_loader
        self.qwen_engine = qwen_engine
        self.fixed_symbol = symbol
        self.fixed_day_idx = day_idx

        # Define Action Space: 0: HOLD, 1: BUY (100% Cash), 2: SELL (100% Crypto)
        self.action_space = spaces.Discrete(self.config.action_dim)

        # Define Observation Space: 31 Continuous features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.state_dim,),
            dtype=np.float32,
        )

        self.rng = np.random.RandomState(seed)
        
        # Episode internal variables
        self.current_step = 0
        self.symbol = "BTC"
        self.day_idx = 0
        self.date_str = ""
        self.features: np.ndarray = None
        self.prices: np.ndarray = None
        self.df_slice: pd.DataFrame = None
        self.news_data: Dict[str, Any] = {}

        # Portfolio state
        self.initial_balance = self.config.initial_balance
        self.cash = self.initial_balance
        self.crypto_held = 0.0
        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0

        # Tracking histories for plotting
        self.history = {
            "step": [],
            "timestamp": [],
            "price": [],
            "action": [],
            "portfolio_value": [],
            "cash": [],
            "crypto_held": [],
            "crypto_val": [],
            "reward": [],
            "benchmark_value": [],
            "trades": [],  # List of dicts for executed trades
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Sample asset and day
        target_symbol = self.fixed_symbol
        target_day = self.fixed_day_idx
        if options and "symbol" in options:
            target_symbol = options["symbol"]
        if options and "day_idx" in options:
            target_day = options["day_idx"]

        (
            self.symbol,
            self.day_idx,
            self.date_str,
            self.features,
            self.prices,
            self.df_slice,
        ) = self.data_loader.get_episode_data(
            symbol=target_symbol,
            day_idx=target_day,
            random_state=self.rng,
        )

        # Retrieve Qwen 8B market news analysis for this day
        self.news_data = self.qwen_engine.get_news_signal(self.symbol, self.date_str)

        # Reset portfolio
        self.current_step = 0
        self.cash = float(self.initial_balance)
        self.crypto_held = 0.0
        self.portfolio_value = float(self.initial_balance)
        self.prev_portfolio_value = float(self.initial_balance)
        self.peak_portfolio_value = float(self.initial_balance)
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0

        # Benchmark: buy-and-hold starting at initial balance
        self.initial_price = float(self.prices[0])
        self.benchmark_units = self.initial_balance / (self.initial_price * (1.0 + self.config.taker_fee))

        # Clear histories
        self.history = {
            "step": [],
            "timestamp": [],
            "price": [],
            "action": [],
            "portfolio_value": [],
            "cash": [],
            "crypto_held": [],
            "crypto_val": [],
            "reward": [],
            "benchmark_value": [],
            "trades": [],
        }

        # Record initial state
        self._record_history(action=0, reward=0.0)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _get_observation(self) -> np.ndarray:
        # 1. Market Features (18 dimensions)
        step_idx = min(self.current_step, len(self.features) - 1)
        mkt_feat = self.features[step_idx]

        # 2. Account Features (5 dimensions)
        current_price = float(self.prices[step_idx])
        pos_val = self.crypto_held * current_price
        port_val = self.cash + pos_val
        
        unrealized_pnl = 0.0
        if self.crypto_held > 0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) * self.crypto_held

        acct_feat = np.array([
            self.cash / self.initial_balance,
            pos_val / self.initial_balance,
            port_val / self.initial_balance,
            unrealized_pnl / self.initial_balance,
            pos_val / (port_val + 1e-8),  # Position allocation ratio [0.0 to 1.0]
        ], dtype=np.float32)

        # 3. Qwen 8B News Vector (8 dimensions)
        qwen_feat = self.news_data["embedding"]

        # Concatenate 18 + 5 + 8 = 31 dimensions
        obs = np.concatenate([mkt_feat, acct_feat, qwen_feat]).astype(np.float32)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = float(self.prices[self.current_step])
        step_action = int(action)
        fee_paid = 0.0
        trade_occurred = False

        # Execute Action
        if step_action == 1:  # BUY
            # Allocate all available cash into crypto
            if self.cash > 10.0:  # Minimum order size $10
                buy_price = current_price * (1.0 + self.config.slippage)
                fee_rate = self.config.taker_fee
                cash_for_crypto = self.cash / (1.0 + fee_rate)
                amount_bought = cash_for_crypto / buy_price
                fee_paid = self.cash - cash_for_crypto

                self.crypto_held += amount_bought
                self.cash = 0.0
                self.entry_price = buy_price
                self.total_trades += 1
                trade_occurred = True

                self.history["trades"].append({
                    "step": self.current_step,
                    "action": "BUY",
                    "price": buy_price,
                    "amount": amount_bought,
                    "fee": fee_paid,
                    "portfolio_value": self.portfolio_value,
                })

        elif step_action == 2:  # SELL
            # Liquidate all crypto holdings to cash
            if self.crypto_held > 1e-6:
                sell_price = current_price * (1.0 - self.config.slippage)
                gross_revenue = self.crypto_held * sell_price
                fee_paid = gross_revenue * self.config.taker_fee
                net_revenue = gross_revenue - fee_paid

                if sell_price > self.entry_price:
                    self.winning_trades += 1

                self.cash += net_revenue
                self.total_trades += 1
                trade_occurred = True

                self.history["trades"].append({
                    "step": self.current_step,
                    "action": "SELL",
                    "price": sell_price,
                    "amount": self.crypto_held,
                    "fee": fee_paid,
                    "portfolio_value": self.portfolio_value,
                })

                self.crypto_held = 0.0
                self.entry_price = 0.0

        # Advance step
        self.current_step += 1
        terminated = bool(self.current_step >= len(self.prices) - 1 or self.current_step >= self.config.episode_steps)
        truncated = False

        # Update portfolio net worth
        new_step_idx = min(self.current_step, len(self.prices) - 1)
        new_price = float(self.prices[new_step_idx])
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_value = self.cash + (self.crypto_held * new_price)

        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value

        # Calculate Reward
        # 1. Log return of portfolio value
        pnl_return = (self.portfolio_value - self.prev_portfolio_value) / (self.prev_portfolio_value + 1e-8)
        reward = pnl_return * self.config.pnl_reward_scale

        # 2. Transaction fee penalty
        if trade_occurred:
            reward -= self.config.churn_penalty_scale

        # 3. Drawdown penalty
        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / (self.peak_portfolio_value + 1e-8)
        reward -= current_drawdown * self.config.drawdown_penalty_scale

        # 4. Terminal reward bonus
        if terminated:
            total_return_pct = (self.portfolio_value - self.initial_balance) / self.initial_balance
            benchmark_final = self.benchmark_units * new_price
            bench_return_pct = (benchmark_final - self.initial_balance) / self.initial_balance
            excess_return = total_return_pct - bench_return_pct
            reward += float(excess_return * 10.0)

        # Record history
        self._record_history(action=step_action, reward=reward)

        obs = self._get_observation()
        info = self._get_info()
        return obs, float(reward), terminated, truncated, info

    def _record_history(self, action: int, reward: float):
        step_idx = min(self.current_step, len(self.prices) - 1)
        current_price = float(self.prices[step_idx])
        ts = self.df_slice["timestamp"].iloc[step_idx] if self.df_slice is not None else self.current_step
        crypto_val = self.crypto_held * current_price
        bench_val = self.benchmark_units * current_price

        self.history["step"].append(self.current_step)
        self.history["timestamp"].append(ts)
        self.history["price"].append(current_price)
        self.history["action"].append(action)
        self.history["portfolio_value"].append(self.portfolio_value)
        self.history["cash"].append(self.cash)
        self.history["crypto_held"].append(self.crypto_held)
        self.history["crypto_val"].append(crypto_val)
        self.history["reward"].append(reward)
        self.history["benchmark_value"].append(bench_val)

    def _get_info(self) -> Dict[str, Any]:
        step_idx = min(self.current_step, len(self.prices) - 1)
        current_price = float(self.prices[step_idx])
        bench_val = self.benchmark_units * current_price
        net_profit = self.portfolio_value - self.initial_balance
        pnl_pct = (net_profit / self.initial_balance) * 100.0
        bench_pnl_pct = ((bench_val - self.initial_balance) / self.initial_balance) * 100.0

        win_rate = (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0.0

        return {
            "symbol": self.symbol,
            "day_idx": self.day_idx,
            "date": self.date_str,
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "crypto_held": self.crypto_held,
            "net_profit": net_profit,
            "pnl_pct": pnl_pct,
            "bench_pnl_pct": bench_pnl_pct,
            "total_trades": self.total_trades,
            "win_rate": win_rate,
            "qwen_sentiment": self.news_data.get("sentiment_score", 0.0),
            "qwen_impact": self.news_data.get("impact_score", 0.0),
            "qwen_headline": self.news_data.get("headline", ""),
        }
