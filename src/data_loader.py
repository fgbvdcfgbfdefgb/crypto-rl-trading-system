"""
Data loader module for loading, preprocessing, and slicing 1-minute cryptocurrency data.
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class CryptoDataLoader:
    """
    Loads 1-minute crypto datasets for BTC, ETH, and LTC.
    Provides normalized feature arrays and day-length (1440 minutes) episode slices.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.data: Dict[str, pd.DataFrame] = {}
        self.features: Dict[str, np.ndarray] = {}
        self.prices: Dict[str, np.ndarray] = {}
        self.timestamps: Dict[str, pd.Series] = {}
        self.day_indices: Dict[str, List[Tuple[int, int, str]]] = {}
        self._load_all_data()

    def _load_all_data(self):
        csv_files = glob.glob(os.path.join(self.data_dir, "*_USDT_1m.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No *_USDT_1m.csv files found in data directory: {self.data_dir}")

        for filepath in sorted(csv_files):
            symbol = os.path.basename(filepath).split("_")[0]  # 'BTC', 'ETH', 'LTC'
            df = pd.read_csv(filepath)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Engineer additional normalized market features
            df = self._add_features(df)

            self.data[symbol] = df
            self.prices[symbol] = df["close"].values.astype(np.float32)
            self.timestamps[symbol] = df["timestamp"]

            # Select 18 key normalized market features
            feature_cols = [
                "returns",
                "ema_9_ratio",
                "ema_21_ratio",
                "sma_50_ratio",
                "rsi_norm",
                "macd",
                "macd_signal",
                "macd_hist",
                "bollinger_pct_norm",
                "bollinger_width",
                "atr_norm",
                "volume_zscore",
                "volatility_15m",
                "high_low_spread",
                "close_open_spread",
                "time_sin",
                "time_cos",
                "momentum_5m",
            ]
            self.features[symbol] = df[feature_cols].values.astype(np.float32)

            # Build 1440-minute day slices
            n_rows = len(df)
            day_slices = []
            episode_len = 1440
            for start in range(0, n_rows - episode_len + 1, episode_len):
                end = start + episode_len
                date_str = df["timestamp"].iloc[start].strftime("%Y-%m-%d")
                day_slices.append((start, end, date_str))

            self.day_indices[symbol] = day_slices

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Bollinger percentage normalized to [-1, 1]
        if "bollinger_pct" in df.columns:
            df["bollinger_pct_norm"] = np.clip((df["bollinger_pct"] - 0.5) * 2.0, -1.0, 1.0)
        else:
            df["bollinger_pct_norm"] = 0.0

        # High-low spread and Close-Open spread
        df["high_low_spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)
        df["close_open_spread"] = (df["close"] - df["open"]) / (df["open"] + 1e-8)

        # Intraday cyclical time encoding (sin & cos of minute 0..1439)
        minutes = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
        df["time_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
        df["time_cos"] = np.cos(2 * np.pi * minutes / 1440.0)

        # 5-minute price momentum
        df["momentum_5m"] = df["close"].pct_change(5).fillna(0.0)

        # Clean any remaining NaNs or Infs
        df = df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        return df

    def get_symbols(self) -> List[str]:
        return list(self.data.keys())

    def get_num_days(self, symbol: str) -> int:
        return len(self.day_indices.get(symbol, []))

    def get_episode_data(
        self,
        symbol: Optional[str] = None,
        day_idx: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> Tuple[str, int, str, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Retrieves a 1440-minute day episode.
        If symbol or day_idx is not provided, samples one randomly.
        """
        rng = random_state if random_state is not None else np.random

        if symbol is None or symbol not in self.data:
            symbol = rng.choice(self.get_symbols())

        day_slices = self.day_indices[symbol]
        if not day_slices:
            raise ValueError(f"No day slices available for symbol {symbol}")

        if day_idx is None or day_idx < 0 or day_idx >= len(day_slices):
            day_idx = rng.randint(0, len(day_slices))

        start_idx, end_idx, date_str = day_slices[day_idx]
        feat_slice = self.features[symbol][start_idx:end_idx]
        price_slice = self.prices[symbol][start_idx:end_idx]
        df_slice = self.data[symbol].iloc[start_idx:end_idx].copy().reset_index(drop=True)

        return symbol, day_idx, date_str, feat_slice, price_slice, df_slice
