"""
Visualizer module for generating comprehensive, publication-quality PNG charts
after EVERY epoch for every parallel running worker.
"""

import os
import textwrap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server execution
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class TradingVisualizer:
    """
    Renders and saves high-resolution multi-panel PNG charts showing:
    1. Asset price curve with buy/sell execution markers
    2. Portfolio net worth & cash/crypto allocation vs $2,000 benchmark
    3. Technical indicators (RSI & MACD) & Qwen 8B daily news sentiment
    4. Action distribution, performance metrics, and news headline banner
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Style configuration
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    def plot_epoch(
        self,
        epoch: int,
        worker_id: int,
        history: Dict[str, Any],
        info: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generates and saves the per-epoch, per-worker PNG plot.
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"epoch_{epoch:04d}_worker_{worker_id}.png")

        steps = np.array(history["step"])
        prices = np.array(history["price"])
        actions = np.array(history["action"])
        portfolio_values = np.array(history["portfolio_value"])
        benchmarks = np.array(history["benchmark_value"])

        symbol = info.get("symbol", "CRYPTO")
        date_str = info.get("date", "Unknown Date")
        net_profit = info.get("net_profit", 0.0)
        pnl_pct = info.get("pnl_pct", 0.0)
        bench_pnl_pct = info.get("bench_pnl_pct", 0.0)
        total_trades = info.get("total_trades", 0)
        win_rate = info.get("win_rate", 0.0) * 100.0
        qwen_sentiment = info.get("qwen_sentiment", 0.0)
        qwen_impact = info.get("qwen_impact", 0.0)
        qwen_headline = info.get("qwen_headline", "No major news recorded.")

        # Create Figure with GridSpec
        fig = plt.figure(figsize=(20, 13), dpi=120)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.1, 0.9], hspace=0.28, wspace=0.22)

        # -------------------------------------------------------------
        # PANEL 1: Price Action & Agent Executions (Top-Left)
        # -------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(steps, prices, color="#1565C0", label=f"{symbol}/USDT Price", linewidth=1.5)

        # Calculate moving averages for price panel
        if len(prices) >= 21:
            ema9 = pd.Series(prices).ewm(span=9, adjust=False).mean().values
            ema21 = pd.Series(prices).ewm(span=21, adjust=False).mean().values
            ax1.plot(steps, ema9, color="#FFB300", linestyle="--", alpha=0.85, label="9-EMA", linewidth=1.1)
            ax1.plot(steps, ema21, color="#FB8C00", linestyle=":", alpha=0.85, label="21-EMA", linewidth=1.1)

        # Mark Buys and Sells
        buy_indices = [t["step"] for t in history.get("trades", []) if t["action"] == "BUY"]
        buy_prices = [t["price"] for t in history.get("trades", []) if t["action"] == "BUY"]
        sell_indices = [t["step"] for t in history.get("trades", []) if t["action"] == "SELL"]
        sell_prices = [t["price"] for t in history.get("trades", []) if t["action"] == "SELL"]

        if buy_indices:
            ax1.scatter(buy_indices, buy_prices, marker="^", color="#00C853", s=80, label=f"BUY ({len(buy_indices)})", zorder=5, edgecolor="black", linewidth=0.5)
        if sell_indices:
            ax1.scatter(sell_indices, sell_prices, marker="v", color="#D50000", s=80, label=f"SELL ({len(sell_indices)})", zorder=5, edgecolor="black", linewidth=0.5)

        ax1.set_title(f"1-Minute Price Chart & RL Execution Points ({symbol}/USDT)", fontsize=12, fontweight="bold", pad=8)
        ax1.set_ylabel("Price (USDT)", fontsize=10)
        ax1.set_xlabel("Minute of Day (0 to 1440)", fontsize=10)
        ax1.legend(loc="upper left", frameon=True, fontsize=8.5)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 2: Portfolio Net Worth vs Benchmark (Top-Right)
        # -------------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(steps, portfolio_values, color="#6200EA", linewidth=2.0, label="RL Agent Net Worth ($)")
        ax2.plot(steps, benchmarks, color="#FF6D00", linestyle="--", linewidth=1.5, alpha=0.85, label="Buy & Hold Benchmark ($)")
        ax2.axhline(2000.0, color="#757575", linestyle=":", alpha=0.7, label="Initial Capital ($2,000)")

        # Fill between initial balance and current balance
        ax2.fill_between(
            steps, portfolio_values, 2000.0,
            where=(portfolio_values >= 2000.0),
            color="#00C853", alpha=0.15, interpolate=True
        )
        ax2.fill_between(
            steps, portfolio_values, 2000.0,
            where=(portfolio_values < 2000.0),
            color="#D50000", alpha=0.15, interpolate=True
        )

        final_val = portfolio_values[-1] if len(portfolio_values) > 0 else 2000.0
        profit_color = "#00C853" if net_profit >= 0 else "#D50000"
        ax2.text(
            0.03, 0.92,
            f"Net Worth: ${final_val:,.2f}  |  Profit: ${net_profit:+,.2f} ({pnl_pct:+.2f}%)",
            transform=ax2.transAxes,
            fontsize=10,
            fontweight="bold",
            color=profit_color,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=profit_color, alpha=0.9),
        )

        ax2.set_title("Portfolio Growth vs Buy & Hold Benchmark ($2,000 Initial)", fontsize=12, fontweight="bold", pad=8)
        ax2.set_ylabel("Portfolio Value ($ USD)", fontsize=10)
        ax2.set_xlabel("Minute of Day (0 to 1440)", fontsize=10)
        ax2.legend(loc="lower left", frameon=True, fontsize=8.5)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 3: Technical Indicators: RSI & MACD Sub-Plots (Bottom-Left)
        # -------------------------------------------------------------
        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=0.35)
        ax3_1 = fig.add_subplot(gs_sub[0, 0])
        ax3_2 = fig.add_subplot(gs_sub[1, 0])

        # RSI Calculation
        if len(prices) > 14:
            diff = np.diff(prices, prepend=prices[0])
            gain = np.where(diff > 0, diff, 0.0)
            loss = np.where(diff < 0, -diff, 0.0)
            avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14).mean().values
            avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14).mean().values
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi = np.nan_to_num(rsi, nan=50.0)
        else:
            rsi = np.full_like(steps, 50.0)

        ax3_1.plot(steps, rsi, color="#7E57C2", linewidth=1.2, label="RSI (14)")
        ax3_1.axhline(70, color="#D32F2F", linestyle="--", alpha=0.7, label="Overbought (70)")
        ax3_1.axhline(30, color="#388E3C", linestyle="--", alpha=0.7, label="Oversold (30)")
        ax3_1.set_ylabel("RSI", fontsize=9)
        ax3_1.set_ylim(10, 90)
        ax3_1.legend(loc="upper left", fontsize=7.5)
        ax3_1.set_title("Relative Strength Index (RSI 14)", fontsize=10, fontweight="bold")
        ax3_1.grid(True, linestyle="--", alpha=0.4)

        # MACD Calculation
        if len(prices) >= 26:
            ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean()
            ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean()
            macd_line = (ema12 - ema26).values
            signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
            macd_hist = macd_line - signal_line
            hist_colors = ["#00C853" if h >= 0 else "#D50000" for h in macd_hist]
            ax3_2.bar(steps, macd_hist, color=hist_colors, width=1.0, alpha=0.6, label="MACD Hist")
            ax3_2.plot(steps, macd_line, color="#1E88E5", linewidth=1.0, label="MACD")
            ax3_2.plot(steps, signal_line, color="#FF8F00", linewidth=1.0, linestyle="--", label="Signal")
        ax3_2.set_ylabel("MACD", fontsize=9)
        ax3_2.set_xlabel("Minute of Day (0 to 1440)", fontsize=9)
        ax3_2.legend(loc="upper left", fontsize=7.5)
        ax3_2.set_title("MACD (12, 26, 9)", fontsize=10, fontweight="bold")
        ax3_2.grid(True, linestyle="--", alpha=0.4)

        # -------------------------------------------------------------
        # PANEL 4: Action Distribution & Qwen 8B Analysis Banner (Bottom-Right)
        # -------------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

        # Action distribution calculations
        n_total = len(actions) if len(actions) > 0 else 1
        n_hold = int(np.sum(actions == 0))
        n_buy = int(np.sum(actions == 1))
        n_sell = int(np.sum(actions == 2))
        pct_hold = (n_hold / n_total) * 100.0
        pct_buy = (n_buy / n_total) * 100.0
        pct_sell = (n_sell / n_total) * 100.0

        # Sub-panel for Action Distribution Bar Chart inside ax4
        gs_sub_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], height_ratios=[0.35, 0.65], hspace=0.35)
        ax4_bars = fig.add_subplot(gs_sub_right[0, 0])
        
        bars = ax4_bars.barh(["SELL", "BUY", "HOLD"], [pct_sell, pct_buy, pct_hold], color=["#D50000", "#00C853", "#78909C"], height=0.55)
        ax4_bars.set_xlim(0, 100)
        ax4_bars.set_xlabel("Action Frequency (%)", fontsize=9)
        ax4_bars.set_title("Agent Action Distribution (1440 Steps)", fontsize=10, fontweight="bold")
        for bar, count in zip(bars, [n_sell, n_buy, n_hold]):
            width = bar.get_width()
            ax4_bars.text(width + 1.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}% ({count})", va="center", fontsize=8.5, fontweight="bold")
        ax4_bars.grid(True, linestyle="--", alpha=0.4)

        # Performance & Qwen Text Card inside bottom half of ax4
        ax4_text = fig.add_subplot(gs_sub_right[1, 0])
        ax4_text.axis("off")

        wrapped_headline = "\n  ".join(textwrap.wrap(f'"{qwen_headline}"', width=68))
        sentiment_label = "BULLISH" if qwen_sentiment > 0.1 else ("BEARISH" if qwen_sentiment < -0.1 else "NEUTRAL")

        summary_box = (
            f"━━━━━━━━━━━━━━━━━━ SESSION SUMMARY ━━━━━━━━━━━━━━━━━━\n"
            f"• Epoch / Worker:      Epoch {epoch:04d}  |  Worker #{worker_id}\n"
            f"• Asset & Date:        {symbol}/USDT  |  {date_str}\n"
            f"• Starting Capital:    $2,000.00\n"
            f"• Ending Net Worth:    ${final_val:,.2f}\n"
            f"• Profit / Return:     ${net_profit:+,.2f} ({pnl_pct:+.2f}%)  [Bench: {bench_pnl_pct:+.2f}%]\n"
            f"• Orders Executed:     {total_trades} trades  (Win Rate: {win_rate:.1f}%)\n"
            f"━━━━━━━━━━━━━━━ QWEN 8B AI NEWS ENGINE ━━━━━━━━━━━━━━━\n"
            f"• Sentiment Signal:    {qwen_sentiment:+.2f} ({sentiment_label})\n"
            f"• Market Impact Score: {qwen_impact:.2f} / 1.00\n"
            f"• Analyzed News:\n  {wrapped_headline}"
        )

        ax4_text.text(
            0.02, 0.95,
            summary_box,
            fontsize=8.8,
            fontfamily="monospace",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ECEFF1", edgecolor="#90A4AE", alpha=0.95),
        )

        # Main Title
        plt.suptitle(
            f"Epoch {epoch:04d} | Agent Worker #{worker_id} | {symbol}/USDT Trading Session ({date_str})",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)

        return save_path

    def save_summary_dashboard(
        self,
        epoch_history: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generates global training summary dashboard across all epochs.
        """
        if not epoch_history:
            return ""

        if save_path is None:
            save_path = os.path.join(self.output_dir, "training_summary.png")

        df = pd.DataFrame(epoch_history)
        epochs = df["epoch"].values
        profits = df["net_profit"].values
        cum_returns = df["pnl_pct"].values

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10), dpi=120)

        # 1. PnL per Epoch
        bar_colors = ["#00C853" if p >= 0 else "#D50000" for p in profits]
        ax1.bar(epochs, profits, color=bar_colors, width=0.8)
        ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax1.set_title("Net Profit ($) per Epoch", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Profit / Loss ($)")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Cumulative Return % Curve
        ax2.plot(epochs, np.cumsum(cum_returns), color="#6200EA", linewidth=2.0, marker="o", markersize=4)
        ax2.set_title("Cumulative PnL (%) Trajectory", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Cumulative Return (%)")
        ax2.grid(True, linestyle="--", alpha=0.5)

        # 3. Profit by Asset Class
        if "symbol" in df.columns:
            asset_grp = df.groupby("symbol")["net_profit"].mean()
            ax3.bar(asset_grp.index, asset_grp.values, color=["#F7931A", "#627EEA", "#345D9D"])
            ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax3.set_title("Average Net Profit by Crypto Asset ($)", fontsize=12, fontweight="bold")
            ax3.set_ylabel("Avg Profit ($)")
            ax3.grid(True, linestyle="--", alpha=0.5)

        # 4. Win Rate / Policy Stats
        if "win_rate" in df.columns:
            ax4.plot(epochs, df["win_rate"] * 100.0, color="#00897B", linewidth=2.0, marker="s", markersize=4)
            ax4.set_title("Trade Win Rate (%) over Training", fontsize=12, fontweight="bold")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Win Rate (%)")
            ax4.set_ylim(0, 100)
            ax4.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("Crypto RL Multi-Asset Trading System: Overall Training Progress", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)

        return save_path
