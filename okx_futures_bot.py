import os
import time
from datetime import datetime

import ccxt
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv


class OkxFuturesBot:
    """Simple futures trading bot for OKX demo trading.

    - Uses 30m timeframe and three indicators (EMA crossover, RSI, MACD).
    - Opens a position when at least two indicators agree.
    - Position size: 90% of free USDT balance with 10x leverage.
    - Only one position can be open at a time.
    - Sends Telegram messages on start, every trade action and hourly PnL report.
    """

    def __init__(self, symbol: str = "BTC/USDT:USDT") -> None:
        load_dotenv()
        self.api_key = os.getenv("OKX_API_KEY")
        self.api_secret = os.getenv("OKX_API_SECRET")
        self.passphrase = os.getenv("OKX_API_PASSWORD")
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

        self.exchange = ccxt.okx({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "password": self.passphrase,
            "enableRateLimit": True,
        })
        self.exchange.set_sandbox_mode(True)  # demo environment

        self.symbol = symbol
        self.timeframe = "30m"
        self.leverage = 10
        self.capital_pct = 0.9
        self.position = None  # type: ignore[assignment]
        self.hourly_pnl = 0.0
        self.total_pnl = 0.0
        self.last_report = time.time()

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------
    def send_telegram(self, message: str) -> None:
        if not (self.telegram_token and self.telegram_chat_id):
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        data = {"chat_id": self.telegram_chat_id, "text": message}
        try:
            requests.post(url, data=data, timeout=10)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Market data / indicators
    # ------------------------------------------------------------------
    def fetch_ohlcv(self, limit: int = 120) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ema_fast"] = df["close"].ewm(span=9).mean()
        df["ema_slow"] = df["close"].ewm(span=21).mean()
        df["ema_signal"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)

        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(14).mean()
        avg_loss = down.rolling(14).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_signal"] = np.where(df["rsi"] < 30, 1, np.where(df["rsi"] > 70, -1, 0))

        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal_line"] = df["macd"].ewm(span=9).mean()
        df["macd_signal"] = np.where(df["macd"] > df["macd_signal_line"], 1, -1)
        return df

    def get_signal(self):
        df = self.compute_indicators(self.fetch_ohlcv())
        last = df.iloc[-1]
        signals = {
            "ema": int(last["ema_signal"]),
            "rsi": int(last["rsi_signal"]),
            "macd": int(last["macd_signal"]),
        }
        long_votes = sum(1 for v in signals.values() if v == 1)
        short_votes = sum(1 for v in signals.values() if v == -1)
        side = None
        if long_votes >= 2:
            side = "long"
        elif short_votes >= 2:
            side = "short"
        return side, last

    # ------------------------------------------------------------------
    # Trading helpers
    # ------------------------------------------------------------------
    def position_size(self) -> float:
        balance = self.exchange.fetch_balance({"type": "swap"})
        free = balance["USDT"]["free"]
        notional = free * self.capital_pct * self.leverage
        price = self.exchange.fetch_ticker(self.symbol)["last"]
        amount = notional / price
        return amount

    def open_position(self, side: str, price: float) -> None:
        amount = self.position_size()
        order_side = "buy" if side == "long" else "sell"
        params = {"tdMode": "cross", "leverage": str(self.leverage)}
        self.exchange.create_market_order(self.symbol, order_side, amount, params=params)
        self.position = {"side": side, "amount": amount, "entry_price": price}
        self.send_telegram(f"Opened {side} at {price:.2f} amount {amount:.4f}")

    def close_position(self, price: float) -> None:
        side = self.position["side"]
        amount = self.position["amount"]
        order_side = "sell" if side == "long" else "buy"
        self.exchange.create_market_order(self.symbol, order_side, amount, params={"tdMode": "cross"})
        pnl = (price - self.position["entry_price"]) * amount
        if side == "short":
            pnl = -pnl
        self.hourly_pnl += pnl
        self.total_pnl += pnl
        self.send_telegram(f"Closed {side} at {price:.2f} PnL {pnl:.2f} USDT")
        self.position = None

    def maybe_close(self, last_row: pd.Series) -> None:
        if not self.position:
            return
        entry = self.position["entry_price"]
        atr = last_row["high"] - last_row["low"]  # simplified ATR
        if self.position["side"] == "long":
            tp = entry + 2 * atr
            sl = entry - atr
            if last_row["close"] >= tp or last_row["close"] <= sl:
                self.close_position(last_row["close"])
        else:
            tp = entry - 2 * atr
            sl = entry + atr
            if last_row["close"] <= tp or last_row["close"] >= sl:
                self.close_position(last_row["close"])

    def hourly_report(self) -> None:
        if time.time() - self.last_report >= 3600:
            msg = f"Hourly PnL: {self.hourly_pnl:.2f} USDT | Total: {self.total_pnl:.2f} USDT"
            self.send_telegram(msg)
            self.hourly_pnl = 0.0
            self.last_report = time.time()

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.send_telegram("Bot started")
        while True:
            try:
                side, last = self.get_signal()
                if self.position is None and side:
                    self.open_position(side, float(last["close"]))
                self.maybe_close(last)
                self.hourly_report()
            except Exception as e:  # pragma: no cover - for runtime safety
                self.send_telegram(f"Error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    bot = OkxFuturesBot()
    bot.run()
