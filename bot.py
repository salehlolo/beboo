#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified scalper bot using user-provided Triple+3 strategies.
Trades 90% of available USDT balance with 10x leverage on Binance
USDT-M futures. All previous filters removed.
"""

import os
import time
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

import ccxt
import ta
import requests


@dataclass
class Config:
    timeframe: str = "5m"
    lookback: int = 800
    ema_fast: int = 9
    ema_slow: int = 21
    atr_window: int = 14
    rsi_len: int = 14
    bb_len: int = 20
    bb_std: float = 2.0
    vol_ma_len: int = 30
    box_len: int = 20
    sr_lookback: int = 50
    keltner_len: int = 20
    keltner_mult: float = 1.5
    
    # dynamic TP/SL
    use_atr_tp_sl: bool = True
    atr_tp_mult: float = 2.0
    atr_sl_mult: float = 1.2
    fixed_tp_pct: float = 0.01
    fixed_sl_pct: float = 0.005

    # capital management
    capital_pct: float = 0.90
    leverage: float = 10.0

    top_n_symbols: int = 5

    # Telegram
    telegram_enabled: bool = True
    telegram_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")


class Notifier:
    """Simple Telegram notifier. Falls back to print if disabled."""

    def __init__(self, cfg: Config):
        self.enabled = bool(cfg.telegram_enabled and cfg.telegram_token and cfg.telegram_chat_id)
        self.base = (
            f"https://api.telegram.org/bot{cfg.telegram_token}" if self.enabled else None
        )
        self.chat_id = cfg.telegram_chat_id

    def send(self, text: str):
        if not self.enabled:
            print(text)
            return
        try:
            r = requests.post(
                f"{self.base}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True},
                timeout=10,
            )
            if r.status_code != 200:
                print("[WARN] Telegram send failed:", r.text)
        except Exception as e:
            print("[WARN] Telegram exception:", e)


class FuturesExchange:
    def __init__(self, cfg: Config):
        key = os.getenv("BINANCE_API_KEY")
        secret = os.getenv("BINANCE_API_SECRET")
        self.x = ccxt.binanceusdm({
            "apiKey": key,
            "secret": secret,
            "options": {"defaultType": "future"},
            "enableRateLimit": True,
        })
        self.x.load_markets()
        self.cfg = cfg

    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        o = self.x.fetch_ohlcv(symbol, timeframe=self.cfg.timeframe, limit=self.cfg.lookback)
        df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("datetime").drop(columns=["ts"])
        return df.apply(pd.to_numeric, errors="coerce").dropna()

    def get_balance_usdt(self) -> float:
        try:
            bal = self.x.fetch_balance(params={"type":"future"})
            return float(bal["total"].get("USDT", 0.0))
        except Exception:
            return 0.0

    def place_order(self, symbol: str, side: str, amount: float, leverage: float):
        try:
            self.x.set_leverage(leverage, symbol=symbol)
        except Exception:
            pass
        try:
            return self.x.create_order(symbol=symbol, type="market", side=side, amount=amount)
        except Exception as e:
            print("[WARN] order failed:", e)
            return None

    def top_symbols(self, n: int) -> List[str]:
        try:
            tickers = self.x.fetch_tickers()
            swaps = []
            for m in self.x.markets.values():
                if m.get("swap") and m.get("quote") == "USDT":
                    sym = m["symbol"]
                    vol = tickers.get(sym, {}).get("quoteVolume") or 0
                    swaps.append((sym, float(vol)))
            swaps.sort(key=lambda x: x[1], reverse=True)
            return [s for s,_ in swaps[:n]]
        except Exception:
            return ["BTC/USDT", "ETH/USDT"]


def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df.copy()
    d["ema9"] = ta.trend.EMAIndicator(d["close"], window=cfg.ema_fast).ema_indicator()
    d["ema21"] = ta.trend.EMAIndicator(d["close"], window=cfg.ema_slow).ema_indicator()
    d["rsi"] = ta.momentum.RSIIndicator(d["close"], window=cfg.rsi_len).rsi()
    bb = ta.volatility.BollingerBands(d["close"], window=cfg.bb_len, window_dev=cfg.bb_std)
    d["bb_mid"], d["bb_up"], d["bb_dn"] = bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()
    atr = ta.volatility.AverageTrueRange(d["high"], d["low"], d["close"], window=cfg.atr_window)
    d["atr"] = atr.average_true_range()
    d["atr_pct"] = d["atr"] / d["close"]
    session = d.index.tz_convert("UTC").normalize()
    d["vwap_num"] = (d["close"]*d["volume"]).groupby(session).cumsum()
    d["vwap_den"] = d["volume"].groupby(session).cumsum().replace(0, np.nan)
    d["vwap"] = d["vwap_num"]/d["vwap_den"]
    d["vol_ma"] = d["volume"].rolling(cfg.vol_ma_len).mean()
    d["vol_spike"] = d["volume"] > (d["vol_ma"]*1.3)
    d["recent_high"] = d["high"].rolling(cfg.box_len).max()
    d["recent_low"] = d["low"].rolling(cfg.box_len).min()
    d["sr_high"] = d["high"].rolling(cfg.sr_lookback).max()
    d["sr_low"] = d["low"].rolling(cfg.sr_lookback).min()
    d["ema9_slope"] = (d["ema9"]-d["ema9"].shift(3))/d["close"]
    d["ema21_slope"] = (d["ema21"]-d["ema21"].shift(3))/d["close"]
    d["bb_width"] = (d["bb_up"]-d["bb_dn"])/d["bb_mid"]
    adx = ta.trend.ADXIndicator(d["high"], d["low"], d["close"], window=cfg.atr_window)
    d["adx"] = adx.adx()
    d["di_pos"] = adx.adx_pos()
    d["di_neg"] = adx.adx_neg()
    ema = ta.trend.EMAIndicator(d["close"], window=cfg.keltner_len).ema_indicator()
    rng = ta.volatility.AverageTrueRange(d["high"], d["low"], d["close"], window=cfg.keltner_len).average_true_range()
    d["kel_mid"] = ema
    d["kel_up"] = ema + cfg.keltner_mult*rng
    d["kel_dn"] = ema - cfg.keltner_mult*rng
    d = d.replace([np.inf,-np.inf], np.nan).dropna()
    return d


@dataclass
class Regime:
    trend: str
    vol_bucket: str


def classify_regime(row: pd.Series, cfg: Config) -> Regime:
    if row["ema9"] > row["ema21"] and row["close"] > row["vwap"]:
        t = "up"
    elif row["ema9"] < row["ema21"] and row["close"] < row["vwap"]:
        t = "down"
    else:
        t = "neutral"
    p = row.get("atr_pct", np.nan)
    if np.isnan(p) or p < 0.0035:
        v = "low"
    elif p > 0.007:
        v = "high"
    else:
        v = "medium"
    return Regime(t, v)


@dataclass
class Signal:
    side: Optional[str]
    sl: float
    tp: float
    model: str
    reason: str
    confidence: float = 0.5


def make_tp_sl_atr(entry: float, side: str, atr: float, cfg: Config) -> Tuple[float, float]:
    if side == "buy":
        return entry + cfg.atr_tp_mult*atr, entry - cfg.atr_sl_mult*atr
    else:
        return entry - cfg.atr_tp_mult*atr, entry + cfg.atr_sl_mult*atr


def make_tp_sl(entry: float, side: str, cfg: Config) -> Tuple[float, float]:
    if side == "buy":
        return entry*(1+cfg.fixed_tp_pct), entry*(1-cfg.fixed_sl_pct)
    else:
        return entry*(1-cfg.fixed_tp_pct), entry*(1+cfg.fixed_sl_pct)


def get_tp_sl(entry: float, side: str, row: pd.Series, cfg: Config) -> Tuple[float, float]:
    atr_val = float(row.get("atr", 0))
    if cfg.use_atr_tp_sl and atr_val > 0:
        return make_tp_sl_atr(entry, side, atr_val, cfg)
    return make_tp_sl(entry, side, cfg)


# ====== Strategies ======

def sig_trend(row: pd.Series, cfg: Config) -> Optional[Signal]:
    regime = classify_regime(row, cfg)
    if regime.trend == "neutral":
        return None
    up = (row["ema9"]>row["ema21"]) and (row["close"]>row["ema9"]) and (row["ema9_slope"]>0)
    dn = (row["ema9"]<row["ema21"]) and (row["close"]<row["ema9"]) and (row["ema9_slope"]<0)
    if up or dn:
        side = "buy" if up else "sell"
        entry = float(row["close"])
        tp, sl = get_tp_sl(entry, side, row, cfg)
        conf = 0.6
        return Signal(side, sl, tp, "TREND", "EMA crossover", conf)
    return None


def sig_bo(row: pd.Series, cfg: Config) -> Optional[Signal]:
    regime = classify_regime(row, cfg)
    if regime.trend == "neutral":
        return None
    atr = float(row.get("atr", 0))
    rng = row["high"] - row["low"]
    if atr <= 0 or rng <= 0:
        return None
    above = row["close"] >= row["recent_high"]*0.999
    below = row["close"] <= row["recent_low"]*1.001
    if above or below:
        side = "buy" if above else "sell"
        entry = float(row["close"])
        tp, sl = get_tp_sl(entry, side, row, cfg)
        conf = 0.6
        return Signal(side, sl, tp, "BO", "Range breakout", conf)
    return None


def sig_mr(row: pd.Series, cfg: Config) -> Optional[Signal]:
    regime = classify_regime(row, cfg)
    if regime.trend != "neutral":
        return None
    price = float(row["close"])
    is_buy = (row["rsi"] <= 25.0) and (price <= row["bb_dn"])
    is_sell = (row["rsi"] >= 75.0) and (price >= row["bb_up"])
    if is_buy or is_sell:
        side = "buy" if is_buy else "sell"
        tp, sl = get_tp_sl(price, side, row, cfg)
        conf = 0.55
        return Signal(side, sl, tp, "MR", "Mean reversion", conf)
    return None


def sig_pb(row: pd.Series, cfg: Config) -> Optional[Signal]:
    regime = classify_regime(row, cfg)
    if regime.trend == "neutral":
        return None
    wick_up = row["high"]-max(row["close"], row["open"])
    wick_dn = min(row["close"], row["open"])-row["low"]
    body = row["high"]-row["low"]
    if row["ema9"]>row["ema21"]:
        near = row["close"] >= row["ema21"]*(1-0.0035) and row["close"] <= row["ema21"]*(1+0.0035)
        if near and body>0 and (wick_dn/body) >= 0.35:
            price=float(row["close"]); tp,sl=get_tp_sl(price,"buy",row,cfg)
            return Signal("buy", sl, tp, "PB", "Pullback", 0.58)
    if row["ema9"]<row["ema21"]:
        near = row["close"] >= row["ema21"]*(1-0.0035) and row["close"] <= row["ema21"]*(1+0.0035)
        if near and body>0 and (wick_up/body) >= 0.35:
            price=float(row["close"]); tp,sl=get_tp_sl(price,"sell",row,cfg)
            return Signal("sell", sl, tp, "PB", "Pullback", 0.58)
    return None


def sig_vwap_r(row: pd.Series, cfg: Config) -> Optional[Signal]:
    regime = classify_regime(row, cfg)
    if regime.trend != "neutral":
        return None
    dev = abs(row["close"]-row["vwap"])/row["close"]
    if dev < 0.0025:
        return None
    price = float(row["close"])
    if row["close"] < row["vwap"] and row["rsi"] <= 25:
        tp, sl = get_tp_sl(price, "buy", row, cfg)
        return Signal("buy", sl, tp, "VWAP-R", "Below VWAP", 0.56)
    if row["close"] > row["vwap"] and row["rsi"] >= 75:
        tp, sl = get_tp_sl(price, "sell", row, cfg)
        return Signal("sell", sl, tp, "VWAP-R", "Above VWAP", 0.56)
    return None


def sig_ksq(row: pd.Series, cfg: Config) -> Optional[Signal]:
    regime = classify_regime(row, cfg)
    if regime.trend == "neutral":
        return None
    squeeze = row["bb_width"] < (row["atr_pct"]*1.6)
    price = float(row["close"])
    if squeeze and price >= row["kel_up"] and row["di_pos"]>row["di_neg"] and row["ema21_slope"]>0:
        tp, sl = get_tp_sl(price, "buy", row, cfg)
        return Signal("buy", sl, tp, "KSQ", "Squeeze breakout", 0.6)
    if squeeze and price <= row["kel_dn"] and row["di_neg"]>row["di_pos"] and row["ema21_slope"]<0:
        tp, sl = get_tp_sl(price, "sell", row, cfg)
        return Signal("sell", sl, tp, "KSQ", "Squeeze breakdown", 0.6)
    return None


def position_size(equity: float, price: float, cfg: Config) -> float:
    notional = equity * cfg.capital_pct * cfg.leverage
    return notional / price if price > 0 else 0.0


class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ex = FuturesExchange(cfg)
        self.notifier = Notifier(cfg)
        self.symbols = self.ex.top_symbols(cfg.top_n_symbols)
        self.notifier.send(
            f"[START] Simple Scalper | TOP {cfg.top_n_symbols} | TF {cfg.timeframe}"
        )

    def run(self):
        while True:
            for sym in self.symbols:
                try:
                    df = self.ex.fetch_ohlcv(sym)
                    d = compute_indicators(df, self.cfg)
                    row = d.iloc[-2]
                    sig = (sig_trend(row,self.cfg) or sig_bo(row,self.cfg) or sig_mr(row,self.cfg) or
                           sig_pb(row,self.cfg) or sig_vwap_r(row,self.cfg) or sig_ksq(row,self.cfg))
                    if not sig:
                        continue
                    price = float(row["close"])
                    bal = self.ex.get_balance_usdt()
                    qty = position_size(bal, price, self.cfg)
                    order = self.ex.place_order(sym, sig.side, qty, self.cfg.leverage)
                    msg = (
                        f"📢 New Trade\n"
                        f"Pair: {sym}\n"
                        f"Side: {sig.side.upper()} | Qty: {qty:.4f}\n"
                        f"Entry: {price:.4f}\n"
                        f"TP: {sig.tp:.4f} | SL: {sig.sl:.4f}\n"
                        f"Lev: {self.cfg.leverage}x"
                    )
                    self.notifier.send(msg)
                    print(
                        f"{dt.datetime.utcnow()} {sym} {sig.side.upper()} qty={qty:.4f} tp={sig.tp:.4f} sl={sig.sl:.4f}"
                    )
                except Exception as e:
                    err_msg = f"[ERR] {sym} {e}"
                    print(err_msg)
                    self.notifier.send(err_msg)
                time.sleep(1)
            time.sleep(5)


if __name__ == "__main__":
    cfg = Config()
    bot = Bot(cfg)
    bot.run()
