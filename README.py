#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bot.py — Triple+3 Strategies (Self-Evolving) Scalper — OKX USDT Swap
(نسخة بدون أي تكامل مع OpenAI — تداول/إشعارات فقط)

تشغيل تجريبي على بيئة الديمو الخاصة بـOKX.
Env:
  OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSWORD
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  (اختياري) CRYPTOPANIC_TOKEN, NEWSAPI_KEY  ← تقدر تسيبهم فاضيين
"""

import os, time, json, argparse, datetime as dt, math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import numpy as np  # مطلوب لحسابات بسيطة بالفلتر
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import ccxt
import ta
import requests

# =========================
# Helpers
# =========================

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)

def fmt_ts(ts: Optional[dt.datetime] = None) -> str:
    t = (ts or now_utc())
    s = t.isoformat()
    return s.replace("+00:00", "Z")

def ensure_dir(path: str):
    d = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def clamp(v, lo, hi): return max(lo, min(hi, v))
def safe_float(x, default=np.nan):
    try: return float(x)
    except Exception: return default
def pct(n): return f"{n*100:.2f}%"

# =========================
# Capital/Leverage Helpers
# =========================

def _count_open_trades(open_trades) -> int:
    try:
        return max(0, int(len(open_trades or [])))
    except Exception:
        return 0

def _used_capital_usdt(open_trades) -> float:
    """Compute total notional (USDT) across open trades."""
    tot = 0.0
    for t in (open_trades or []):
        try:
            if isinstance(t, dict):
                tot += float(t.get("notional", 0.0))
            else:
                tot += float(getattr(t, "notional", 0.0))
        except Exception:
            pass
    return tot

# =========================
# Config
# =========================

@dataclass
class Config:
    timeframe: str = "30m"
    lookback: int = 300

    # ===== Default Risk Settings (updated) =====
    leverage_x: float = 10.0      # ثابت X10
    capital_pct: float = 0.90     # 90% من رأس المال
    max_open_trades: int = 3          # الحد الأقصى للصفقات المتزامنة
    equal_split_mode: bool = False    # تقسيم ثابت (True) أم ديناميكي (False)
    dynamic_split_mode: bool = True   # تقسيم ديناميكي للمتبقي على الخانات المتبقية
    min_notional: float = 10.0        # حد أدنى للقيمة الإسمية للصفقة
    max_notional: float = 1_000_000.0 # حد أقصى للقيمة الإسمية (أمان)

    # ===== Filters & Quality Sizing =====
    filters_enabled: bool = True       # فلترة الإشارات
    trend_filter: bool = True          # فلتر اتجاه
    min_atr_pct: float = 0.15          # كان 0.20 → لِين بسيط
    min_vol_z: float = 0.20            # كان 0.50 → يسمح بسيولة متوسطة
    min_vol_pct_alt: float = 0.40      # حد بديل: الحجم ضمن أعلى 40% من نطاقه المحلي
    # أوضاع الفلترة: strict | any | kofn | soft
    filter_mode: str = "kofn"          # الافتراضي: 2 من 3 يمرّ
    filters_k: int = 2                 # k-of-n
    # لـ soft: تصغير/تكبير حجم الصفقة بدل الحظر
    soft_scale_low: float = 0.70
    soft_scale_high: float = 1.00
    quality_sizing: bool = True        # ربط الحجم بجودة الإشارة
    min_quality: float = 0.35          # أقل جودة مسموح بها
    size_scale_low: float = 0.60       # معامل التصغير عند الجودة الدنيا المقبولة
    size_scale_high: float = 1.30      # معامل التكبير عند الجودة العالية

    # Indicators / windows
    ema_fast: int = 9
    ema_slow: int = 21
    atr_window: int = 14
    rsi_len: int = 14
    bb_len: int = 20
    bb_std: float = 2.0
    vol_ma_len: int = 30
    box_len: int = 20
    regime_lookback: int = 120
    low_vol_pct: float = 0.35
    high_vol_pct: float = 0.70

    # Fixed TP/SL (يستخدم كخيار احتياطي)
    fixed_tp_pct: float = 0.01
    fixed_sl_pct: float = 0.005

    # ==== تعديل #2: إضافة إعدادات TP/SL الديناميكية ====
    use_atr_tp_sl: bool = True    # لتفعيل/تعطيل الميزة بسهولة
    atr_tp_mult: float = 2.0      # TP = السعر + (2.0 * ATR)
    atr_sl_mult: float = 1.2      # SL = السعر - (1.2 * ATR)
    
    # TREND
    trend_min_slope: float = 0.0003
    trend_vol_mult: float = 1.3

    # BO
    bo_vol_mult: float = 1.2
    bo_range_share: float = 0.5

    # MR
    mr_rsi_buy: float = 25.0
    mr_rsi_sell: float = 75.0
    sr_lookback: int = 50

    # PB (Pullback)
    pb_pullback_pct: float = 0.0035
    pb_wick_ratio: float = 0.35

    # VWAP-R
    vwap_dev_mult: float = 1.5

    # KSQ (Keltner Squeeze)
    keltner_len: int = 20
    keltner_mult: float = 1.5
    squeeze_bb_mult: float = 1.6

    # SCALP strategy tuning
    scalp_rsi_buy: float = 40.0
    scalp_rsi_sell: float = 60.0
    scalp_tp_atr_mult: float = 1.2
    scalp_sl_atr_mult: float = 0.8
    debug_signals: bool = True   # كان False، فعّله افتراضيًا للتشخيص

    # Filters
    funding_filter: bool = True
    max_abs_funding: float = 0.003

    # Account modes (override if auto-detect fails)
    okx_pos_mode: Optional[str] = None   # "net_mode" or "long_short_mode"
    okx_margin_mode: Optional[str] = None  # "cross" or "isolated"
    okx_demo: bool = True  # استخدم حساب الـ Demo بدلاً من الحقيقي


    # Quiet windows (UTC HH:MM)
    event_quiet_minutes: int = 10
    quiet_windows_utc: Tuple[str, ...] = ()

    # News Guard — افتراضيًا مقفول
    news_enabled: bool = False
    news_lookback_minutes: int = 60
    news_keywords: Tuple[str, ...] = ("ETF","hack","exploit","ban","SEC","lawsuit","fork","upgrade","halving")

    # Universe
    # ==== تعديل #1: تغيير العدد إلى 10 ====
    top_n_symbols: int = 10
    refresh_universe_minutes: int = 360
    health_refresh_minutes: int = 90
    health_test_limit: int = 50

    # Telegram
    telegram_enabled: bool = True
    telegram_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")

    # Throttles
    min_minutes_between_same_signal: int = 3
    min_seconds_between_alerts_global: int = 50

    # Committee & Bandit
    exploration_eps: float = 0.08
    dyn_quorum_base: float = 0.55
    quorum_boost_high_vol: float = -0.07
    quorum_boost_good_hit: float = -0.05
    quorum_penalty_bad_hit: float = 0.05

    # Self-Evolving (محلي — مش بيعتمد على OpenAI)
    evolve_enabled: bool = True
    evolve_mutations_per_round: int = 2
    evolve_trial_weight: float = 0.05
    evolve_decay: float = 0.95

    # ==== الإضافات الجديدة (إدارة المخاطر المتقدمة) ====
    # Confidence Filter
    min_confidence_accept: float = 0.75

    # Committee Override: لازم أقل حاجة X نماذج تتفق على نفس الاتجاه
    committee_min_agree: int = 2

    # Files
    logs_dir: str = "./logs"
    signals_csv: str = "./logs/signals_log.csv"
    trades_csv: str  = "./logs/trades_log.csv"
    models_csv: str  = "./logs/models_log.csv"
    ml_csv: str      = "./logs/ml_dataset.csv"
    state_json: str  = "./logs/state.json"

# =========================
# Telegram
# =========================

class Notifier:
    def __init__(self, cfg: Config):
        self.enabled = bool(cfg.telegram_enabled and cfg.telegram_token and cfg.telegram_chat_id)
        self.base = f"https://api.telegram.org/bot{cfg.telegram_token}" if self.enabled else None
        self.chat_id = cfg.telegram_chat_id
    def send(self, text: str):
        print(text)
        if not self.enabled:
            return
        try:
            r = requests.post(f"{self.base}/sendMessage",
                              json={"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True},
                              timeout=10)
            if r.status_code != 200:
                print("[WARN] Telegram send failed:", r.text)
        except Exception as e:
            print("[WARN] Telegram exception:", e)

# =========================
# Exchange
# =========================

class FuturesExchange:
    def __init__(self, cfg: Config):
        key = os.getenv("OKX_API_KEY")
        secret = os.getenv("OKX_API_SECRET")
        password = os.getenv("OKX_API_PASSWORD") or os.getenv("OKX_API_PASSPHRASE")
        opts = {"defaultType": "swap"}
        headers = {}
        if cfg.okx_demo:
            opts["demo"] = True
            headers["x-simulated-trading"] = "1"
        self.x = ccxt.okx({
            "apiKey": key,
            "secret": secret,
            "password": password,
            "options": opts,
            "headers": headers,
            "enableRateLimit": True,
            "timeout": 15000,
        })
        if cfg.okx_demo:
            # Force all requests to hit the demo environment
            try:
                self.x.set_sandbox_mode(True)
            except Exception:
                pass
        if cfg.okx_demo:
            # Demo accounts cannot access the private currencies endpoint; disable it (نغلق من الجهتين)
            try:
                self.x.options["fetchCurrencies"] = False
            except Exception:
                pass
            try:
                self.x.has["fetchCurrencies"] = False
            except Exception:
                pass
        self.x.load_markets()

        # Determine account modes for proper order parameters
        self.pos_mode = "net"        # "net" أو "long_short"
        self.margin_mode = "cross"   # "cross" أو "isolated"
        self.leverage = float(cfg.leverage_x)
        try:
            info = self.x.privateGetAccountConfig()
            data = info.get("data", [])
            if data:
                cfg0 = data[0]
                pm = cfg0.get("posMode") or self.pos_mode
                mm = cfg0.get("marginMode") or cfg0.get("mgnMode") or self.margin_mode
                self.pos_mode = str(pm).replace("-", "_").lower()
                self.margin_mode = str(mm).lower()
        except Exception as e:
            print("[WARN] fetch account config failed:", e)
        # Manual overrides from config
        if cfg.okx_pos_mode:
            self.pos_mode = str(cfg.okx_pos_mode).replace("-", "_").lower()
        if cfg.okx_margin_mode:
            self.margin_mode = str(cfg.okx_margin_mode).lower()

        # Ensure (or set) account modes to avoid 51010
        try:
            setup_symbol = next((s for s in self.x.symbols if s.endswith(":USDT")), "BTC/USDT:USDT")
            hedged = not self.pos_mode.startswith("net")
            try:
                self.x.set_position_mode(hedged)
            except Exception:
                self.x.privatePostAccountSetPositionMode({
                    "posMode": "long_short_mode" if hedged else "net_mode"
                })
            try:
                self.x.set_leverage(self.leverage, setup_symbol, {"mgnMode": self.margin_mode})
            except Exception:
                m = self.x.market(setup_symbol)
                self.x.privatePostAccountSetLeverage({
                    "instId": m["id"],
                    "lever": str(self.leverage),
                    "mgnMode": self.margin_mode,
                })
        except Exception as e:
            msg = str(getattr(e, 'args', [''])[0])
            if "59000" in msg:
                print("[WARN] ensure account modes failed: open orders or positions exist; attempting to flatten")
                self.flatten_all()
                try:
                    self.x.set_position_mode(hedged)
                    self.x.set_leverage(self.leverage, setup_symbol, {"mgnMode": self.margin_mode})
                except Exception as e2:
                    msg2 = str(getattr(e2, 'args', [''])[0])
                    if "59000" in msg2:
                        print("[WARN] ensure account modes skipped: close orders/positions before changing modes (code 59000)")
                    else:
                        print("[WARN] ensure account modes failed:", e2)
            else:
                print("[WARN] ensure account modes failed:", e)

        self.cfg = cfg
        self._universe_cache: Dict[str, any] = {"ts": 0.0, "symbols": []}
        self._health_cache: Dict[str, float] = {}
        self._bad_cache: Dict[str, float] = {}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        ohlcv = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime").drop(columns=["timestamp"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna()

    def fetch_ohlcv_paged(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Fetch OHLCV data in multiple requests to bypass the 300-candle limit on OKX.
        Useful for demo accounts that enforce a per-call maximum.
        """
        max_per_call = 300
        out = []
        since = None
        remaining = int(limit)
        while remaining > 0:
            batch = min(remaining, max_per_call)
            try:
                ohlcv = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=batch, since=since)
            except Exception as e:
                print(f"[WARN] fetch_ohlcv failed for {symbol}: {e}")
                break
            if not ohlcv:
                break
            out.extend(ohlcv)
            if len(ohlcv) < batch:
                break
            last_ts = int(ohlcv[-1][0])
            since = last_ts + 1
            remaining -= len(ohlcv)
            time.sleep(0.05)
        if not out:
            return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)
        df = pd.DataFrame(out, columns=["timestamp","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime").drop(columns=["timestamp"]).apply(pd.to_numeric, errors="coerce").dropna()
        return df.sort_index()

    def fetch_ticker_price(self, symbol: str) -> Optional[float]:
        """Fetch the latest trade price for a symbol."""
        try:
            ticker = self.x.fetch_ticker(symbol)
            return safe_float(ticker.get("last")) or safe_float(ticker.get("close"))
        except Exception:
            return None

    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        try:
            fr = self.x.fetch_funding_rate(symbol)
            return safe_float(fr.get("fundingRate"))
        except Exception:
            return None

    def get_balance_usdt(self) -> float:
        try:
            bal = self.x.fetch_balance(params={"type": "swap"})
            free = bal.get("free", {}).get("USDT")
            if free is None:
                details = bal.get("info", {}).get("data", [{}])[0].get("details", [])
                for d in details:
                    if d.get("ccy") == "USDT":
                        free = d.get("availBal")
                        break
            return float(free or 0.0)
        except Exception:
            return 0.0

    def create_order(self, symbol: str, side: str, contract_amt: float, reduce_only: bool = False):
        """Execute a market order in contract units on OKX."""
        params = {"tdMode": self.margin_mode, "reduceOnly": reduce_only}
        if not self.pos_mode.startswith("net"):
            params["posSide"] = "long" if side.lower() == "buy" else "short"
        if self.leverage:
            params["lever"] = str(self.leverage)
        try:
            o = self.x.create_order(symbol, "market", side, contract_amt, params=params)
            oid = o.get("id") or o.get("orderId") or o.get("info", {}).get("ordId")
            if not oid:
                print("[WARN] create_order returned no id:", o)
                return None
            return o
        except ccxt.InsufficientFunds:
            bal = self.get_balance_usdt()
            print(f"[WARN] insufficient USDT margin: available {bal:.2f} — skipping order for {symbol}")
            return None
        except Exception as e:
            msg = str(e)
            if "Insufficient" in msg and "margin" in msg:
                bal = self.get_balance_usdt()
                print(f"[WARN] insufficient USDT margin ({bal:.2f} available): {msg}")
            else:
                print("[WARN] create_order failed:", msg)
            return None

    def close_position(self, symbol: str, orig_side: str, contract_amt: float):
        """Close an open position by sending the opposite order in contract units."""
        opp = "sell" if orig_side.lower() == "buy" else "buy"
        return self.create_order(symbol, opp, contract_amt, reduce_only=True)

    def flatten_all(self):
        try:
            for s in self.x.symbols:
                try:
                    self.x.cancel_all_orders(s)
                except Exception:
                    continue
            try:
                positions = self.x.fetch_positions(params={"type": "swap"})
                for p in positions:
                    amt = abs(float(p.get("contracts") or p.get("positionAmt") or 0))
                    if amt <= 0:
                        continue
                    sym = p.get("symbol") or p.get("info", {}).get("instId")
                    side = p.get("side") or ("long" if float(p.get("contracts", 0)) > 0 else "short")
                    orig = "buy" if side == "long" else "sell"
                    self.close_position(sym, orig, amt)
            except Exception:
                pass
        except Exception as e:
            print("[WARN] flatten failed:", e)

    def get_top_symbols(self, n: int = 50) -> List[str]:
        nowt = time.time()
        if (nowt - self._universe_cache["ts"]) < self.cfg.refresh_universe_minutes*60 and self._universe_cache["symbols"]:
            return self._universe_cache["symbols"]
        top: List[Tuple[str,float]] = []
        try:
            tickers = self.x.fetch_tickers()
            for m in self.x.markets.values():
                if not m.get("swap") or not m.get("contract"): continue
                if m.get("quote") != "USDT": continue
                if not m.get("active", True): continue
                sym = m["symbol"]
                t = tickers.get(sym, {})
                qv = t.get("quoteVolume")
                if qv is None:
                    qv = float(t.get("info", {}).get("quoteVolume", 0) or 0)
                top.append((sym, float(qv)))
            top.sort(key=lambda x: x[1], reverse=True)
            syms = [s for s,_ in top[:n]] or ["BTC/USDT:USDT","ETH/USDT:USDT"]
        except Exception:
            syms = ["BTC/USDT:USDT","ETH/USDT:USDT"]
        self._universe_cache = {"ts": nowt, "symbols": syms}
        return syms

    def filter_healthy(self, symbols: List[str]) -> List[str]:
        ok = []
        nowt = time.time()
        for s in symbols:
            last_bad = self._bad_cache.get(s, 0)
            if nowt - last_bad < self.cfg.health_refresh_minutes*60:
                continue
            last_ok = self._health_cache.get(s, 0)
            if nowt - last_ok < self.cfg.health_refresh_minutes*60:
                ok.append(s); continue
            try:
                _ = self.fetch_ohlcv_paged(s, self.cfg.timeframe, limit=self.cfg.lookback)
                if _.shape[0] >= self.cfg.lookback:
                    ok.append(s)
                    self._health_cache[s] = nowt
                else:
                    print(
                        f"[WARN] insufficient history for {s}: "
                        f"{_.shape[0]} < {self.cfg.lookback}"
                    )
                    self._bad_cache[s] = nowt
            except Exception:
                self._bad_cache[s] = nowt
            time.sleep(0.05)
        if not ok:
            fallback = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
            for s in fallback:
                last_bad = self._bad_cache.get(s, 0)
                if nowt - last_bad >= self.cfg.health_refresh_minutes * 60:
                    ok.append(s)
        return ok

# =========================
# Indicators
# =========================

def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df.copy()

    d["ema_fast"] = ta.trend.EMAIndicator(d["close"], window=cfg.ema_fast).ema_indicator()
    d["ema_slow"] = ta.trend.EMAIndicator(d["close"], window=cfg.ema_slow).ema_indicator()
    d["ema9"]  = d["ema_fast"]
    d["ema21"] = d["ema_slow"]

    d["rsi"] = ta.momentum.RSIIndicator(d["close"], window=cfg.rsi_len).rsi()

    bb = ta.volatility.BollingerBands(d["close"], window=cfg.bb_len, window_dev=cfg.bb_std)
    d["bb_mid"], d["bb_up"], d["bb_dn"] = bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()

    atr = ta.volatility.AverageTrueRange(d["high"], d["low"], d["close"], window=cfg.atr_window)
    d["atr"] = atr.average_true_range()
    atr_ma_win = max(10, int(cfg.atr_window * 3))
    d["atr_ma"] = d["atr"].rolling(atr_ma_win, min_periods=atr_ma_win//2).mean()
    d["atr_pct"] = d["atr"] / d["atr_ma"]

    session = d.index.tz_convert("UTC").normalize()
    d["vwap_num"] = (d["close"] * d["volume"]).groupby(session).cumsum()
    d["vwap_den"] = d["volume"].groupby(session).cumsum().replace(0, np.nan)
    d["vwap"] = d["vwap_num"] / d["vwap_den"]

    vol_ma = d["volume"].rolling(cfg.vol_ma_len, min_periods=cfg.vol_ma_len//2).mean()
    vol_std = d["volume"].rolling(cfg.vol_ma_len, min_periods=cfg.vol_ma_len//2).std()
    d["vol_ma"] = vol_ma
    d["vol_z"] = (d["volume"] - vol_ma) / vol_std.replace(0, np.nan)

    # بديل للسيولة: نسبة الحجم داخل نطاقه المحلي (0..1)
    _w = max(10, int(cfg.vol_ma_len))
    vmin = d["volume"].rolling(_w, min_periods=_w//2).min()
    vmax = d["volume"].rolling(_w, min_periods=_w//2).max()
    vrng = (vmax - vmin).replace(0, np.nan)
    d["vol_pct"] = (d["volume"] - vmin) / vrng

    d["vol_spike"] = d["volume"] > (vol_ma * cfg.trend_vol_mult)
    d["bo_vol_spike"] = d["volume"] > (vol_ma * cfg.bo_vol_mult)

    d["recent_high"] = d["high"].rolling(cfg.box_len).max()
    d["recent_low"]  = d["low"].rolling(cfg.box_len).min()

    d["sr_high"] = d["high"].rolling(cfg.sr_lookback).max()
    d["sr_low"]  = d["low"].rolling(cfg.sr_lookback).min()

    d["ema9_slope"] = (d["ema9"] - d["ema9"].shift(3)) / d["close"]
    d["ema21_slope"] = (d["ema21"] - d["ema21"].shift(3)) / d["close"]
    d["bb_width"] = (d["bb_up"] - d["bb_dn"]) / d["bb_mid"]

    adx = ta.trend.ADXIndicator(d["high"], d["low"], d["close"], window=cfg.atr_window)
    d["adx"] = adx.adx()
    d["di_pos"] = adx.adx_pos()
    d["di_neg"] = adx.adx_neg()

    ema = ta.trend.EMAIndicator(d["close"], window=cfg.keltner_len).ema_indicator()
    rng = ta.volatility.AverageTrueRange(d["high"], d["low"], d["close"], window=cfg.keltner_len).average_true_range()
    d["kel_mid"] = ema
    d["kel_up"] = ema + cfg.keltner_mult * rng
    d["kel_dn"] = ema - cfg.keltner_mult * rng

    def rolling_pctl(x: pd.Series):
        last = x.iloc[-1]
        return float((x <= last).mean())
    d["atr_pct_pctl"] = d["atr_pct"].rolling(cfg.regime_lookback).apply(rolling_pctl, raw=False)

    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    return d

# =============== Filters Engine (k-of-n / soft) ===============
def eval_filters(side: str, row: pd.Series, cfg: Config,
                 px: float, bb_lo: float, bb_hi: float, rsi: float) -> Tuple[bool, Dict[str, bool], float]:
    """Evaluate liquidity, ATR and trend filters.

    Returns:
        passed (bool): decision for strict/any/kofn (always True for soft)
        mask (dict): individual results for liq/atr/trend
        score (float): fraction of passed filters, used for soft scaling
    """
    vol_z = float(row.get("vol_z", -10.0))
    vol_pct = float(row.get("vol_pct", -1.0))
    pass_liq = (vol_z >= cfg.min_vol_z) or (vol_pct >= cfg.min_vol_pct_alt)

    atr_pct = float(row.get("atr_pct", 0.0))
    pass_atr = atr_pct >= cfg.min_atr_pct

    ema_fast = float(row.get("ema_fast", np.nan))
    ema_slow = float(row.get("ema_slow", np.nan))
    pass_trend = True
    if cfg.trend_filter:
        band_w = max(1e-9, float(bb_hi - bb_lo))
        touch_buy = float(np.clip((bb_lo + 0.10 * band_w - px) / band_w, 0.0, 1.0))
        touch_sell = float(np.clip((px - (bb_hi - 0.10 * band_w)) / band_w, 0.0, 1.0))
        rsi_buy_extreme = rsi <= max(25.0, getattr(cfg, "scalp_rsi_buy", 40.0) - 5.0)
        rsi_sell_extreme = rsi >= min(75.0, getattr(cfg, "scalp_rsi_sell", 60.0) + 5.0)
        allow_ct_buy = rsi_buy_extreme and (touch_buy >= 0.25)
        allow_ct_sell = rsi_sell_extreme and (touch_sell >= 0.25)
        if side == "BUY":
            pass_trend = (ema_fast > ema_slow) or allow_ct_buy
        else:
            pass_trend = (ema_fast < ema_slow) or allow_ct_sell

    mask = {"liq": pass_liq, "atr": pass_atr, "trend": pass_trend}
    n_pass = sum(1 for v in mask.values() if v)
    n_total = len(mask)
    score = n_pass / n_total if n_total else 0.0

    mode = (cfg.filter_mode or "kofn").lower()
    if mode == "strict":
        return (n_pass == n_total, mask, score)
    if mode == "any":
        return (n_pass >= 1, mask, score)
    if mode == "kofn":
        k = max(1, int(cfg.filters_k))
        return (n_pass >= k, mask, score)
    if mode == "soft":
        return (True, mask, score)
    return (n_pass == n_total, mask, score)

# =========================
# Quality Scoring
# =========================
def score_signal(side: str, row: pd.Series, cfg: Config) -> float:
    px = float(row["close"])
    bb_lo = float(row.get("bb_dn", np.nan))
    bb_hi = float(row.get("bb_up", np.nan))
    rsi   = float(row.get("rsi", np.nan))
    atr_pct = float(row.get("atr_pct", 0.0))
    vol_z   = float(row.get("vol_z", -10.0))
    ema_fast = float(row.get("ema_fast", np.nan))
    ema_slow = float(row.get("ema_slow", np.nan))
    if any([np.isnan(v) for v in [bb_lo, bb_hi, rsi, ema_fast, ema_slow]]):
        return 0.0
    band_w = max(1e-12, bb_hi - bb_lo)
    if side == "BUY":
        band_score = np.clip((bb_lo + (band_w * 0.15) - px) / band_w, 0.0, 1.0)
        rsi_score  = np.clip((45.0 - rsi) / 45.0, 0.0, 1.0)
        trend_score = 1.0 if ema_fast > ema_slow else 0.0
    else:
        band_score = np.clip((px - (bb_hi - (band_w * 0.15))) / band_w, 0.0, 1.0)
        rsi_score  = np.clip((rsi - 55.0) / 45.0, 0.0, 1.0)
        trend_score = 1.0 if ema_fast < ema_slow else 0.0
    atr_score = np.clip(atr_pct / 1.5, 0.0, 1.0)
    vol_score = np.clip((vol_z - 0.0) / 2.0, 0.0, 1.0)
    q = (0.30*band_score + 0.20*rsi_score + 0.20*atr_score + 0.15*vol_score + 0.15*trend_score)
    return float(np.clip(q, 0.0, 1.0))

# =========================
# Regime
# =========================

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
    p = row.get("atr_pct_pctl", np.nan)
    if np.isnan(p): v = "medium"
    elif p < cfg.low_vol_pct: v = "low"
    elif p > cfg.high_vol_pct: v = "high"
    else: v = "medium"
    return Regime(t, v)

# =========================
# Signal object
# =========================

@dataclass
class Signal:
    side: Optional[str]
    sl: float
    tp: float
    model: str
    reason: str
    confidence: float = 0.5

# ==== تعديل #2: إضافة دالة لحساب TP/SL بناءً على ATR ====
def make_tp_sl_atr(entry: float, side: str, atr: float, cfg: Config) -> Tuple[float, float]:
    """يحسب TP/SL بناءً على مضاعفات الـ ATR."""
    if side == "buy":
        tp = entry + (cfg.atr_tp_mult * atr)
        sl = entry - (cfg.atr_sl_mult * atr)
    else: # sell
        tp = entry - (cfg.atr_tp_mult * atr)
        sl = entry + (cfg.atr_sl_mult * atr)
    return tp, sl

def make_tp_sl(entry: float, side: str, cfg: Config) -> Tuple[float, float]:
    if side == "buy":
        tp = entry * (1 + cfg.fixed_tp_pct)
        sl = entry * (1 - cfg.fixed_sl_pct)
    else:
        tp = entry * (1 - cfg.fixed_tp_pct)
        sl = entry * (1 + cfg.fixed_sl_pct)
    return tp, sl

def get_tp_sl(entry: float, side: str, row: pd.Series, cfg: Config) -> Tuple[float, float]:
    """دالة موحدة لاختيار طريقة حساب TP/SL."""
    atr_val = safe_float(row.get("atr"))
    if cfg.use_atr_tp_sl and not np.isnan(atr_val) and atr_val > 0:
        return make_tp_sl_atr(entry, side, atr_val, cfg)
    else:
        return make_tp_sl(entry, side, cfg)

# ====== Strategy: SCALP (BB + RSI) ======
def bbands(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return lower, ma, upper

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1 / n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ma_up / (ma_dn.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def strat(df: pd.DataFrame):
    if len(df) < 40:
        return None
    a = atr(df, 14).iloc[-1]
    if np.isnan(a) or a <= 0:
        return None
    lo, _, hi = bbands(df["close"], n=20, k=2.0)
    bb_lo = float(lo.iloc[-1])
    bb_hi = float(hi.iloc[-1])
    r = float(rsi(df["close"], n=14).iloc[-1])
    px = float(df["close"].iloc[-1])
    tp, sl = 1.2 * a, 0.8 * a
    if px <= bb_lo and r <= 40:
        return ("buy", px - sl, px + tp, f"SCALP: px<=BBlo & RSI={r:.1f}")
    if px >= bb_hi and r >= 60:
        return ("sell", px + sl, px - tp, f"SCALP: px>=BBhi & RSI={r:.1f}")
    return None

# SCALP strategy (BB + RSI)
# =========================

def sig_scalp(symbol: str, row: pd.Series, cfg: Config) -> Optional[Tuple[str, str, float, float]]:
    """SCALP logic with flexible filters."""
    atr = safe_float(row.get("atr", np.nan))
    px = float(row["close"])
    bb_lo = float(row["bb_dn"])
    bb_hi = float(row["bb_up"])
    rsi_val = float(row["rsi"])

    if cfg.debug_signals:
        print(f"[DEBUG] SCALP inputs: px={px:.4f} bb_lo={bb_lo:.4f} bb_hi={bb_hi:.4f} rsi={rsi_val:.2f} atr={atr:.4f}")

    if np.isnan(atr) or atr <= 0:
        if cfg.debug_signals:
            print(f"[DEBUG] SCALP skip: invalid ATR={atr}")
        return None

    buy_cond  = (px <= bb_lo) and (rsi_val <= cfg.scalp_rsi_buy) and (atr > 0)
    sell_cond = (px >= bb_hi) and (rsi_val >= cfg.scalp_rsi_sell) and (atr > 0)

    filter_scale = 1.0
    if cfg.filters_enabled and (buy_cond or sell_cond):
        side = "BUY" if buy_cond else "SELL"
        passed, mask, score = eval_filters(side, row, cfg, px, bb_lo, bb_hi, rsi_val)
        mode = (cfg.filter_mode or "kofn").lower()
        if mode in ("strict", "any", "kofn"):
            if not passed:
                if cfg.debug_signals:
                    print(f"[FILTER] {symbol} BLOCK mode={mode} mask={mask} score={score:.2f}")
                buy_cond = sell_cond = False
        else:
            lo = float(cfg.soft_scale_low)
            hi = float(cfg.soft_scale_high)
            filter_scale = float(np.clip(lo + (hi - lo) * score, lo, hi))
            if cfg.debug_signals:
                print(f"[FILTER] {symbol} SOFT scale={filter_scale:.2f} mask={mask} score={score:.2f}")

    side = "BUY" if buy_cond else ("SELL" if sell_cond else None)
    if side is None:
        if cfg.debug_signals:
            print(f"[DEBUG] SCALP skip: px={px:.4f} bb_lo={bb_lo:.4f} bb_hi={bb_hi:.4f} rsi={rsi_val:.2f} atr={atr:.4f}")
        return None

    quality = score_signal(side, row, cfg)
    if cfg.debug_signals:
        print(f"[QUAL] {symbol} side={side} q={quality:.2f}")
    if quality < float(cfg.min_quality):
        if cfg.debug_signals:
            print(f"[QUAL] {symbol} skip: q<{cfg.min_quality}")
        return None

    reason = (
        f"SCALP: px<=BBlo & RSI={rsi_val:.1f}" if side == "BUY" else f"SCALP: px>=BBhi & RSI={rsi_val:.1f}"
    )
    return (side.lower(), reason, quality, filter_scale)

def ctx_key(regime: Regime) -> str:
    return f"{regime.trend}|{regime.vol_bucket}"
# =========================
# Paper Engine
# =========================


@dataclass
class PaperTrade:
    id: str
    timestamp: str
    symbol: str
    timeframe: str
    side: str
    entry: float
    sl: float
    tp: float
    model: str
    qty: float
    notional: float
    status: str = "open"
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    result: Optional[str] = None
    pnl_usd: Optional[float] = None

class Paper:
    def __init__(self, cfg: Config, ref_equity: float):
        self.cfg = cfg
        self.ref_equity = ref_equity
        self.open: Dict[str, PaperTrade] = {}
        ensure_dir(cfg.signals_csv); ensure_dir(cfg.trades_csv); ensure_dir(cfg.ml_csv); ensure_dir(cfg.models_csv); ensure_dir(cfg.state_json)
        if not os.path.exists(cfg.signals_csv):
            pd.DataFrame(columns=[
                "time","symbol","tf","price","side","model","tp","sl",
                "ref_qty","ref_notional","rr","reason","conf",
                "trend","vol_bucket","ctx_key"
            ]).to_csv(cfg.signals_csv, index=False)
        if not os.path.exists(cfg.trades_csv):
            pd.DataFrame(columns=[
                "id","open_time","close_time","symbol","tf","side","entry","exit","result","model","pnl_usd","hold_sec","ctx_key"
            ]).to_csv(cfg.trades_csv, index=False)
        if not os.path.exists(cfg.ml_csv):
            pd.DataFrame(columns=[
                "trade_id","symbol","tf","side","model","open_time","close_time","result","pnl_usd",
                "price","ema9","ema21","ema9_slope","ema21_slope","rsi","bb_mid","bb_up","bb_dn","bb_width",
                "atr","atr_pct","atr_pct_pctl","vwap","vol","vol_ma","vol_spike","bo_vol_spike","recent_high",
                "recent_low","sr_high","sr_low","regime_trend","regime_vol","ctx_key","adx","di_pos","di_neg",
                "kel_mid","kel_up","kel_dn"
            ]).to_csv(cfg.ml_csv, index=False)
        if not os.path.exists(cfg.models_csv):
            pd.DataFrame(columns=["time","symbol","tf","model","ctx_key","decision_score","accepted","weight","conf","notes"]).to_csv(cfg.models_csv, index=False)

    def _gen_id(self) -> str: return f"T{int(time.time()*1000)}"

    def open_virtual(self, symbol: str, price: float, sig: Signal, qty: float,
                     notional: float, cfg: Config) -> PaperTrade:
        t = PaperTrade(
            id=self._gen_id(), timestamp=fmt_ts(), symbol=symbol, timeframe=cfg.timeframe,
            side=sig.side, entry=price, sl=float(sig.sl), tp=float(sig.tp),
            model=sig.model, qty=qty, notional=notional
        )
        self.open[t.id] = t
        return t

    def _hit(self, side: str, high: float, low: float, level: float, is_tp: bool) -> bool:
        if side == "buy":
            return high >= level if is_tp else low <= level
        else:
            return low <= level if is_tp else high >= level

    def update_with_candle(self, symbol: str, high: float, low: float, ts) -> List[PaperTrade]:
        to_close = []
        for tid, t in list(self.open.items()):
            if t.status != "open" or t.symbol != symbol:
                continue
            # لا نغلق الصفقة باستخدام شمعة أقدم من وقت فتحها
            if pd.to_datetime(ts) <= pd.to_datetime(t.timestamp):
                continue
            hit_tp = self._hit(t.side, high, low, t.tp, True)
            hit_sl = self._hit(t.side, high, low, t.sl, False)
            if hit_tp and hit_sl:
                res, px = "sl", t.sl
            elif hit_tp:
                res, px = "tp", t.tp
            elif hit_sl:
                res, px = "sl", t.sl
            else:
                continue
            t.status = "closed"; t.result = res; t.exit_price = float(px); t.exit_time = fmt_ts()
            qty = t.qty
            pnl = (t.exit_price - t.entry) * qty * (1 if t.side == "buy" else -1)
            t.pnl_usd = round(pnl, 4)
            to_close.append(tid)
        closed = [self.open[k] for k in to_close]
        for k in to_close:
            self.open.pop(k, None)
        return closed

    def force_close(self, trade: PaperTrade, price: float, reason: str = "timeout") -> PaperTrade:
        """Forcefully close an open trade at the given ``price``.

        Parameters
        ----------
        trade: PaperTrade
            The trade to close.
        price: float
            The price at which the trade is closed.
        reason: str
            Label to store in ``trade.result`` describing why the trade was
            closed (default: ``"timeout"``).
        """
        trade.status = "closed"
        trade.result = reason
        trade.exit_price = float(price)
        trade.exit_time = fmt_ts()
        pnl = (trade.exit_price - trade.entry) * trade.qty * (1 if trade.side == "buy" else -1)
        trade.pnl_usd = round(pnl, 4)
        self.open.pop(trade.id, None)
        return trade

    def persist_closed(self, closed: List[PaperTrade], cfg: Config, ctx: str):
        if not closed: return
        rows=[]
        for t in closed:
            hold = int((pd.to_datetime(t.exit_time) - pd.to_datetime(t.timestamp)).total_seconds())
            rows.append({
                "id": t.id, "open_time": t.timestamp, "close_time": t.exit_time, "symbol": t.symbol, "tf": t.timeframe,
                "side": t.side, "entry": round(t.entry,6), "exit": round(t.exit_price,6),
                "result": t.result, "model": t.model, "pnl_usd": t.pnl_usd, "hold_sec": hold, "ctx_key": ctx
            })
        pd.DataFrame(rows).to_csv(cfg.trades_csv, mode="a", header=False, index=False)

    def ml_snapshot(self, trade_id:str, symbol:str, row:pd.Series, regime:Regime):
        feat = {
            "trade_id": trade_id, "symbol": symbol, "tf": self.cfg.timeframe, "side": "", "model": "",
            "open_time": fmt_ts(), "close_time":"", "result":"", "pnl_usd":"",
            "price": float(row["close"]),
            "ema9": float(row.get("ema9", np.nan) or np.nan),
            "ema21": float(row.get("ema21", np.nan) or np.nan),
            "ema9_slope": float(row.get("ema9_slope", np.nan) or np.nan),
            "ema21_slope": float(row.get("ema21_slope", np.nan) or np.nan),
            "rsi": float(row.get("rsi", np.nan) or np.nan),
            "bb_mid": float(row.get("bb_mid", np.nan) or np.nan),
            "bb_up": float(row.get("bb_up", np.nan) or np.nan),
            "bb_dn": float(row.get("bb_dn", np.nan) or np.nan),
            "bb_width": float(row.get("bb_width", np.nan) or np.nan),
            "atr": float(row.get("atr", np.nan) or np.nan),
            "atr_pct": float(row.get("atr_pct", np.nan) or np.nan),
            "atr_pct_pctl": float(row.get("atr_pct_pctl", np.nan) or np.nan),
            "vwap": float(row.get("vwap", np.nan) or np.nan),
            "vol": float(row.get("volume", np.nan) or np.nan),
            "vol_ma": float(row.get("vol_ma", np.nan) or np.nan),
            "vol_spike": bool(row.get("vol_spike", False)),
            "bo_vol_spike": bool(row.get("bo_vol_spike", False)),
            "recent_high": float(row.get("recent_high", np.nan) or np.nan),
            "recent_low": float(row.get("recent_low", np.nan) or np.nan),
            "sr_high": float(row.get("sr_high", np.nan) or np.nan),
            "sr_low": float(row.get("sr_low", np.nan) or np.nan),
            "regime_trend": regime.trend, "regime_vol": regime.vol_bucket, "ctx_key": ctx_key(regime),
            "adx": float(row.get("adx", np.nan) or np.nan),
            "di_pos": float(row.get("di_pos", np.nan) or np.nan),
            "di_neg": float(row.get("di_neg", np.nan) or np.nan),
            "kel_mid": float(row.get("kel_mid", np.nan) or np.nan),
            "kel_up": float(row.get("kel_up", np.nan) or np.nan),
            "kel_dn": float(row.get("kel_dn", np.nan) or np.nan),
        }
        pd.DataFrame([feat]).to_csv(self.cfg.ml_csv, mode="a", header=False, index=False)

    def log_signal(self, symbol: str, row: pd.Series, sig: Signal, qty_ref: float,
                   notional_ref: float, rr: Optional[float], cfg: Config, regime: Regime):
        pd.DataFrame([{
            "time": fmt_ts(), "symbol": symbol, "tf": cfg.timeframe, "price": float(row["close"]),
            "side": sig.side, "model": sig.model, "tp": round(sig.tp,6), "sl": round(sig.sl,6),
            "ref_qty": round(qty_ref,8), "ref_notional": round(notional_ref,2),
            "rr": rr if rr is not None else "", "reason": sig.reason, "conf": round(sig.confidence,2),
            "trend": regime.trend, "vol_bucket": regime.vol_bucket, "ctx_key": ctx_key(regime)
        }]).to_csv(cfg.signals_csv, mode="a", header=False, index=False)

    def log_model_vote(self, cfg: Config, symbol: str, regime: Regime, model: str, score: float, accepted: bool, weight: float, conf: float, notes: str = ""):
        pd.DataFrame([{
            "time": fmt_ts(), "symbol": symbol, "tf": cfg.timeframe, "model": model, "ctx_key": ctx_key(regime),
            "decision_score": round(score,4), "accepted": int(accepted), "weight": round(weight,4), "conf": round(conf,4),
            "notes": notes[:120]
        }]).to_csv(cfg.models_csv, mode="a", header=False, index=False)

# =========================
# News (اختياري)
# =========================

class NewsGuard:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cp_token = os.getenv("CRYPTOPANIC_TOKEN")
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        self.cache: Dict[str, float] = {}
    def too_hot(self, asset: str) -> bool:
        if not self.cfg.news_enabled: return False
        hot = False
        try:
            if self._cryptopanic(asset): hot = True
        except Exception: pass
        try:
            if self._newsapi(asset): hot = True
        except Exception: pass
        if hot: self.cache[asset] = time.time()
        else:
            if asset in self.cache and (time.time() - self.cache[asset]) < 600:
                hot = True
        return hot
    def _cryptopanic(self, asset: str) -> bool:
        if not self.cp_token: return False
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {"auth_token": self.cp_token, "currencies": asset.lower(), "kind": "news", "public": "true"}
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200: return False
        data = r.json().get("results", [])
        since = int((now_utc() - dt.timedelta(minutes=self.cfg.news_lookback_minutes)).timestamp())
        for it in data:
            pub = it.get("published_at") or it.get("created_at") or ""
            try: ts = dt.datetime.fromisoformat(pub.replace("Z","+00:00")).timestamp()
            except Exception: ts = 0
            if ts >= since:
                title = (it.get("title") or "").lower()
                important = it.get("importance") in ("high","very_high")
                kw = any(k.lower() in title for k in self.cfg.news_keywords)
                if important or kw: return True
        return False
    def _newsapi(self, asset: str) -> bool:
        if not self.newsapi_key: return False
        q = "bitcoin" if asset.upper()=="BTC" else ("ethereum" if asset.upper()=="ETH" else asset)
        url = "https://newsapi.org/v2/everything"
        since = (now_utc() - dt.timedelta(minutes=self.cfg.news_lookback_minutes)).isoformat()
        params = {"q": q, "from": since, "language": "en", "sortBy": "publishedAt",
                  "apiKey": self.newsapi_key, "pageSize": 20}
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200: return False
        arts = r.json().get("articles", [])
        for a in arts:
            title = (a.get("title") or "").lower()
            desc  = (a.get("description") or "").lower()
            if any(k.lower() in title or k.lower() in desc for k in self.cfg.news_keywords):
                return True
        return False

# =========================
# Quiet & sizing
# =========================

def in_quiet_window(cfg: Config) -> bool:
    if not cfg.quiet_windows_utc: return False
    nowt = now_utc().time().replace(second=0, microsecond=0)
    for hhmm in cfg.quiet_windows_utc:
        try:
            t = dt.datetime.strptime(hhmm, "%H:%M").time()
        except Exception:
            continue
        start = (dt.datetime.combine(dt.date.today(), t, tzinfo=dt.timezone.utc)
                 - dt.timedelta(minutes=cfg.event_quiet_minutes)).time()
        end   = (dt.datetime.combine(dt.date.today(), t, tzinfo=dt.timezone.utc)
                 + dt.timedelta(minutes=cfg.event_quiet_minutes)).time()
        if start <= nowt <= end:
            return True
    return False

# =========================
# Bot
# =========================

class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ex = FuturesExchange(cfg)
        self.notifier = Notifier(cfg)
        self.ref_equity = self.ex.get_balance_usdt() or 10000.0
        self.paper = Paper(cfg, self.ref_equity)
        base_universe = self.ex.get_top_symbols(cfg.top_n_symbols)
        self.symbols: List[str] = self.ex.filter_healthy(base_universe)
        self.news = NewsGuard(cfg)
        self.last_key: Dict[str, Optional[str]] = {}
        self.last_time: Dict[str, Optional[dt.datetime]] = {}
        self.last_alert_ts: float = 0.0
        self.closed_trades: List[PaperTrade] = []
        self.data_fail: Dict[str, int] = {}
        self.model_state: Dict[str, dict] = {}
        self.load_model_state()
        self.last_hourly_report = now_utc()
        self.last_daily_report_date = now_utc().date()

    def _maybe_hourly_report(self):
        now = now_utc()
        if now - self.last_hourly_report >= dt.timedelta(hours=1):
            since = self.last_hourly_report
            trades = [t for t in self.closed_trades if pd.to_datetime(t.exit_time) >= since]
            profit = sum((t.pnl_usd or 0) for t in trades if (t.pnl_usd or 0) > 0)
            loss = sum((t.pnl_usd or 0) for t in trades if (t.pnl_usd or 0) < 0)
            net = profit + loss
            msg = (f"⏱ Hourly Report\n"
                   f"Trades: {len(trades)}\n"
                   f"Profit: {profit:.2f} USDT\n"
                   f"Loss: {loss:.2f} USDT\n"
                   f"Net: {net:.2f} USDT")
            self.notifier.send(msg)
            self.last_hourly_report = now
            cutoff = now - dt.timedelta(days=1)
            self.closed_trades = [t for t in self.closed_trades if pd.to_datetime(t.exit_time) >= cutoff]

    def _send_daily_report(self, date: dt.date):
        if not os.path.exists(self.cfg.trades_csv):
            self.notifier.send(f"📅 Daily Report {date}: No trades")
            return
        df = pd.read_csv(self.cfg.trades_csv)
        if df.empty:
            msg = f"📅 Daily Report {date}\nNo trades"
        else:
            df['close_time'] = pd.to_datetime(df['close_time'])
            day_df = df[df['close_time'].dt.date == date]
            if day_df.empty:
                msg = f"📅 Daily Report {date}\nNo trades"
            else:
                total = day_df['pnl_usd'].sum()
                lines = [f"📅 Daily Report {date}", f"Total PnL: {total:.2f} USDT"]
                for sym, val in day_df.groupby('symbol')['pnl_usd'].sum().items():
                    lines.append(f"{sym}: {val:+.2f} USDT")
                msg = "\n".join(lines)
        self.notifier.send(msg)

    def _maybe_daily_report(self):
        today = now_utc().date()
        if today != self.last_daily_report_date:
            self._send_daily_report(self.last_daily_report_date)
            self.last_daily_report_date = today

    def load_model_state(self) -> None:
        """Load model-performance weights from ``state.json``.

        If the file does not exist or parsing fails, ``self.model_state`` is set
        to an empty dictionary and a warning is printed.  This method is
        typically called once at startup.
        """
        try:
            if os.path.exists(self.cfg.state_json):
                with open(self.cfg.state_json, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.model_state = data
                        return
        except Exception:
            print("[WARN] failed to load model_state; starting fresh")
        self.model_state = {}

    def save_model_state(self) -> None:
        """Persist the current :attr:`model_state` dictionary to disk."""
        try:
            ensure_dir(self.cfg.state_json)
            with open(self.cfg.state_json, "w") as f:
                json.dump(self.model_state, f)
        except Exception as e:
            print("[WARN] failed to save model_state:", e)

    def update_model_performance(self, model_name: str, pnl: float, entry: float, sl: float) -> None:
        """Update bandit weights after a trade closes.

        Parameters
        ----------
        model_name: str
            Identifier of the strategy/model.
        pnl: float
            Realized profit or loss in absolute terms.
        entry: float
            Trade entry price.
        sl: float
            Stop-loss price used to compute risk.

        Example
        -------
        >>> trade = ...  # some closed trade object
        >>> bot.update_model_performance(trade.model, trade.pnl_usd, trade.entry, trade.sl)
        """
        risk = abs(entry - sl)
        if risk <= 0:
            return
        R = pnl / risk
        st = self.model_state.setdefault(model_name, {"w": 1.0, "r_avg": 0.0, "n": 0})
        st["n"] += 1
        st["r_avg"] = 0.9 * st["r_avg"] + 0.1 * R
        decay = getattr(self.cfg, "evolve_decay", 0.95)
        trial_w = getattr(self.cfg, "evolve_trial_weight", 0.05)
        st["w"] = max(0.1, st["w"] * decay + trial_w * R)
        self.save_model_state()

    # ==========================================

    def can_alert_now(self) -> bool:
        return (time.time() - self.last_alert_ts) >= self.cfg.min_seconds_between_alerts_global

    def _committee(self, symbol: str, row: pd.Series, regime: Regime) -> Optional[Tuple[str, str, float, float]]:
        return sig_scalp(symbol, row, self.cfg)

    def run(self):
        self.notifier.send(f"[START] Evolving Scalper | TOP {self.cfg.top_n_symbols} | TF {self.cfg.timeframe} | RefEq={self.ref_equity:.2f} USDT")
        while True:
            try:
                self.loop_once()
                time.sleep(2)
            except KeyboardInterrupt:
                self.notifier.send("[EXIT] Stopping…"); break
            except Exception as e:
                self.notifier.send(f"[ERROR main] {e}"); time.sleep(3)

    def loop_once(self):
        # رولات اليوم
        self._maybe_daily_report()

        base_universe = self.ex.get_top_symbols(self.cfg.top_n_symbols)
        self.symbols = self.ex.filter_healthy(base_universe)

        # تحديث إغلاقات الصفقات (حتى أثناء الوقف/التهدئة)
        for symbol in self.symbols:
            try:
                df = self.ex.fetch_ohlcv_paged(symbol, self.cfg.timeframe, limit=self.cfg.lookback)
                if len(df) < self.cfg.lookback:
                    print(
                        f"[WARN] insufficient OHLCV for {symbol}: "
                        f"{len(df)} < {self.cfg.lookback}"
                    )
                    self.data_fail[symbol] = self.data_fail.get(symbol, 0) + 1
                    if self.data_fail[symbol] >= 2:
                        self.ex._bad_cache[symbol] = time.time()
                    continue
                self.data_fail[symbol] = 0
                d  = compute_indicators(df, self.cfg)
                if len(d) < 2:
                    print(
                        f"[WARN] insufficient indicator data for {symbol}: "
                        f"{len(d)} < 2"
                    )
                    continue

                last = d.iloc[-1]
                if self.cfg.debug_signals:
                    print(
                        f"[DBG][{symbol}] len={len(d)} last.close={last['close']:.6f} "
                        f"rsi={last['rsi']:.2f} bb_dn={last['bb_dn']:.6f} "
                        f"bb_up={last['bb_up']:.6f} atr={last['atr']:.6f}"
                    )
                closed = self.paper.update_with_candle(symbol, float(last["high"]), float(last["low"]), last.name)
                if closed:
                    for t in closed:
                        mkt = self.ex.x.market(t.symbol)
                        contract_size = float(mkt.get("contractSize") or 1)
                        contract_qty = t.qty / contract_size
                        contract_qty = float(self.ex.x.amount_to_precision(t.symbol, contract_qty))
                        self.ex.close_position(t.symbol, t.side, contract_qty)
                    reg_row = d.iloc[-2] if len(d)>1 else last
                    regime = classify_regime(reg_row, self.cfg)
                    ctx = ctx_key(regime)
                    self.paper.persist_closed(closed, self.cfg, ctx)

                    # تحديث المخاطر (streak + daily pnl)
                    pnl_sum = 0.0
                    sl_count = 0
                    for t in closed:
                        pnl_sum += float(t.pnl_usd or 0.0)
                        if t.result == "sl": sl_count += 1
                        emoji = "✅" if t.result=="tp" else "❌"
                        hold_s = int((pd.to_datetime(t.exit_time)-pd.to_datetime(t.timestamp)).total_seconds())
                        self.notifier.send(
                            f"📤 Trade Closed {emoji}\n"
                            f"• Pair: {t.symbol} | TF: {t.timeframe}\n"
                            f"• Side: {t.side.upper()} | Model: {t.model}\n"
                            f"• Entry: {t.entry:.4f} → Exit: {t.exit_price:.4f}\n"
                            f"• PnL: {t.pnl_usd:+.2f} USDT | Hold: {hold_s}s",
                        )
                        self.update_model_performance(t.model, float(t.pnl_usd or 0.0), t.entry, t.sl)
                        self.closed_trades.append(t)
            except Exception:
                continue

        # Force-close trades that have been open for more than 24h
        now = now_utc()
        max_age = dt.timedelta(hours=24)
        for tid, t in list(self.paper.open.items()):
            opened = pd.to_datetime(t.timestamp)
            if now - opened > max_age:
                price = self.ex.fetch_ticker_price(t.symbol)
                if price is None:
                    continue
                mkt = self.ex.x.market(t.symbol)
                contract_size = float(mkt.get("contractSize") or 1)
                contract_qty = t.qty / contract_size
                contract_qty = float(self.ex.x.amount_to_precision(t.symbol, contract_qty))
                self.ex.close_position(t.symbol, t.side, contract_qty)
                closed_t = self.paper.force_close(t, price, reason="timeout")
                self.paper.persist_closed([closed_t], self.cfg, "timeout")
                self.closed_trades.append(closed_t)
                self.notifier.send(f"[FORCE-CLOSE] {t.symbol} after 24h")

        self._maybe_hourly_report()

        # لو في صفقات مفتوحة — نكتفي بتتبع الإغلاق فقط
        if len(self.paper.open) >= self.cfg.max_open_trades:
            return

        # هدوء أحداث أو ثروتل
        if not self.can_alert_now(): return

        # البحث عن إشارة جديدة
        for symbol in self.symbols:
            try:

                df = self.ex.fetch_ohlcv_paged(symbol, self.cfg.timeframe, limit=self.cfg.lookback)
                if len(df) < self.cfg.lookback:
                    print(
                        f"[WARN] insufficient OHLCV for {symbol}: "
                        f"{len(df)} < {self.cfg.lookback}"
                    )
                    self.data_fail[symbol] = self.data_fail.get(symbol, 0) + 1
                    if self.data_fail[symbol] >= 2:
                        self.ex._bad_cache[symbol] = time.time()
                    continue
                self.data_fail[symbol] = 0
                d  = compute_indicators(df, self.cfg)
                if len(d) < 3:
                    print(
                        f"[WARN] insufficient indicator data for {symbol}: "
                        f"{len(d)} < 3"
                    )
                    continue

                row = d.iloc[-2]  # الشمعة المغلقة الأخيرة
                if self.cfg.debug_signals:
                    print(
                        f"[DBG][{symbol}] prev.close={row['close']:.6f} rsi={row['rsi']:.2f} "
                        f"in_band=({row['close']<=row['bb_dn']:.0f}/{row['close']>=row['bb_up']:.0f}) "
                        f"atr={row['atr']:.6f}"
                    )
                regime = classify_regime(row, self.cfg)

                # Determine if SCALP conditions are met
                sig_info = self._committee(symbol, row, regime)
                if not sig_info:
                    continue

                side, reason, quality, filter_scale = sig_info

                # Use live ticker price for entry to avoid stale candles
                price = self.ex.fetch_ticker_price(symbol)
                if price is None:
                    continue

                tp, sl = get_tp_sl(price, side, row, self.cfg)
                sig = Signal(side, sl, tp, "SCALP", reason)

                # مانع تكرار نفس الإشارة
                key = f"{symbol}:{self.cfg.timeframe}:{sig.model}:{sig.side}"
                if self.last_key.get(symbol) == key and self.last_time.get(symbol):
                    if (now_utc() - self.last_time[symbol]).total_seconds()/60.0 < self.cfg.min_minutes_between_same_signal:
                        continue

                bal = float(self.ex.get_balance_usdt())
                lev = max(1.0, float(self.cfg.leverage_x))
                max_slots = max(1, int(self.cfg.max_open_trades))

                total_capital = bal * float(self.cfg.capital_pct)
                open_trades = list(self.paper.open.values())
                open_cnt = _count_open_trades(open_trades)
                used_capital = _used_capital_usdt(open_trades)
                remaining_slots = max(1, max_slots - open_cnt)

                if self.cfg.equal_split_mode and not self.cfg.dynamic_split_mode:
                    per_trade_capital = total_capital / max_slots
                else:
                    rem_cap = max(0.0, total_capital - used_capital)
                    per_trade_capital = rem_cap / remaining_slots if remaining_slots > 0 else 0.0

                target_notional = max(0.0, per_trade_capital) * lev
                if filter_scale != 1.0:
                    target_notional *= filter_scale
                    if self.cfg.debug_signals:
                        print(f"[FILTER] {symbol} scale={filter_scale:.2f} target_notional={target_notional:.2f}")
                if self.cfg.quality_sizing:
                    q = max(0.0, min(1.0, quality))
                    lo_q = float(self.cfg.min_quality)
                    if q <= lo_q:
                        scale = float(self.cfg.size_scale_low)
                    else:
                        t = (q - lo_q) / (1.0 - lo_q)
                        scale = float(self.cfg.size_scale_low) + t * (float(self.cfg.size_scale_high) - float(self.cfg.size_scale_low))
                    target_notional *= max(0.0, scale)
                    if self.cfg.debug_signals:
                        print(f"[QUAL] {symbol} scale={scale:.2f} target_notional={target_notional:.2f}")
                target_notional = max(float(self.cfg.min_notional),
                                      min(float(self.cfg.max_notional), target_notional))

                mkt = self.ex.x.market(symbol)
                contract_size = float(mkt.get("contractSize") or 1.0)
                base_qty = target_notional / price
                contract_qty = base_qty / contract_size
                try:
                    contract_qty = float(self.ex.x.amount_to_precision(symbol, contract_qty))
                except Exception:
                    contract_qty = float(contract_qty)

                min_amt = float(((mkt.get("limits") or {}).get("amount") or {}).get("min") or 0.0)
                if contract_qty < max(1e-9, min_amt):
                    print(f"[WARN] {symbol} contract_qty < {max(1e-9, min_amt)} after precision — skipping")
                    continue

                notional = contract_qty * contract_size * float(price)
                req_margin = notional / lev
                if req_margin > bal:
                    print(f"[WARN] insufficient USDT margin: need {req_margin:.2f} > available {bal:.2f} — skipping {symbol}")
                    continue

                if self.cfg.debug_signals:
                    print(f"[RISK] {symbol} open={open_cnt}/{max_slots} rem={remaining_slots} used={used_capital:.2f} "
                          f"total={total_capital:.2f} per={per_trade_capital:.2f} q={quality:.2f} "
                          f"lev={lev:.1f} notional={notional:.2f} req_margin={req_margin:.2f} qty={contract_qty:.6f}")

                base_qty = contract_qty * contract_size
                notional_ref = notional
                risk = abs(price - sig.sl); reward = abs(sig.tp - price)
                rr = round(reward / risk, 2) if risk > 0 else None

                order = self.ex.create_order(symbol, sig.side, contract_qty)
                env = "OKX Demo" if self.cfg.okx_demo else "OKX Live"
                status_line = f"🚀 Executed on {env}" if order else f"⚠️ Execution failed on {env}"
                msg = (
                    f"📢 [SCALP] New Signal\n\n"
                    f"📍 Pair: {symbol}\n"
                    f"🕒 TF: {self.cfg.timeframe} | Ctx: trend={regime.trend}, vol={regime.vol_bucket}\n"
                    f"📈 Side: {sig.side.upper()} | Conf: {sig.confidence:.2f}\n\n"
                    f"💰 Entry: {price:.4f}\n"
                    f"🎯 TP: {sig.tp:.4f} ({'+' if sig.tp > price else ''}{pct((sig.tp-price)/price)})\n"
                    f"🛡 SL: {sig.sl:.4f} ({'-' if sig.sl < price else '+'}{pct(abs(sig.sl-price)/price)})\n"
                    f"📏 R:R = {rr if rr is not None else 'n/a'}\n\n"
                    f"🧠 Why: {sig.reason}\n"
                    f"📦 SizeRef: ~{base_qty:.6f} ({notional_ref:.2f} USDT)\n"
                    f"{status_line}"
                )
                self.notifier.send(msg)
                self.last_alert_ts = time.time()

                self.paper.log_signal(symbol, row, sig, base_qty, notional_ref, rr, self.cfg, regime)
                if not order:
                    self.last_key[symbol] = key
                    self.last_time[symbol] = now_utc()
                    continue

                t = self.paper.open_virtual(symbol, price, sig, base_qty, notional, self.cfg)
                self.paper.ml_snapshot(t.id, symbol, row, regime)

                self.last_key[symbol] = key
                self.last_time[symbol] = now_utc()
                break

            except Exception:
                continue

# =========================
# CLI
# =========================

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Evolving Committee Scalper (Alerts Only) — No OpenAI")
    p.add_argument("--timeframe", default="30m")
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--no-debug", action="store_true")            # لإطفاء الديباج عند الحاجة
    p.add_argument("--no-funding-filter", action="store_true")   # لتعطيل فلتر الفاندنغ مؤقتًا
    p.add_argument("--quiet", nargs="*", default=None, help="UTC HH:MM times to avoid (e.g., 12:30 18:00)")
    p.add_argument("--top", type=int, default=None, help="Top N USDT perpetuals to scan (override config)")
    p.add_argument("--live", action="store_true", help="Use real trading environment instead of OKX demo")

    # === CLI overrides for capital/leverage tool ===
    p.add_argument("--leverage-x", type=float, default=None)   # تستطيع تغييره من CLI عند الحاجة
    p.add_argument("--capital-pct", type=float, default=None)  # تستطيع تغييره من CLI عند الحاجة
    p.add_argument("--max-open-trades", type=int, default=None, help="max concurrent trades (e.g., 3)")
    p.add_argument("--equal-split", action="store_true", help="use equal split: total/slots")
    p.add_argument("--dynamic-split", action="store_true", help="use dynamic split of remaining capital")
    p.add_argument("--min-notional", type=float, default=None)
    p.add_argument("--max-notional", type=float, default=None)
    # ---- CLI: فلاتر/جودة ----
    p.add_argument("--filters-off", action="store_true")
    p.add_argument("--no-trend-filter", action="store_true")
    p.add_argument("--min-atr-pct", type=float, default=None)
    p.add_argument("--min-vol-z", type=float, default=None)
    p.add_argument("--min-vol-pct-alt", type=float, default=None, help="volume percentile alt threshold (0..1)")
    p.add_argument("--filter-mode", type=str, choices=["strict","any","kofn","soft"], default=None)
    p.add_argument("--filters-k", type=int, default=None)
    p.add_argument("--soft-scale-low", type=float, default=None)
    p.add_argument("--soft-scale-high", type=float, default=None)
    p.add_argument("--quality-sizing-off", action="store_true")
    p.add_argument("--min-quality", type=float, default=None)
    p.add_argument("--size-scale-low", type=float, default=None)
    p.add_argument("--size-scale-high", type=float, default=None)
    args = p.parse_args()
    cfg = Config()
    cfg.timeframe = args.timeframe
    cfg.okx_demo = not args.live
    if args.lookback is not None:
        cfg.lookback = int(args.lookback)
    if args.no_debug:
        cfg.debug_signals = False
    if args.no_funding_filter:
        cfg.funding_filter = False
    # apply CLI overrides
    if args.leverage_x is not None:       cfg.leverage_x = float(args.leverage_x)
    if args.capital_pct is not None:      cfg.capital_pct = float(args.capital_pct)
    if args.max_open_trades is not None:  cfg.max_open_trades = int(args.max_open_trades)
    if args.equal_split:
        cfg.equal_split_mode = True
        cfg.dynamic_split_mode = False
    if args.dynamic_split:
        cfg.dynamic_split_mode = True
    if args.min_notional is not None:     cfg.min_notional = float(args.min_notional)
    if args.max_notional is not None:     cfg.max_notional = float(args.max_notional)
    # تطبيق CLI: فلاتر/جودة
    if args.filters_off:                  cfg.filters_enabled = False
    if args.no_trend_filter:              cfg.trend_filter = False
    if args.min_atr_pct is not None:      cfg.min_atr_pct = float(args.min_atr_pct)
    if args.min_vol_z is not None:        cfg.min_vol_z = float(args.min_vol_z)
    if args.min_vol_pct_alt is not None:  cfg.min_vol_pct_alt = float(args.min_vol_pct_alt)
    if args.filter_mode is not None:      cfg.filter_mode = str(args.filter_mode)
    if args.filters_k is not None:        cfg.filters_k = int(args.filters_k)
    if args.soft_scale_low is not None:   cfg.soft_scale_low = float(args.soft_scale_low)
    if args.soft_scale_high is not None:  cfg.soft_scale_high = float(args.soft_scale_high)
    if args.quality_sizing_off:           cfg.quality_sizing = False
    if args.min_quality is not None:      cfg.min_quality = float(args.min_quality)
    if args.size_scale_low is not None:   cfg.size_scale_low = float(args.size_scale_low)
    if args.size_scale_high is not None:  cfg.size_scale_high = float(args.size_scale_high)
    # اسمح بتجاوز الإعدادات من سطر الأوامر
    if args.top is not None:
        cfg.top_n_symbols = int(args.top)
    if args.quiet is not None:
        cfg.quiet_windows_utc = tuple(args.quiet)
    ensure_dir(cfg.logs_dir)
    ensure_dir(cfg.signals_csv); ensure_dir(cfg.trades_csv); ensure_dir(cfg.ml_csv); ensure_dir(cfg.models_csv)
    return cfg

def main():
    cfg = parse_args()
    # إخفاء توكن التيليجرام في الطباعة
    _cfg = asdict(cfg)
    if _cfg.get("telegram_token"):
        _tok = _cfg["telegram_token"]
        _cfg["telegram_token"] = (_tok[:4] + "…" + _tok[-4:]) if len(_tok) > 8 else "****"
    print("Config:\n", json.dumps(_cfg, indent=2, default=str))
    bot = Bot(cfg)
    bot.run()

if __name__ == "__main__":
    main()