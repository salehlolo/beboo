import os, ccxt
from dotenv import load_dotenv

load_dotenv()  # يحمّل مفاتيح .env


def make_okx(demo: bool = True,
             leverage: float = 10,
             margin_mode: str = "cross",   # "cross" أو "isolated"
             pos_mode: str = "net"         # "net" أو "long_short"
             ) -> ccxt.okx:
    """
    يُرجع كائن OKX مضبوطاً لسواب USDT.
    - demo=True يفعّل بيئة الديمو + x-simulated-trading: 1
    - يضبط وضع الحساب (NET أو long/short) والرافعة.
    """
    key = os.getenv("OKX_API_KEY")
    secret = os.getenv("OKX_API_SECRET")
    password = os.getenv("OKX_API_PASSWORD") or os.getenv("OKX_API_PASSPHRASE")

    headers = {"x-simulated-trading": "1"} if demo else {}
    ex = ccxt.okx({
        "apiKey": key,
        "secret": secret,
        "password": password,
        "enableRateLimit": True,
        "timeout": 15000,
        "options": {"defaultType": "swap", "demo": demo},
        "headers": headers,
    })

    # تفعيل الديمو (ساندبوكس)
    if demo:
        try:
            ex.set_sandbox_mode(True)
        except Exception:
            pass

    # تضبيط أوضاع الحساب + الرافعة
    try:
        hedged = (pos_mode != "net")
        try:
            ex.set_position_mode(hedged)  # True = long/short, False = NET
        except Exception:
            ex.privatePostAccountSetPositionMode({
                "posMode": "long_short_mode" if hedged else "net_mode"
            })

        ref_sym = "BTC/USDT:USDT"
        try:
            ex.set_leverage(leverage, ref_sym, {"mgnMode": margin_mode})
        except Exception:
            m = ex.market(ref_sym)
            ex.privatePostAccountSetLeverage({
                "instId": m["id"],
                "lever": str(leverage),
                "mgnMode": margin_mode,
            })
    except Exception as e:
        print("[WARN] mode/leverage setup:", e)

    ex.load_markets()
    return ex


if __name__ == "__main__":
    okx = make_okx(demo=True)  # غيّر إلى False للحساب الحقيقي
    # اختبار سريع:
    bal = okx.fetch_balance(params={"type": "swap"})
    print("USDT swap free:", (bal.get("free", {}) or {}).get("USDT"))

    t = okx.fetch_ticker("ETH/USDT:USDT")
    print("ETH last:", t.get("last") or t.get("close"))

    # مثال أمر سوق (NET mode: posSide اختياري؛ في long/short لازم posSide)
    order = okx.create_order(
        "ETH/USDT:USDT", "market", "buy", 1.0,
        params={"tdMode": "cross", "posSide": "net"}
    )
    print("OrderId:", order.get("id") or order.get("info", {}).get("ordId"))
