# =============================================================================
#  vp_gex_core.py - motore di calcolo Volume Profile + GEX
# -----------------------------------------------------------------------------
#  Usato da:
#    - vp_gex_multi.py : script giornaliero multi-strumento con invio su Telegram
#    - vp_gex_app.py   : app Streamlit per analizzare un singolo strumento
#  Questo file NON contiene token o chiavi: la chiave Massive si imposta con
#  set_api_key() oppure con la variabile d'ambiente MASSIVE_API_KEY.
# -----------------------------------------------------------------------------
#  - Volume profile: yfinance, serie continua NON aggiustata per i future (prezzi
#    realmente scambiati, come ES1!/NQ1! su TradingView), barre 1h (fallback daily).
#    Metodo del Volume Profile Analyzer (200 righe) con volume di ogni barra diviso
#    tra le righe toccate e POC al centro del plateau in caso di pari merito.
#  - GEX: catene opzioni da Massive (API), scadenze scelte in base all'orario di NY:
#       * prima dell'apertura e durante la sessione cash -> scadenza di oggi
#       * dopo le 16:00 ET, weekend e festivita NYSE -> giorno di borsa successivo
#    (se in quella data non c'e' una scadenza si usa la prima disponibile dopo).
#  - Due grafici per strumento: (A) scadenza piu prossima, (B) 7 scadenze pesate
#    1/sqrt(DTE+1).
#  - Gamma magnet: strike con |net GEX| massimo entro una finestra attorno allo
#    spot; + = pinning, - = accelerazione.
#  - Tutti i testi delle immagini sono in inglese.
# =============================================================================

import os
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  # nessuna finestra: adatto a Task Scheduler e Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm

# =============================
#  CONFIG
# =============================
MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY", "")
MASSIVE_BASE_URL = "https://api.massive.com"

VP_DAYS = 62               # sedute nel volume profile (default)
VP_ROWS = 200              # righe del profilo (come nel Volume Profile Analyzer)
VP_USE_HOURLY = True       # True: barre 1h; False (o orarie non disponibili): barre daily
VP_MIN_BARS_PER_SESSION = 4  # sotto questa media di barre orarie per seduta -> barre daily
VALUE_AREA_PCT = 0.70

N_EXP_SHORT = 1            # grafico A: solo la scadenza piu prossima
N_EXP_LONG = 7             # grafico B: 7 scadenze
TOP_N_STRIKES = 8          # strike con |GEX| maggiore da etichettare
GEX_BAR_ALPHA = 0.45       # trasparenza barre GEX (0 = invisibili, 1 = piene)
FLIP_GRID_POINTS = 401     # punti della griglia per il gamma flip

ET = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)


def set_api_key(key):
    global MASSIVE_API_KEY
    MASSIVE_API_KEY = key


# =============================
#  STRUMENTI
# =============================
#   name            nome mostrato sul grafico
#   vp_ticker       ticker yfinance per il volume profile
#   under           nome del sottostante delle opzioni
#   opt_underlying  sottostante per lo snapshot opzioni di Massive (indici con prefisso I:)
#   under_yf        ticker yfinance del sottostante (chiusure per il basis)
#   is_future       True: strike traslati con il basis future - indice
#   pm_prefixes     prefissi dei ticker opzione PM-settled (scadenza 16:00); gli altri
#                   scadono all'apertura (AM-settled). None: tutte alle 16:00 (azioni/ETF)
#   strike_pct      strike considerati entro +/- questa quota dallo spot
#   magnet_short    finestra gamma magnet, grafico scadenza piu prossima
#   magnet_long     finestra gamma magnet, grafico 7 scadenze
#   exp_search_days giorni di calendario in cui cercare le scadenze
FUTURE_ES = dict(name="ES", vp_ticker="ES=F", under="SPX", opt_underlying="I:SPX", under_yf="^GSPC",
                 is_future=True, pm_prefixes=("O:SPXW",), strike_pct=0.05,
                 magnet_short=0.01, magnet_long=0.02, exp_search_days=21)
FUTURE_NQ = dict(name="NQ", vp_ticker="NQ=F", under="NDX", opt_underlying="I:NDX", under_yf="^NDX",
                 is_future=True, pm_prefixes=("O:NDXP",), strike_pct=0.05,
                 magnet_short=0.01, magnet_long=0.02, exp_search_days=21)
ETF_DEFAULTS = dict(strike_pct=0.05, magnet_short=0.015, magnet_long=0.03, exp_search_days=60)
STOCK_DEFAULTS = dict(strike_pct=0.10, magnet_short=0.02, magnet_long=0.04, exp_search_days=60)


def make_instrument(kind, ticker=None, **overrides):
    """
    kind: "ES", "NQ", "ETF" oppure "STOCK".
    Per ETF e azioni il ticker e' lo stesso per volume profile e opzioni (es. GLD, NVDA).
    """
    kind = kind.upper()
    if kind == "ES":
        inst = dict(FUTURE_ES)
    elif kind == "NQ":
        inst = dict(FUTURE_NQ)
    elif kind in ("ETF", "STOCK"):
        if not ticker:
            raise ValueError("serve il ticker per ETF e azioni")
        t = ticker.strip().upper()
        inst = dict(name=t, vp_ticker=t, under=t, opt_underlying=t, under_yf=t,
                    is_future=False, pm_prefixes=None,
                    **(ETF_DEFAULTS if kind == "ETF" else STOCK_DEFAULTS))
    else:
        raise ValueError(f"tipo di strumento sconosciuto: {kind}")
    inst.update({k: v for k, v in overrides.items() if v is not None})
    return inst

# =============================
#  CALENDARIO NYSE (algoritmico)
# =============================
def _easter(year):
    """Domenica di Pasqua (calendario gregoriano, algoritmo anonimo)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year, month, weekday, n):
    """n-esimo giorno della settimana del mese (weekday: lun=0). n=-1 -> ultimo."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d):
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year):
    hol = set()
    ny = date(year, 1, 1)
    if ny.weekday() != 5:          # se Capodanno cade di sabato la NYSE non recupera
        hol.add(_observed(ny))
    hol.add(_nth_weekday(year, 1, 0, 3))    # Martin Luther King
    hol.add(_nth_weekday(year, 2, 0, 3))    # Presidents Day
    hol.add(_easter(year) - timedelta(days=2))  # Good Friday
    hol.add(_nth_weekday(year, 5, 0, -1))   # Memorial Day
    if year >= 2022:
        hol.add(_observed(date(year, 6, 19)))   # Juneteenth
    hol.add(_observed(date(year, 7, 4)))    # Independence Day
    hol.add(_nth_weekday(year, 9, 0, 1))    # Labor Day
    hol.add(_nth_weekday(year, 11, 3, 4))   # Thanksgiving
    hol.add(_observed(date(year, 12, 25)))  # Natale
    return hol


def is_trading_day(d):
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def next_trading_day(d):
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def prev_trading_day(d):
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def trading_days_between(d0, d1):
    """Numero di giorni di borsa in (d0, d1]. 0 se d1 <= d0."""
    n, d = 0, d0
    while d < d1:
        d = next_trading_day(d)
        n += 1
    return n


def market_context(now_et):
    """
    Ritorna (stato, data_riferimento_scadenze, ultima_sessione_chiusa_o_in_corso).
    Le mezze sedute non sono gestite.
    """
    today = now_et.date()
    t = now_et.time()
    if is_trading_day(today):
        if t < SESSION_OPEN:
            return "pre-market", today, prev_trading_day(today)
        if t < SESSION_CLOSE:
            return "cash session open", today, today
        return "after the close", next_trading_day(today), today
    return "market closed", next_trading_day(today), prev_trading_day(today)


# =============================
#  DATI YFINANCE
# =============================
def _to_et(h):
    idx = pd.to_datetime(h.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    h.index = idx.tz_convert(ET)
    return h[~h.index.duplicated(keep="last")]


def load_daily(ticker, period="1y"):
    h = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if h is None or h.empty or "Close" not in h:
        raise ValueError(f"nessun dato su yfinance per il ticker '{ticker}'")
    h = h[["Open", "High", "Low", "Close", "Volume"]].copy()
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    h = h[~h.index.duplicated(keep="last")]
    return h


def load_intraday(ticker):
    """Barre 1m degli ultimi giorni, comprese le fasce pre/after market per le azioni."""
    h = yf.Ticker(ticker).history(period="5d", interval="1m", prepost=True)
    return h if h.empty else _to_et(h)


def load_live_yf(ticker):
    h = yf.Ticker(ticker).history(period="1d", interval="1m")
    return float(h["Close"].iloc[-1]) if not h.empty else np.nan


def load_hourly(ticker):
    """Barre 1h (yfinance le fornisce fino a 730 giorni); per le azioni solo sessione regolare."""
    try:
        h = yf.Ticker(ticker).history(period="6mo", interval="1h")
    except Exception as e:
        print(f"[yfinance] barre orarie {ticker} non disponibili ({e})")
        return pd.DataFrame()
    if h.empty:
        return h
    return _to_et(h[["Open", "High", "Low", "Close", "Volume"]].copy())


def add_session_date(h):
    """
    Sessione CME: 18:00 ET del giorno prima -> 17:00 ET.
    Spostando l'orario di +6h, ogni barra cade nella data della sua sessione
    (la domenica sera finisce nel lunedi). Per le azioni (9:30-16:00) resta la data.
    """
    h = h.copy()
    h["session"] = pd.to_datetime((h.index + pd.Timedelta(hours=6)).date)
    return h


# =============================
#  VOLUME PROFILE
# =============================
def build_volume_profile(bars, rows):
    """
    Metodo del Volume Profile Analyzer (logica Pine Script) con una correzione:
    il range high-low del periodo e' diviso in `rows` righe uguali e il volume di
    ogni barra e' ripartito IN PARTI UGUALI tra le righe che il suo range attraversa
    (cosi' ogni barra pesa per il suo volume e non per l'ampiezza del suo range).
    E' una stima OHLCV-based, non un volume-at-price tick-by-tick.
    """
    lowest = bars["Low"].min()
    highest = bars["High"].max()
    edges = np.linspace(lowest, highest, rows + 1)

    highs = bars["High"].to_numpy()[:, None]
    lows = bars["Low"].to_numpy()[:, None]
    vols = bars["Volume"].to_numpy(dtype=float)

    mask = (highs > edges[:-1][None, :]) & (lows < edges[1:][None, :])
    n_rows = mask.sum(axis=1)
    # barre con high == low esattamente su un bordo: volume alla riga della chiusura
    flat = n_rows == 0
    if flat.any():
        i = np.clip(np.searchsorted(edges, bars["Close"].to_numpy()[flat]) - 1, 0, rows - 1)
        mask[np.where(flat)[0], i] = True
        n_rows = mask.sum(axis=1)
    vol = (mask * (vols / n_rows)[:, None]).sum(axis=0)
    return edges, vol


def poc_index(vol):
    """Riga del POC; con pari merito sulle righe contigue prende il centro del plateau."""
    i = int(np.argmax(vol))
    vmax = vol[i]
    lo = hi = i
    while lo - 1 >= 0 and np.isclose(vol[lo - 1], vmax, rtol=1e-9):
        lo -= 1
    while hi + 1 < len(vol) and np.isclose(vol[hi + 1], vmax, rtol=1e-9):
        hi += 1
    return lo, hi


def value_area(edges, vol, pct):
    n = len(vol)
    plat_lo, plat_hi = poc_index(vol)
    total = vol.sum()
    lo, hi = plat_lo, plat_hi
    acc = vol[lo:hi + 1].sum()
    while acc < pct * total:
        up = vol[hi + 1] if hi + 1 < n else -1
        dn = vol[lo - 1] if lo - 1 >= 0 else -1
        if up < 0 and dn < 0:
            break
        if up >= dn:
            hi += 1
            acc += up
        else:
            lo -= 1
            acc += dn
    poc = (edges[plat_lo] + edges[plat_hi + 1]) / 2
    return poc, edges[lo], edges[hi + 1]


def volume_profile_for(inst, vp_days=None):
    """Profilo delle ultime vp_days sedute: barre 1h se disponibili, altrimenti daily."""
    daily = load_daily(inst["vp_ticker"])
    daily = daily[daily["Volume"] > 0]
    sessions = daily.index[-(vp_days or VP_DAYS):]

    bars, src = None, ""
    if VP_USE_HOURLY:
        hourly = load_hourly(inst["vp_ticker"])
        if not hourly.empty:
            hourly = add_session_date(hourly)
            hourly = hourly[(hourly["Volume"] > 0) & hourly["session"].isin(sessions)]
            if len(hourly) >= VP_MIN_BARS_PER_SESSION * len(sessions):
                bars, src = hourly, f"1h bars ({len(hourly)})"
            else:
                print(f"[VP {inst['name']}] solo {len(hourly)} barre orarie, uso le daily")
    if bars is None:
        bars = daily.loc[sessions]
        src = "daily bars" + (" (hourly unavailable)" if VP_USE_HOURLY else "")

    edges, vol = build_volume_profile(bars, VP_ROWS)
    poc, val, vah = value_area(edges, vol, VALUE_AREA_PCT)
    return daily, sessions, (edges, vol, poc, val, vah), src


# =============================
#  MASSIVE
# =============================
def massive_index_live(ticker):
    """Valore live di un indice dallo snapshot di Massive (serve il piano Indices)."""
    try:
        r = requests.get(f"{MASSIVE_BASE_URL}/v3/snapshot",
                         params={"ticker.any_of": ticker, "apiKey": MASSIVE_API_KEY},
                         timeout=20)
        r.raise_for_status()
        res = r.json().get("results", [])
        if res and res[0].get("value"):
            return float(res[0]["value"])
    except Exception as e:
        print(f"[Massive] {ticker} live non disponibile ({e}), uso yfinance")
    return np.nan


def fetch_chain(inst, spot, ref_date):
    if not MASSIVE_API_KEY:
        raise RuntimeError("chiave API Massive non impostata (MASSIVE_API_KEY)")
    url = f"{MASSIVE_BASE_URL}/v3/snapshot/options/{inst['opt_underlying']}"
    pct = inst["strike_pct"]
    params = {
        "expiration_date.gte": ref_date.isoformat(),
        "expiration_date.lte": (ref_date + timedelta(days=inst["exp_search_days"])).isoformat(),
        "strike_price.gte": round(spot * (1 - pct), 2),
        "strike_price.lte": round(spot * (1 + pct), 2),
        "limit": 250,
        "apiKey": MASSIVE_API_KEY,
    }
    rows = []
    while url:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        for res in js.get("results", []):
            det = res.get("details", {}) or {}
            gk = res.get("greeks", {}) or {}
            rows.append({
                "ticker": det.get("ticker", ""),
                "type": det.get("contract_type"),
                "expiration": det.get("expiration_date"),
                "strike": det.get("strike_price"),
                "gamma_massive": gk.get("gamma", np.nan),
                "iv": res.get("implied_volatility", np.nan),
                "oi": res.get("open_interest", 0) or 0,
            })
        url = js.get("next_url")
        params = {"apiKey": MASSIVE_API_KEY}
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Massive non ha restituito opzioni per {inst['opt_underlying']} "
                           f"(strumento senza opzioni quotate o non incluso nel piano)")
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    df["strike"] = df["strike"].astype(float)
    return df


def prepare_chain(df, inst, now_et):
    df = df.copy()
    if inst["pm_prefixes"]:
        # indici: PM-settled alle 16:00, mensili AM-settled all'apertura
        is_pm = df["ticker"].str.startswith(tuple(inst["pm_prefixes"]))
        exp_time = np.where(is_pm, "16:00", "09:30")
    else:
        exp_time = np.full(len(df), "16:00")
    df["expiry_dt"] = [
        datetime.combine(e, time.fromisoformat(t), tzinfo=ET)
        for e, t in zip(df["expiration"], exp_time)
    ]
    df = df[df["expiry_dt"] > now_et]
    df = df[df["oi"] > 0]
    return df


# =============================
#  GEX
# =============================
T_MIN_YEARS = 15 / (365 * 24 * 60)   # minimo 15 minuti a scadenza


def bs_gamma(S, K, T, iv):
    S = np.asarray(S, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + 0.5 * iv ** 2 * T) / (iv * np.sqrt(T))
        g = norm.pdf(d1) / (S * iv * np.sqrt(T))
    return np.nan_to_num(g)


def compute_gex(chain, spot, now_et, ref_date, weighted, magnet_window, grid_pct):
    df = chain.copy()
    df["T"] = [max((e - now_et).total_seconds() / (365 * 86400), T_MIN_YEARS)
               for e in df["expiry_dt"]]
    has_iv = df["iv"].notna() & (df["iv"] > 0)
    df["gamma"] = df["gamma_massive"]
    df.loc[has_iv, "gamma"] = bs_gamma(spot, df.loc[has_iv, "strike"],
                                       df.loc[has_iv, "T"], df.loc[has_iv, "iv"])
    df = df[df["gamma"].notna()]

    df["dte"] = [trading_days_between(ref_date, e) for e in df["expiration"]]
    df["w"] = 1 / np.sqrt(df["dte"] + 1) if weighted else 1.0
    df["sign"] = np.where(df["type"] == "call", 1.0, -1.0)
    # $ di gamma per 1% di movimento dello spot (convenzione dealer long call / short put)
    df["gex"] = df["sign"] * df["w"] * df["gamma"] * df["oi"] * 100 * spot ** 2 * 0.01

    by_strike = df.pivot_table(index="strike", columns="type", values="gex",
                               aggfunc="sum", fill_value=0.0)
    for col in ("call", "put"):
        if col not in by_strike:
            by_strike[col] = 0.0
    by_strike["net"] = by_strike["call"] + by_strike["put"]
    by_strike = by_strike.sort_index()

    # Gamma flip: zero del GEX totale ricalcolato su una griglia di spot ipotetici
    flip = np.nan
    iv_rows = df[df["iv"].notna() & (df["iv"] > 0)]
    if not iv_rows.empty:
        grid = spot * (1 + np.linspace(-grid_pct, grid_pct, FLIP_GRID_POINTS))
        K = iv_rows["strike"].values
        T = iv_rows["T"].values
        iv = iv_rows["iv"].values
        coef = (iv_rows["sign"] * iv_rows["w"] * iv_rows["oi"] * 100).values
        G = bs_gamma(grid[:, None], K[None, :], T[None, :], iv[None, :])
        total = (G * coef[None, :]).sum(axis=1) * grid ** 2 * 0.01
        sc = np.where(np.diff(np.sign(total)) != 0)[0]
        if len(sc):
            cands = [grid[i] - total[i] * (grid[i + 1] - grid[i]) / (total[i + 1] - total[i])
                     for i in sc]
            flip = float(min(cands, key=lambda x: abs(x - spot)))

    call_wall = float(by_strike["call"].idxmax()) if by_strike["call"].max() > 0 else np.nan
    put_wall = float(by_strike["put"].idxmin()) if by_strike["put"].min() < 0 else np.nan
    # Gamma magnet: strike con |net GEX| massimo entro la finestra attorno allo spot.
    # Segno +: dealer long gamma -> pinning. Segno -: dealer short gamma -> accelerazione.
    near = by_strike[np.abs(by_strike.index.values / spot - 1) <= magnet_window]
    if near.empty or (near["net"] == 0).all():
        magnet = (np.nan, 0.0)
    else:
        k = float(near["net"].abs().idxmax())
        magnet = (k, float(np.sign(by_strike.loc[k, "net"])))

    # Regime: segno del GEX totale allo spot attuale (sopra il flip -> positivo)
    regime = "positive gamma" if by_strike["net"].sum() > 0 else "negative gamma"
    return by_strike, flip, call_wall, put_wall, magnet, regime


# =============================
#  FORMATTAZIONE
# =============================
def gex_unit(values):
    """Unita' di misura del GEX in base all'ordine di grandezza: miliardi o milioni."""
    m = np.abs(values).max() if len(values) else 0.0
    return (1e9, "bn") if m >= 1e9 else (1e6, "mn")


def fmt_strike(k):
    return f"{k:.0f}" if abs(k - round(k)) < 1e-9 else f"{k:.1f}"


def fmt_level(k, ctx):
    """Livello GEX: 'SPX 6700 | ES 6725' per i future, solo il prezzo per azioni/ETF."""
    if ctx["is_future"]:
        return f"{ctx['under']} {k:.0f} | {ctx['name']} {k + ctx['basis']:.0f}"
    return fmt_strike(k)


# =============================
#  GRAFICO
# =============================
def _spread_labels(ys, min_gap):
    """Sposta le etichette per non sovrapporle (ys ordinati per valore)."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for j in range(1, len(order)):
        prev, cur = order[j - 1], order[j]
        if out[cur] - out[prev] < min_gap:
            out[cur] = out[prev] + min_gap
    return out


def make_chart(vp, gex, ctx, label, exps):
    edges, vol, poc, val, vah = vp
    by_strike, flip, call_wall, put_wall, magnet, regime = gex
    basis = ctx["basis"]
    name, under = ctx["name"], ctx["under"]

    fig, ax = plt.subplots(figsize=(13, 11))
    fig.subplots_adjust(left=0.08, right=0.74, top=0.89, bottom=0.15)

    # --- Volume profile
    mids = (edges[:-1] + edges[1:]) / 2
    row_h = edges[1] - edges[0]
    vnorm = vol / vol.max()
    in_va = (mids >= val) & (mids <= vah)
    ax.barh(mids[~in_va], vnorm[~in_va], height=row_h * 0.9,
            color="steelblue", alpha=0.25)
    ax.barh(mids[in_va], vnorm[in_va], height=row_h * 0.9,
            color="steelblue", alpha=0.50)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"Normalized volume ({ctx['vp_days']} sessions, {ctx['vp_src']}, {VP_ROWS} rows)")
    ax.set_ylabel(f"{name} price")

    # --- GEX (asse x superiore)
    ax2 = ax.twiny()
    strikes_px = by_strike.index.values + basis
    div, unit = gex_unit(by_strike["net"].values)
    g = by_strike["net"].values / div
    gmax = np.abs(g).max() if len(g) and np.abs(g).max() > 0 else 1.0
    ax2.set_xlim(-gmax * 1.25, gmax * 1.25)
    step = np.median(np.diff(by_strike.index.values)) if len(by_strike) > 1 else row_h
    ax2.barh(strikes_px, g, height=step * 0.7,
             color=np.where(g >= 0, "green", "red"), alpha=GEX_BAR_ALPHA,
             edgecolor=np.where(g >= 0, "darkgreen", "darkred"), linewidth=0.6)
    ax2.axvline(0, color="gray", linewidth=0.8)
    ax2.set_xlabel(f"{under} net GEX ($ {unit} per 1% move)")

    # Etichette sugli strike con |GEX| maggiore (scarta quelle troppo vicine fra loro)
    y_span = (max(edges[-1], strikes_px.max()) - min(edges[0], strikes_px.min())
              if len(strikes_px) else edges[-1] - edges[0])
    min_gap = y_span * 0.02
    top = []
    for k in by_strike["net"].abs().sort_values(ascending=False).index:
        if all(abs(k - t) >= min_gap for t in top):
            top.append(k)
        if len(top) == TOP_N_STRIKES:
            break
    for k in top:
        v = by_strike.loc[k, "net"] / div
        txt = (f"{k:.0f} | {k + basis:.0f}" if ctx["is_future"] else fmt_strike(k))
        ax2.text(v + np.sign(v) * gmax * 0.02, k + basis, txt, fontsize=7.5,
                 ha="left" if v >= 0 else "right", va="center",
                 color="darkgreen" if v >= 0 else "darkred")

    # --- Range verticale
    y_lo = min(edges[0], strikes_px.min() if len(strikes_px) else edges[0])
    y_hi = max(edges[-1], strikes_px.max() if len(strikes_px) else edges[-1])
    pad = (y_hi - y_lo) * 0.02
    ax.set_ylim(y_lo - pad, y_hi + pad)

    # --- Livelli
    levels = [
        (f"Last {name} {ctx['last']:.2f}", ctx["last"], "black", "-", 2.0),
        (f"POC {poc:.2f}", poc, "darkorange", "-", 2.0),
        (f"VAH {vah:.2f}", vah, "dimgray", "--", 1.2),
        (f"VAL {val:.2f}", val, "dimgray", "--", 1.2),
    ]
    if not np.isnan(flip):
        flip_txt = (f"{under} {flip:.0f} | {name} {flip + basis:.0f}" if ctx["is_future"]
                    else f"{flip:.2f}")
        levels.append((f"Gamma flip {flip_txt}", flip + basis, "purple", "--", 2.0))
    if not np.isnan(call_wall):
        levels.append((f"Call wall {fmt_level(call_wall, ctx)}",
                       call_wall + basis, "green", ":", 2.2))
    if not np.isnan(put_wall):
        levels.append((f"Put wall {fmt_level(put_wall, ctx)}",
                       put_wall + basis, "red", ":", 2.2))
    mk, msign = magnet
    if not np.isnan(mk):
        tag = "+, pin" if msign > 0 else "-, accelerator"
        levels.append((f"Gamma magnet ({tag}) {fmt_level(mk, ctx)}",
                       mk + basis, "darkgreen" if msign > 0 else "darkred", "-.", 2.6))

    for _, y, c, ls, lw in levels:
        ax.axhline(y, color=c, linestyle=ls, linewidth=lw, alpha=0.9)

    ys = [l[1] for l in levels]
    ylab = _spread_labels(ys, (y_hi - y_lo + 2 * pad) * 0.032)
    tr = ax.get_yaxis_transform()
    for (txt, y, c, _, _), yl in zip(levels, ylab):
        ax.annotate(txt, xy=(1.0, y), xycoords=tr, xytext=(1.02, yl), textcoords=tr,
                    fontsize=9, color=c, va="center", annotation_clip=False,
                    bbox=dict(facecolor="white", edgecolor=c, alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8))

    # --- Legenda
    handles = [
        mpatches.Patch(color="steelblue", alpha=0.5, label="Volume profile (value area)"),
        mpatches.Patch(color="steelblue", alpha=0.25, label="Volume profile (outside VA)"),
        mpatches.Patch(facecolor="green", edgecolor="darkgreen", alpha=GEX_BAR_ALPHA, label="Net GEX > 0"),
        mpatches.Patch(facecolor="red", edgecolor="darkred", alpha=GEX_BAR_ALPHA, label="Net GEX < 0"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    # --- Titolo e riquadro informativo
    exp_txt = ", ".join(e.strftime("%b %d") for e in exps)
    gex_name = f"{under} GEX" if ctx["is_future"] else "GEX"
    ax.set_title(f"{name} - {ctx['vp_days']}-session Volume Profile + {gex_name} ({label})\n"
                 f"Expiries: {exp_txt}   |   {ctx['now'].strftime('%Y-%m-%d %H:%M')} ET",
                 fontsize=12, pad=38)
    if ctx["is_future"]:
        line2 = (f"Basis {name}-{under}: {basis:+.2f} ({ctx['basis_src']})  |  ")
        series = f"{ctx['vp_ticker']} continuous, unadjusted  |  Strike labels: {under} | {name}"
    else:
        line2 = ""
        series = f"{ctx['vp_ticker']}  |  Spot: {ctx['spot_src']}"
    info = (f"Market state: {ctx['state']}  |  Reference expiry: {ctx['ref_date'].strftime('%Y-%m-%d')}  |  "
            f"{under} spot used for GEX: {ctx['spot']:.2f}"
            + (f" ({ctx['spot_src']})" if ctx["is_future"] else "") + "\n"
            + line2
            + f"Total net GEX: {by_strike['net'].sum() / div:+.2f} $ {unit}/1%  |  Gamma regime: {regime}  |  "
            f"Magnet window: +/-{ctx['magnet_window'] * 100:g}% of spot\n"
            f"Price series: {series}")
    fig.text(0.08, 0.02, info, fontsize=8.5, family="monospace", va="bottom")
    return fig


# =============================
#  ANALISI DI UNO STRUMENTO
# =============================
def spot_and_basis(inst, daily, state, last_session):
    """
    Future: basis future - indice su prezzi sincronizzati (live in sessione, chiusure
    delle 16:00 fuori sessione) e spot dell'indice implicito dal future live.
    Azioni/ETF: nessun basis, spot = ultimo prezzo (anche pre/after market).
    """
    intr = load_intraday(inst["vp_ticker"])
    last = float(intr["Close"].iloc[-1]) if not intr.empty else float(daily["Close"].iloc[-1])

    if not inst["is_future"]:
        return dict(last=last, basis=0.0, spot=last,
                    spot_src="last trade incl. extended hours", basis_src="")

    if state == "cash session open":
        live = massive_index_live(inst["opt_underlying"])
        src = "Massive"
        if np.isnan(live):
            live, src = load_live_yf(inst["under_yf"]), "yfinance"
        return dict(last=last, basis=last - live, spot=live,
                    spot_src=f"{inst['under']} live {src}", basis_src=f"live {inst['name']} and {inst['under']}")

    close16 = np.nan
    if not intr.empty:
        sub = intr[(intr.index.date == last_session) & (intr.index.time < SESSION_CLOSE)]
        if not sub.empty:
            close16 = float(sub["Close"].iloc[-1])
    if np.isnan(close16):
        close16 = float(daily.loc[pd.Timestamp(last_session), "Close"])
    under_close = float(load_daily(inst["under_yf"]).loc[pd.Timestamp(last_session), "Close"])
    basis = close16 - under_close
    return dict(last=last, basis=basis, spot=last - basis,
                spot_src=f"{inst['under']} implied from live {inst['name']}",
                basis_src=f"4:00 pm closes of {last_session:%Y-%m-%d}")


def analyze(inst, now=None, vp_days=None):
    """
    Analizza uno strumento e restituisce un dizionario con:
      state, ref_date, last_session, vp (edges, vol, poc, val, vah), vp_src, sessions,
      px (last, basis, spot, ...), exps_all e charts: una voce per grafico con
      label, tag, exps, fig, by_strike, flip, call_wall, put_wall, magnet, regime,
      unit (div, nome) e caption (testo riassuntivo in inglese).
    Le figure matplotlib restano aperte: chi chiama le chiude con plt.close().
    """
    now = now or datetime.now(ET)
    vp_days = vp_days or VP_DAYS
    state, ref_date, last_session = market_context(now)
    name = inst["name"]

    daily, sessions, vp, vp_src = volume_profile_for(inst, vp_days)
    edges, vol, poc, val, vah = vp
    px = spot_and_basis(inst, daily, state, last_session)

    chain = prepare_chain(fetch_chain(inst, px["spot"], ref_date), inst, now)
    exps_all = sorted(chain["expiration"].unique())
    if not exps_all:
        raise RuntimeError("no expiries with open interest in the search window")

    ctx = dict(now=now, state=state, ref_date=ref_date, vp_src=vp_src, vp_days=vp_days, **px,
               name=name, under=inst["under"], is_future=inst["is_future"],
               vp_ticker=inst["vp_ticker"])
    basis = px["basis"]

    nearest_dte = trading_days_between(ref_date, exps_all[0])
    configs = [
        (f"nearest expiry, {nearest_dte}DTE", N_EXP_SHORT, False, inst["magnet_short"], "nearest"),
        ("7 expiries, weight 1/sqrt(DTE+1)", N_EXP_LONG, True, inst["magnet_long"], "7exp"),
    ]
    charts = []
    for label, n, weighted, mwin, tag in configs:
        ctx["magnet_window"] = mwin
        exps = exps_all[:n]
        sub = chain[chain["expiration"].isin(exps)]
        gex = compute_gex(sub, px["spot"], now, ref_date, weighted, mwin, inst["strike_pct"])
        by_strike, flip, cw, pw, magnet, regime = gex
        fig = make_chart(vp, gex, ctx, label, exps)

        def fmt(x):
            return "n/a" if np.isnan(x) else fmt_level(x, ctx)
        div, unit = gex_unit(by_strike["net"].values)
        caption = (f"{name} - {vp_days}d VP + {inst['under']} GEX ({label})\n"
                   f"Expiries: {', '.join(e.strftime('%b %d') for e in exps)}\n"
                   f"{name} {px['last']:.2f}"
                   + (f" | basis {basis:+.2f}" if inst["is_future"] else "") + "\n"
                   f"POC {poc:.2f} | VA {val:.2f} - {vah:.2f}\n"
                   f"Gamma flip: {fmt(flip)}\n"
                   f"Call wall: {fmt(cw)}\n"
                   f"Put wall: {fmt(pw)}\n"
                   f"Gamma magnet ({'+' if magnet[1] > 0 else '-'}): {fmt(magnet[0])}\n"
                   f"Gamma regime: {regime}\n"
                   f"Total net GEX: {by_strike['net'].sum() / div:+.2f} $ {unit}/1%")
        charts.append(dict(label=label, tag=tag, exps=exps, fig=fig, by_strike=by_strike,
                           flip=flip, call_wall=cw, put_wall=pw, magnet=magnet, regime=regime,
                           unit=(div, unit), caption=caption, magnet_window=mwin))

    return dict(state=state, ref_date=ref_date, last_session=last_session, vp=vp,
                vp_src=vp_src, sessions=sessions, px=px, exps_all=exps_all,
                charts=charts, ctx=ctx)
