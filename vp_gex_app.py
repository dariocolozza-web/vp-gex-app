"""
VP + GEX Analyzer - Resonance QuantLab
App Streamlit: l'utente sceglie uno strumento (future ES, future NQ, un ETF o
un'azione) e ottiene Volume Profile + livelli GEX calcolati al volo.
Il calcolo e' in vp_gex_core.py (stessa cartella).
La chiave Massive si legge da st.secrets["MASSIVE_API_KEY"] (o dalla variabile
d'ambiente MASSIVE_API_KEY): non va mai scritta nel codice.
"""

import os
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import vp_gex_core as core

st.set_page_config(page_title="VP + GEX Analyzer", layout="wide")


# ============================================================
# CHIAVE API E CACHE
# ============================================================

def _api_key():
    try:
        key = st.secrets.get("MASSIVE_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("MASSIVE_API_KEY", "")


core.set_api_key(_api_key())

# Cache sulle funzioni di download del core (una sola volta per processo):
# lo storico cambia lentamente, prezzi live e catene opzioni in pochi minuti.
if not getattr(core, "_st_cached", False):
    core.load_daily = st.cache_data(ttl=900, show_spinner=False)(core.load_daily)
    core.load_hourly = st.cache_data(ttl=900, show_spinner=False)(core.load_hourly)
    core.load_intraday = st.cache_data(ttl=60, show_spinner=False)(core.load_intraday)
    core.load_live_yf = st.cache_data(ttl=60, show_spinner=False)(core.load_live_yf)
    core.massive_index_live = st.cache_data(ttl=60, show_spinner=False)(core.massive_index_live)
    core.fetch_chain = st.cache_data(ttl=300, show_spinner=False)(core.fetch_chain)
    core._st_cached = True


# ============================================================
# INPUT
# ============================================================

KINDS = {
    "Future ES (GEX SPX)": "ES",
    "Future NQ (GEX NDX)": "NQ",
    "ETF": "ETF",
    "Stock": "STOCK",
}

st.title("📊 Volume Profile + GEX Analyzer")
st.caption("Resonance QuantLab — Volume Profile and dealer gamma exposure levels "
           "for ES/NQ futures, ETFs and stocks")

c1, c2, c3, c5, c4 = st.columns([2.2, 1.5, 2.2, 1.4, 1])
with c1:
    kind_label = st.selectbox("Instrument", list(KINDS.keys()))
kind = KINDS[kind_label]
with c2:
    needs_ticker = kind in ("ETF", "STOCK")
    ticker = st.text_input(
        "Ticker", value="GLD" if kind == "ETF" else ("NVDA" if kind == "STOCK" else ""),
        disabled=not needs_ticker, key=f"ticker_{kind}",
        placeholder="e.g. GLD, SPY" if kind == "ETF" else "e.g. NVDA, AAPL",
        help="Yahoo Finance ticker; it must have US-listed options.",
    )
with c3:
    vp_days = st.slider("Volume profile sessions", 20, 250, core.VP_DAYS, 1,
                        help="Number of sessions used to build the Volume Profile (hourly bars).")
with c5:
    exp_days_long = st.number_input(
        "Expiries within (days)", 1, core.MAX_EXP_DAYS,
        core.make_instrument(kind, "X")["exp_days_long"], 1, key=f"expdays_{kind}",
        help="Second chart: sums all expiries within this number of calendar days "
             f"(max {core.MAX_EXP_DAYS}), weighted 1/√(DTE+1). "
             "More days = more data to download and a slower analysis.")
with c4:
    st.write("")
    st.write("")
    run = st.button("Analyze", type="primary", width="stretch")

defaults = core.make_instrument(kind, ticker or "X")
with st.expander("Advanced settings (windows)"):
    a1, a2, a3 = st.columns(3)
    with a1:
        strike_pct = st.number_input("Strikes considered (± % from spot)", 1.0, 30.0,
                                     defaults["strike_pct"] * 100, 0.5, key=f"sp_{kind}") / 100
    with a2:
        magnet_short = st.number_input("Gamma magnet, nearest expiry (± %)", 0.25, 15.0,
                                       defaults["magnet_short"] * 100, 0.25, key=f"ms_{kind}") / 100
    with a3:
        magnet_long = st.number_input("Gamma magnet, second chart (± %)", 0.25, 20.0,
                                      defaults["magnet_long"] * 100, 0.25, key=f"ml_{kind}") / 100


# ============================================================
# ANALISI
# ============================================================

def fig_to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def strike_table(ch, ctx, top=15):
    """Strike con |net GEX| maggiore, nell'unita' del grafico."""
    div, unit = ch["unit"]
    bs = ch["by_strike"]
    t = bs.loc[bs["net"].abs().sort_values(ascending=False).index[:top]].copy()
    out = pd.DataFrame({f"Strike {ctx['under']}": t.index.values})
    if ctx["is_future"]:
        out[f"{ctx['name']} level"] = t.index.values + ctx["basis"]
    out[f"Call GEX ($ {unit})"] = t["call"].values / div
    out[f"Put GEX ($ {unit})"] = t["put"].values / div
    out[f"Net GEX ($ {unit})"] = t["net"].values / div
    return out.sort_values(out.columns[0], ascending=False).reset_index(drop=True)


if run:
    if needs_ticker and not ticker.strip():
        st.warning("Enter the ETF or stock ticker.")
    elif not core.MASSIVE_API_KEY:
        st.error("Massive API key not configured: add MASSIVE_API_KEY to the app secrets.")
    else:
        inst = core.make_instrument(kind, ticker, strike_pct=strike_pct,
                                    magnet_short=magnet_short, magnet_long=magnet_long)
        try:
            with st.spinner(f"Downloading data and running the analysis for {inst['name']}..."):
                res = core.analyze(inst, vp_days=vp_days, exp_days_long=exp_days_long)
            for ch in res["charts"]:
                ch["png"] = fig_to_png(ch.pop("fig"))
            st.session_state["result"] = res
            st.session_state["inst"] = inst
            st.session_state["exp_days_used"] = exp_days_long
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Analysis failed for {inst['name']}: {e}")


# ============================================================
# RISULTATI (restano visibili anche dopo un download)
# ============================================================

res = st.session_state.get("result")
if res is None:
    if not run:
        st.info("Choose an instrument and press **Analyze** to run the analysis.")
else:
    inst = st.session_state["inst"]
    ctx, px = res["ctx"], res["px"]
    edges, vol, poc, val, vah = res["vp"]
    name, under = inst["name"], inst["under"]

    def lvl(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        if inst["is_future"]:
            return f"{x + px['basis']:.2f}"
        return f"{x:.2f}"

    def lvl_help(x):
        if inst["is_future"] and not np.isnan(x):
            return f"{under} {x:.0f}"
        return None

    st.subheader(f"{name} — {ctx['now'].strftime('%Y-%m-%d %H:%M')} ET")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"Last {name}", f"{px['last']:.2f}")
    m2.metric("POC", f"{poc:.2f}", f"price {px['last'] - poc:+.2f} vs POC", delta_color="off")
    m3.metric("Value Area (70%)", f"{val:.2f} – {vah:.2f}")
    if inst["is_future"]:
        m4.metric(f"Basis {name}-{under}", f"{px['basis']:+.2f}", help=px["basis_src"])
    else:
        m4.metric("Spot used for GEX", f"{px['spot']:.2f}", help=px["spot_src"])
    m5.metric("Market state", res["state"], f"ref. expiry {res['ref_date']:%b %d}", delta_color="off")

    tabs = st.tabs([f"Nearest expiry ({res['charts'][0]['label'].split(', ')[1]})",
                    f"Expiries within {st.session_state.get('exp_days_used', '')} days "
                    f"({len(res['charts'][1]['exps'])}, weight 1/√(DTE+1))"])
    for tab, ch in zip(tabs, res["charts"]):
        with tab:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Gamma flip", lvl(ch["flip"]), help=lvl_help(ch["flip"]))
            k2.metric("Call wall", lvl(ch["call_wall"]), help=lvl_help(ch["call_wall"]))
            k3.metric("Put wall", lvl(ch["put_wall"]), help=lvl_help(ch["put_wall"]))
            mk, msign = ch["magnet"]
            k4.metric(f"Gamma magnet ({'+ pin' if msign > 0 else '− accelerator'})" if not np.isnan(mk)
                      else "Gamma magnet", lvl(mk), help=lvl_help(mk))
            div, unit = ch["unit"]
            k5.metric("Regime", ch["regime"],
                      f"net GEX {ch['by_strike']['net'].sum() / div:+.2f} $ {unit}/1%", delta_color="off")

            st.image(ch["png"], width="stretch")
            st.download_button("Download chart (PNG)", ch["png"],
                               file_name=f"{name.lower()}_vp_gex_{ch['tag']}.png",
                               mime="image/png", key=f"dl_{ch['tag']}")
            st.caption("Expiries: " + ", ".join(e.strftime("%d %b %Y") for e in ch["exps"]))

            st.markdown("**Strikes with the largest GEX**")
            tbl = strike_table(ch, ctx)
            st.dataframe(tbl.style.format({c: "{:.2f}" for c in tbl.columns}),
                         hide_index=True, width="stretch")

    with st.expander("Methodology notes"):
        fut_note = (
            f"- **Futures**: the volume profile uses the unadjusted continuous series `{inst['vp_ticker']}` "
            f"(prices actually traded, as on TradingView). {under} strikes are mapped onto the future with "
            f"the {name}−{under} basis: live prices during the cash session, 4:00 pm closes outside it. "
            f"In the weeks after a quarterly roll, the older part of the profile can be offset by a few "
            f"tens of points versus the current contract.\n"
            if inst["is_future"] else
            "- **ETFs/stocks**: profile and GEX are on the same instrument; spot is the last available "
            "price, including pre- and after-market. Gamma is computed with Black-Scholes (European "
            "approximation, also for American options).\n"
        )
        st.markdown(
            f"- **Volume profile**: last {ctx['vp_days']} sessions on hourly bars (daily bars if hourly "
            f"data are unavailable), {core.VP_ROWS} rows; each bar's volume is split across the rows it "
            "spans. 70% value area around the POC.\n"
            "- **GEX**: open interest and implied volatility from Massive; gamma recomputed at the current "
            "spot; dealer long call / short put convention; values in $ per 1% move.\n"
            "- **Expiries**: until 4:00 pm ET the reference is today's expiry, afterwards the next one "
            "(weekends and NYSE holidays excluded).\n"
            "- **Gamma flip**: level where total GEX changes sign. **Call/Put wall**: strike with the "
            "largest call/put GEX. **Gamma magnet**: strike with the largest |net GEX| near spot; if "
            "positive it tends to pin price, if negative it tends to accelerate moves.\n"
            + fut_note +
            "- Data may be delayed depending on the Massive plan and Yahoo Finance limits. "
            "Analysis tool only, not investment advice."
        )
