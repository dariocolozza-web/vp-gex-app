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
    "Azione": "STOCK",
}

st.title("📊 Volume Profile + GEX Analyzer")
st.caption("Resonance QuantLab — Volume Profile e livelli di gamma exposure dei dealer "
           "per future ES/NQ, ETF e azioni")

c1, c2, c3, c5, c4 = st.columns([2.2, 1.5, 2.2, 1.4, 1])
with c1:
    kind_label = st.selectbox("Strumento", list(KINDS.keys()))
kind = KINDS[kind_label]
with c2:
    needs_ticker = kind in ("ETF", "STOCK")
    ticker = st.text_input(
        "Ticker", value="GLD" if kind == "ETF" else ("NVDA" if kind == "STOCK" else ""),
        disabled=not needs_ticker, key=f"ticker_{kind}",
        placeholder="es. GLD, SPY" if kind == "ETF" else "es. NVDA, AAPL",
        help="Ticker Yahoo Finance; deve avere opzioni quotate negli USA.",
    )
with c3:
    vp_days = st.slider("Sedute nel volume profile", 20, 250, core.VP_DAYS, 1,
                        help="Numero di sedute usate per il Volume Profile (barre orarie).")
with c5:
    n_exp_long = st.number_input("Scadenze 2° grafico", 2, 20, core.N_EXP_LONG, 1,
                                 help="Quante scadenze sommare nel secondo grafico (pesate 1/√(DTE+1)), "
                                      "partendo dalla più prossima.")
with c4:
    st.write("")
    st.write("")
    run = st.button("Analizza", type="primary", width="stretch")

defaults = core.make_instrument(kind, ticker or "X")
with st.expander("Impostazioni avanzate (finestre)"):
    a1, a2, a3 = st.columns(3)
    with a1:
        strike_pct = st.number_input("Strike considerati (± % dallo spot)", 1.0, 30.0,
                                     defaults["strike_pct"] * 100, 0.5, key=f"sp_{kind}") / 100
    with a2:
        magnet_short = st.number_input("Gamma magnet, scadenza più prossima (± %)", 0.25, 15.0,
                                       defaults["magnet_short"] * 100, 0.25, key=f"ms_{kind}") / 100
    with a3:
        magnet_long = st.number_input("Gamma magnet, 2° grafico (± %)", 0.25, 20.0,
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
        out[f"Livello {ctx['name']}"] = t.index.values + ctx["basis"]
    out[f"Call GEX ($ {unit})"] = t["call"].values / div
    out[f"Put GEX ($ {unit})"] = t["put"].values / div
    out[f"Net GEX ($ {unit})"] = t["net"].values / div
    return out.sort_values(out.columns[0], ascending=False).reset_index(drop=True)


if run:
    if needs_ticker and not ticker.strip():
        st.warning("Inserisci il ticker dell'ETF o dell'azione.")
    elif not core.MASSIVE_API_KEY:
        st.error("Chiave API Massive non configurata: aggiungi MASSIVE_API_KEY nei secrets dell'app.")
    else:
        inst = core.make_instrument(kind, ticker, strike_pct=strike_pct,
                                    magnet_short=magnet_short, magnet_long=magnet_long)
        try:
            with st.spinner(f"Scarico i dati e calcolo il profilo per {inst['name']}..."):
                res = core.analyze(inst, vp_days=vp_days, n_exp_long=n_exp_long)
            for ch in res["charts"]:
                ch["png"] = fig_to_png(ch.pop("fig"))
            st.session_state["result"] = res
            st.session_state["inst"] = inst
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Analisi non riuscita per {inst['name']}: {e}")


# ============================================================
# RISULTATI (restano visibili anche dopo un download)
# ============================================================

res = st.session_state.get("result")
if res is None:
    if not run:
        st.info("Scegli lo strumento e premi **Analizza** per generare l'analisi.")
else:
    inst = st.session_state["inst"]
    ctx, px = res["ctx"], res["px"]
    edges, vol, poc, val, vah = res["vp"]
    name, under = inst["name"], inst["under"]

    def lvl(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/d"
        if inst["is_future"]:
            return f"{x + px['basis']:.2f}"
        return f"{x:.2f}"

    def lvl_help(x):
        if inst["is_future"] and not np.isnan(x):
            return f"{under} {x:.0f}"
        return None

    st.subheader(f"{name} — {ctx['now'].strftime('%Y-%m-%d %H:%M')} ET")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"Ultimo {name}", f"{px['last']:.2f}")
    m2.metric("POC", f"{poc:.2f}", f"{px['last'] - poc:+.2f} dal prezzo", delta_color="off")
    m3.metric("Value Area (70%)", f"{val:.2f} – {vah:.2f}")
    if inst["is_future"]:
        m4.metric(f"Basis {name}-{under}", f"{px['basis']:+.2f}", help=px["basis_src"])
    else:
        m4.metric("Spot usato per il GEX", f"{px['spot']:.2f}", help=px["spot_src"])
    m5.metric("Stato mercato", res["state"], f"scadenza rif. {res['ref_date']:%d/%m}", delta_color="off")

    tabs = st.tabs([f"Scadenza più prossima ({res['charts'][0]['label'].split(', ')[1]})",
                    f"{len(res['charts'][1]['exps'])} scadenze (peso 1/√(DTE+1))"])
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
            st.download_button("Scarica il grafico (PNG)", ch["png"],
                               file_name=f"{name.lower()}_vp_gex_{ch['tag']}.png",
                               mime="image/png", key=f"dl_{ch['tag']}")
            st.caption("Scadenze: " + ", ".join(e.strftime("%d %b %Y") for e in ch["exps"]))

            st.markdown("**Strike con GEX più elevato**")
            tbl = strike_table(ch, ctx)
            st.dataframe(tbl.style.format({c: "{:.2f}" for c in tbl.columns}),
                         hide_index=True, width="stretch")

    with st.expander("Note metodologiche"):
        fut_note = (
            f"- **Future**: il volume profile usa la serie continua `{inst['vp_ticker']}` non aggiustata "
            f"(prezzi realmente scambiati, come su TradingView). Gli strike di {under} sono traslati sul "
            f"future con il basis {name}−{under}: in sessione su prezzi live, fuori sessione sulle chiusure "
            f"delle 16:00. Nelle settimane dopo un roll trimestrale la parte vecchia del profilo può "
            f"risultare sfasata di alcune decine di punti rispetto al contratto corrente.\n"
            if inst["is_future"] else
            "- **ETF/azioni**: profilo e GEX sono sullo stesso strumento; lo spot è l'ultimo prezzo "
            "disponibile, pre e after market compresi. Il gamma è calcolato con Black-Scholes "
            "(approssimazione europea anche per opzioni americane).\n"
        )
        st.markdown(
            f"- **Volume profile**: ultime {ctx['vp_days']} sedute su barre orarie (se non disponibili, "
            f"daily), {core.VP_ROWS} righe; il volume di ogni barra è diviso tra le righe che attraversa. "
            "Value area al 70% attorno al POC.\n"
            "- **GEX**: open interest e volatilità implicita da Massive; gamma ricalcolato allo spot "
            "corrente; convenzione dealer long call / short put; valori in $ per 1% di movimento.\n"
            "- **Scadenze**: fino alle 16:00 ET si parte dalla scadenza di oggi, dopo dalla successiva "
            "(weekend e festività NYSE esclusi).\n"
            "- **Gamma flip**: livello in cui il GEX totale cambia segno. **Call/Put wall**: strike con "
            "il GEX call/put più alto. **Gamma magnet**: strike con |net GEX| massimo vicino allo spot; "
            "se positivo tende a fare da calamita (pinning), se negativo ad accelerare i movimenti.\n"
            + fut_note +
            "- I dati possono essere ritardati in base al piano Massive e ai limiti di Yahoo Finance. "
            "Strumento di analisi, non una raccomandazione di investimento."
        )
