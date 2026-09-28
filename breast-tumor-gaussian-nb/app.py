"""
app.py — Aplikasi web "Breast Tumor Diagnosis Classification Using Gaussian
Naive Bayes Based on Morphological Features".

Jalankan:  streamlit run app.py

Aplikasi ini tidak berisi logika Naive Bayes. Semua perhitungan model
dipanggil dari src/naive_bayes.NaiveBayesClassifier (yang membungkus
src/gnb_manual.py buatan tim).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src import project_info as PI
import datetime as _dt
import html as _html
from src.data_utils import (
    all_feature_names, load_dataframe, plain_feature, pretty_name, stratified_split_indices,
)
from src.metrics import auc, classification_report, roc_curve_points
from src.naive_bayes import NaiveBayesClassifier

ROOT = Path(__file__).resolve().parent
TITLE = "Breast Tumor Diagnosis Classification Using Gaussian Naive Bayes Based on Morphological Features"

# Palet diambil dari pewarnaan H&E pada preparat sitologi:
# hematoksilin (biru-ungu) untuk Benign, eosin (merah muda) untuk Malignant.
HEMA, EOSIN, INK, MUTED, LINE, GLASS = "#2F3A7A", "#C2185B", "#1C1B2E", "#5F6480", "#DDE1EC", "#F4F6FB"
CLS = {0: {"name": "Jinak (Benign)", "short": "Jinak", "color": HEMA, "icon": "🟦"},
       1: {"name": "Ganas (Malignant)", "short": "Ganas", "color": EOSIN, "icon": "🟥"}}

st.set_page_config(page_title="Asisten Diagnosis · Gaussian Naive Bayes",
                   page_icon="🔬", layout="wide", initial_sidebar_state="auto")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp input, .stApp textarea, .stApp button p, .stApp [data-baseweb="tab"] p {{
  font-family: 'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.015em; }}
h1 {{ font-weight: 800; font-size: 2.05rem; line-height: 1.15; }}
h2 {{ font-weight: 700; font-size: 1.45rem; }}
h3 {{ font-weight: 700; font-size: 1.15rem; }}
p, li {{ line-height: 1.55; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0.35rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-size: 0.95rem; font-weight: 600; padding: 0.55rem 0.7rem; }}
.hero {{ border-radius: 14px; padding: 1.4rem 1.6rem; margin: 0.4rem 0 1.2rem 0;
        background: linear-gradient(120deg, {HEMA} 0%, #4B2F74 58%, {EOSIN} 130%); color: white; }}
.hero h1 {{ color: white; margin: 0 0 0.4rem 0; font-size: 1.75rem; }}
.hero p {{ color: #E7E8F7; margin: 0; max-width: 62rem; font-size: 1.02rem; }}
.verdict {{ border-radius: 12px; padding: 1.1rem 1.3rem; border: 2px solid var(--c);
           background: {GLASS}; }}
.verdict .label {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 0.15rem; }}
.verdict .big {{ font-size: 1.9rem; font-weight: 800; color: var(--c); line-height: 1.15; }}
.verdict .prob {{ font-size: 1.05rem; color: {INK}; margin-top: 0.35rem; }}
.pill {{ display: inline-block; border-radius: 999px; padding: 0.12rem 0.65rem; font-size: 0.85rem;
        font-weight: 600; margin-top: 0.55rem; }}
.pill.ok {{ background: #E3F3EA; color: #1C6B3E; }}
.pill.no {{ background: #FCE4EC; color: #8E1244; }}
.pill.na {{ background: #ECEEF5; color: {MUTED}; }}
.stepcard {{ border-left: 4px solid {HEMA}; background: {GLASS}; padding: 0.8rem 1rem;
            border-radius: 0 10px 10px 0; margin-bottom: 0.6rem; }}
.stepcard b {{ color: {HEMA}; }}
.muted {{ color: {MUTED}; font-size: 0.9rem; }}
.warnbox {{ background: #FFF6E5; border: 1px solid #F2D49B; border-radius: 10px;
           padding: 0.75rem 1rem; color: #6B4A00; font-size: 0.93rem; }}
[data-testid="stMetricValue"] {{ font-weight: 700; }}
/* fokus keyboard terlihat jelas */
a:focus-visible, button:focus-visible, [role="tab"]:focus-visible, input:focus-visible,
[data-baseweb="select"]:focus-within {{ outline: 3px solid {EOSIN}; outline-offset: 2px; }}
.sr-only {{ position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden;
           clip: rect(0 0 0 0); white-space: nowrap; border: 0; }}
/* tabel dan grafik tidak memaksa halaman melebar */
[data-testid="stDataFrame"], .stPlotlyChart {{ max-width: 100%; }}
@media (max-width: 640px) {{
  .block-container {{ padding-left: 0.8rem; padding-right: 0.8rem; padding-top: 1rem; }}
  h1 {{ font-size: 1.45rem; }} h2 {{ font-size: 1.2rem; }} h3 {{ font-size: 1.02rem; }}
  .hero {{ padding: 1rem 1.05rem; border-radius: 12px; }}
  .hero h1 {{ font-size: 1.18rem; }}
  .hero p {{ font-size: 0.92rem; }}
  .stTabs [data-baseweb="tab-list"] {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
  .stTabs [data-baseweb="tab"] {{ font-size: 0.88rem; padding: 0.45rem 0.55rem; white-space: nowrap; }}
  .verdict .big {{ font-size: 1.45rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.35rem; }}
  .stDataFrame, [data-testid="stDataFrame"] {{ font-size: 0.82rem; }}
}}
section[data-testid="stSidebar"] {{ background: #FBFCFE; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] h4 {{ font-size: 0.82rem; font-weight: 700; color: {MUTED};
  margin: 1.1rem 0 0.2rem 0; }}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] p {{ font-size: 0.86rem; }}
.brand {{ display: flex; gap: 0.6rem; align-items: center; padding: 0.15rem 0 0.9rem 0;
         border-bottom: 1px solid {LINE}; margin-bottom: 0.3rem; }}
.brand-mark {{ font-size: 1.5rem; line-height: 1; }}
.brand-name {{ font-weight: 700; color: {INK}; font-size: 1rem; line-height: 1.2; }}
.brand-sub {{ color: {MUTED}; font-size: 0.78rem; margin-top: 0.1rem; }}
.thrbox {{ background: {GLASS}; border-radius: 10px; padding: 0.7rem 0.8rem; margin-top: 0.4rem; }}
.thrrow {{ display: flex; justify-content: space-between; font-size: 0.84rem; color: {INK}; }}
.thrbar {{ height: 6px; border-radius: 3px; background: #D7DCEA; margin: 0.2rem 0 0.6rem 0; overflow: hidden; }}
.thrbar i {{ display: block; height: 100%; }}
.thrnote {{ font-size: 0.78rem; color: {MUTED}; margin-top: 0.1rem; }}
.lastbox {{ border-left: 4px solid var(--c); background: {GLASS}; border-radius: 0 8px 8px 0;
           padding: 0.5rem 0.7rem; }}
.lastsrc {{ font-size: 0.78rem; color: {MUTED}; }}
.lastres {{ font-size: 0.95rem; font-weight: 700; color: var(--c); }}
.dsbox {{ background: {GLASS}; border-radius: 10px; padding: 0.6rem 0.8rem; font-size: 0.86rem;
         color: {INK}; margin-bottom: 0.2rem; }}
.dsbox .muted {{ font-size: 0.78rem; }}
.sidefoot {{ color: {MUTED}; font-size: 0.75rem; line-height: 1.45; margin-top: 0.9rem;
            padding-top: 0.7rem; border-top: 1px solid {LINE}; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sumber data: dataset bawaan (WDBC) atau dataset penyakit lain yang diunggah
# --------------------------------------------------------------------------- #
BUILTIN_NAME = "Kanker payudara (WDBC)"


def builtin_ds():
    """Dataset bawaan proyek: Breast Cancer Wisconsin (Diagnostic)."""
    return {"name": BUILTIN_NAME, "raw": None, "builtin": True, "label_col": "diagnosis",
            "pos": "M", "neg_name": "Jinak (Benign)", "pos_name": "Ganas (Malignant)",
            "neg_short": "Jinak", "pos_short": "Ganas",
            "feats": tuple(config.FEATURES) if config.FEATURES else tuple(all_feature_names())}


@st.cache_data
def builtin_df():
    return load_dataframe()


@st.cache_data(show_spinner=False)
def parse_csv(raw: bytes):
    txt = raw.decode("utf-8", errors="ignore")
    head = txt.splitlines()[0] if txt.splitlines() else ""
    sep = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(io.StringIO(txt), sep=sep)


@st.cache_resource(show_spinner=False)
def build(dataset_name: str, label_col: str, pos_value: str, feats: tuple, raw: bytes | None):
    """Latih model tim pada dataset aktif dan siapkan split train/test."""
    if raw is None:
        d = builtin_df().copy()
    else:
        d = parse_csv(raw).copy()
        d["label"] = (d[label_col].astype(str).str.strip() == pos_value).astype(int)
    if "id" not in d.columns:
        d.insert(0, "id", np.arange(1, len(d) + 1))
    feats = list(feats)
    X = d[feats].apply(pd.to_numeric, errors="coerce")
    keep = ~X.isna().any(axis=1)
    d, X = d[keep].reset_index(drop=True), X[keep].to_numpy(float)
    y = d["label"].to_numpy(int)
    tr, te = stratified_split_indices(y, config.TEST_SIZE, config.RANDOM_SEED)
    model = NaiveBayesClassifier(priors=config.PRIORS if raw is None else None).fit(X[tr], y[tr])
    return {"model": model, "feats": feats, "X": X, "y": y, "tr": tr, "te": te,
            "df": d, "dropped": int((~keep).sum())}


@st.cache_data(show_spinner=False)
def cv_results(dataset_name: str, feats: tuple, threshold: float, raw_key: str):
    from src.data_utils import stratified_kfold_indices
    X, y = X_ALL, Y_ALL
    rows = []
    for k, (tr, te) in enumerate(stratified_kfold_indices(y, config.CV_FOLDS, config.RANDOM_SEED), 1):
        m = NaiveBayesClassifier().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        r = classification_report(y[te], (p >= threshold).astype(int))
        f, t, _ = roc_curve_points(y[te], p)
        rows.append({"Fold": k, "Akurasi": r["accuracy"], "Sensitivitas": r["sensitivity"],
                     "Spesifisitas": r["specificity"], "Presisi": r["precision"], "AUC": auc(f, t)})
    return pd.DataFrame(rows)


# ---- sumber data: dataset bawaan atau berkas yang diunggah pengguna ----
with st.sidebar:
    st.markdown(f"""
    <div class="brand">
      <div class="brand-mark" aria-hidden="true">🔬</div>
      <div>
        <div class="brand-name">Asisten Diagnosis Naive Bayes</div>
        <div class="brand-sub">Gaussian Naive Bayes untuk data penyakit</div>
      </div>
    </div>""", unsafe_allow_html=True)
    with st.expander("Sumber data", expanded=False):
        src_mode = st.radio("Dataset", [BUILTIN_NAME, "Unggah dataset penyakit lain"],
                            label_visibility="collapsed", key="ds_mode")
        DS, DS_MSG = (builtin_ds(), None) if src_mode == BUILTIN_NAME else (None, None)
        if src_mode != BUILTIN_NAME:
            st.caption("File CSV: kolom-kolom angka sebagai ciri, ditambah satu kolom label "
                       "berisi dua nilai (mis. positif/negatif).")
            up_ds = st.file_uploader("Pilih file CSV", type=["csv"], key="ds_file")
            if up_ds is None:
                DS_MSG = "Belum ada berkas. Dataset bawaan tetap dipakai sampai file diunggah."
            else:
                raw = up_ds.getvalue()
                try:
                    dd = parse_csv(raw)
                except Exception as err:
                    dd, DS_MSG = None, f"File tidak bisa dibaca: {err}"
                if dd is not None and len(dd.columns) < 2:
                    dd, DS_MSG = None, "File harus punya minimal dua kolom."
                if dd is not None:
                    guess = [c for c in dd.columns
                             if dd[c].nunique(dropna=True) == 2 and not pd.api.types.is_float_dtype(dd[c])]
                    lab = st.selectbox("Kolom label (hasil yang diprediksi)", list(dd.columns),
                                       index=list(dd.columns).index(guess[0]) if guess else len(dd.columns) - 1,
                                       key="ds_label")
                    vals = sorted(dd[lab].astype(str).str.strip().dropna().unique())
                    if len(vals) != 2:
                        DS_MSG = (f"Kolom label **{lab}** punya {len(vals)} nilai berbeda, padahal harus "
                                  "tepat 2. Pilih kolom lain.")
                    else:
                        counts_ = dd[lab].astype(str).str.strip().value_counts()
                        rare = min(vals, key=lambda v: counts_.get(v, 0))
                        pos = st.selectbox("Nilai yang dianggap kasus positif (sakit)", vals,
                                           index=vals.index(rare), key="ds_pos",
                                           help="Biasanya kelas yang lebih jarang, mis. pasien sakit.")
                        neg = [v for v in vals if v != pos][0]
                        numeric = [c for c in dd.columns
                                   if c != lab and pd.to_numeric(dd[c], errors="coerce").notna().mean() > 0.9]
                        if len(numeric) < 2:
                            DS_MSG = "Dibutuhkan minimal dua kolom angka sebagai ciri."
                        else:
                            if any(v not in numeric for v in st.session_state.get("ds_feats", [])):
                                st.session_state.pop("ds_feats", None)
                            fsel = st.multiselect("Ciri yang dipakai", numeric, default=numeric,
                                                  key="ds_feats")
                            dsname = st.text_input("Nama dataset (untuk tampilan)",
                                                   value=up_ds.name.rsplit(".", 1)[0], key="ds_name")
                            if len(fsel) >= 2:
                                DS = {"name": dsname or up_ds.name, "raw": raw, "builtin": False,
                                      "label_col": lab, "pos": pos, "pos_name": str(pos),
                                      "neg_name": str(neg), "pos_short": str(pos), "neg_short": str(neg),
                                      "feats": tuple(fsel)}
                            else:
                                DS_MSG = "Pilih minimal dua ciri."
        if DS is None:
            DS = builtin_ds()
        if DS_MSG:
            st.info(DS_MSG)

BUILTIN = DS["builtin"]
CLS = {0: {"name": DS["neg_name"], "short": DS["neg_short"], "color": HEMA, "icon": "🟦"},
       1: {"name": DS["pos_name"], "short": DS["pos_short"], "color": EOSIN, "icon": "🟥"}}

try:
    B = build(DS["name"], DS["label_col"], DS["pos"], DS["feats"], DS["raw"])
except NotImplementedError as e:
    st.error(f"Model belum siap: {e}. Lengkapi src/gnb_manual.py lalu muat ulang halaman.")
    st.stop()
except Exception as e:  # dataset unggahan bermasalah
    st.error(f"Dataset tidak bisa dipakai: {e}")
    st.stop()

model, FEATS = B["model"], B["feats"]
X_ALL, Y_ALL, TR, TE = B["X"], B["y"], B["tr"], B["te"]
df = B["df"]
RAW_KEY = DS["name"] if BUILTIN else f"{DS['name']}:{len(DS['raw'])}"
if st.session_state.get("_ds_key") != RAW_KEY:          # dataset berganti
    _keep = {"ds_mode", "ds_file", "ds_label", "ds_pos", "ds_feats", "ds_name", "_ds_key"}
    for k in [k for k in st.session_state.keys() if k not in _keep]:
        st.session_state.pop(k, None)
    st.session_state["_ds_key"] = RAW_KEY

if min(int(Y_ALL.sum()), int((1 - Y_ALL).sum())) < 10:
    st.warning("Salah satu kelas punya kurang dari 10 sampel, sehingga hasil evaluasi tidak dapat "
               "dipercaya. Gunakan dataset dengan jumlah sampel memadai untuk kedua kelas.")


P1 = CLS[1]["short"].lower()      # sebutan kelas positif, mis. "ganas"
P0 = CLS[0]["short"].lower()      # sebutan kelas negatif, mis. "jinak"
CASE = "tumor" if BUILTIN else "kasus"
PU1 = CLS[1]["short"]
PU0 = CLS[0]["short"]

if BUILTIN:
    HERO_SUB = ("Aplikasi ini memperkirakan apakah sebuah tumor payudara <b>jinak</b> atau <b>ganas</b> "
                "dari 30 ukuran bentuk inti sel pada citra biopsi jarum halus (FNA), lalu menunjukkan "
                "<b>alasan</b> di balik setiap keputusan.")
    DISCLAIMER = ("Prototipe tugas kuliah, <b>bukan alat diagnosis</b>. Diagnosis tumor hanya dapat "
                  "ditegakkan dokter atau ahli patologi.")
else:
    HERO_SUB = (f"Dataset <b>{_html.escape(DS['name'])}</b>: model memperkirakan peluang sebuah kasus "
                f"tergolong <b>{_html.escape(PU1)}</b> atau <b>{_html.escape(PU0)}</b> dari "
                f"{len(FEATS)} ciri numerik, lalu menunjukkan <b>alasan</b> di balik setiap keputusan.")
    DISCLAIMER = ("Prototipe tugas kuliah, <b>bukan alat diagnosis</b>. Keputusan medis harus "
                  "ditegakkan tenaga kesehatan.")


def nice(col: str) -> str:
    if BUILTIN:
        return pretty_name(col)
    return str(col).replace("_", " ").strip().capitalize()


def plain(col: str) -> str:
    if BUILTIN:
        return plain_feature(col)
    return f"Kolom numerik \"{col}\" dari dataset {DS['name']}."


def pct(v, d=1):
    if d >= 1 and 0 < v < 10 ** -(d + 2):
        return "<" + f"{10 ** -d:.{d}f}".replace(".", ",") + "%"
    if d >= 1 and 1 - 10 ** -(d + 2) < v < 1:
        return ">" + f"{100 - 10 ** -d:.{d}f}".replace(".", ",") + "%"
    return f"{v * 100:.{d}f}".replace(".", ",") + "%"


def num(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def fmt_times(v):
    if v >= 1e6:
        return "lebih dari 1 juta"
    return f"{v:,.1f}".replace(",", "_").replace(".", ",").replace("_", ".")


def chart(fig, key=None):
    """Tampilkan grafik Plotly: lebar mengikuti kolom, tanpa toolbar (lebih lega di ponsel)."""
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displayModeBar": False, "responsive": True})


def logit(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(np.log(p / (1 - p)))


def gauss_pdf(x, mu, var):
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def explain(x):
    """Semua angka untuk menjelaskan keputusan pada satu sampel."""
    fll = model.feature_log_likelihoods(x[None, :])[0]            # (F, 2)
    llr = fll[:, 1] - fll[:, 0]
    prior_lr = float(model.log_prior_[1] - model.log_prior_[0])
    total = prior_lr + float(llr.sum())
    p = model.predict_proba(x[None, :])[0]
    tbl = pd.DataFrame({"feature": FEATS, "Ciri": [nice(f) for f in FEATS], "Nilai": x,
                        "log P(x|negatif)": fll[:, 0], "log P(x|positif)": fll[:, 1], "Bukti": llr})
    return {"fll": fll, "llr": llr, "prior_lr": prior_lr, "total": total, "p": p, "tbl": tbl}


def train_stats():
    Xtr = X_ALL[TR]
    return Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-12


def similar_cases(x, k=5):
    """Sampel latih paling mirip (jarak Euclid pada nilai terstandardisasi).
    Hanya sebagai referensi; bukan bagian dari model Naive Bayes."""
    mu, sd = train_stats()
    Z = (X_ALL[TR] - mu) / sd
    z = (x - mu) / sd
    d = np.sqrt(((Z - z) ** 2).sum(axis=1))
    order = np.argsort(d)[:k]
    return pd.DataFrame({"Sampel latih": [f"#{int(df.iloc[TR[i]]['id'])}" for i in order],
                         "Label asli": [CLS[int(Y_ALL[TR[i]])]["short"] for i in order],
                         "Jarak kemiripan": d[order]})


def report_html(src, E, thr, true_label=None):
    p = float(E["p"][1])
    pred = int(p >= thr)
    t = E["tbl"].reindex(E["tbl"]["Bukti"].abs().sort_values(ascending=False).index).head(10)
    rows = "".join(
        f"<tr><td>{_html.escape(r.Ciri)}</td><td>{r.Nilai:.4g}</td><td>{r.Bukti:+.3f}</td>"
        f"<td>{P1 if r.Bukti > 0 else P0}</td></tr>" for r in t.itertuples())
    truth = "" if true_label is None else (
        f"<p>Diagnosis asli: <b>{CLS[true_label]['name']}</b> "
        f"({'cocok' if true_label == pred else 'tidak cocok'} dengan perkiraan model).</p>")
    team = ", ".join(f"{m['name']} ({m['nim']})" for m in PI.TEAM)
    return f"""<!doctype html><html lang="id"><head><meta charset="utf-8">
<title>Hasil pemeriksaan - {_html.escape(src)}</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;max-width:760px;margin:2rem auto;color:#1C1B2E;padding:0 1rem}}
h1{{font-size:1.4rem}} .v{{border:2px solid {CLS[pred]['color']};border-radius:10px;padding:1rem}}
.big{{font-size:1.6rem;font-weight:800;color:{CLS[pred]['color']}}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #DDE1EC;padding:.35rem .5rem;text-align:left;font-size:.9rem}} th{{background:#F4F6FB}}
.note{{background:#FFF6E5;border:1px solid #F2D49B;border-radius:8px;padding:.7rem;font-size:.9rem}}
small{{color:#5F6480}}</style></head><body>
<h1>{_html.escape(PI.TITLE)}</h1>
<small>Dibuat {_dt.datetime.now().strftime('%d-%m-%Y %H:%M')}</small>
<h2>{_html.escape(src)}</h2>
<div class="v"><div>Perkiraan model:</div><div class="big">{CLS[pred]['name']}</div>
<p>Peluang {P1} <b>{pct(p)}</b>, peluang {P0} <b>{pct(1 - p)}</b>. Batas keputusan {pct(thr, 0)}.</p></div>
{truth}
<h3>10 ciri paling berpengaruh</h3>
<table><tr><th>Ciri</th><th>Nilai</th><th>Skor bukti</th><th>Mendorong ke</th></tr>{rows}</table>
<p><small>Skor bukti = log P(ciri | {P1}) − log P(ciri | {P0}). Prior log-rasio {E['prior_lr']:+.3f};
skor akhir {E['total']:+.3f}.</small></p>
<p class="note">Dokumen ini dihasilkan oleh prototipe tugas kuliah dan <b>bukan diagnosis medis</b>.</p>
<p><small>{_html.escape(PI.COURSE)}, {PI.YEAR}. Tim: {_html.escape(team)}.</small></p>
</body></html>"""


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f'<div class="dsbox"><b>Dataset aktif</b><br>{_html.escape(DS["name"])}<br>'
                f'<span class="muted">{len(df)} sampel · {len(FEATS)} ciri · '
                f'{int(Y_ALL.sum())} {_html.escape(CLS[1]["short"])} / '
                f'{int((1 - Y_ALL).sum())} {_html.escape(CLS[0]["short"])}</span></div>',
                unsafe_allow_html=True)

    st.markdown("#### Batas keputusan")
    THR = st.slider(f"Peluang {CLS[1]['short'].lower()} minimum agar sampel dinyatakan {CLS[1]['short'].lower()}",
                    0.05, 0.95, float(config.THRESHOLD), 0.01, format="%.2f", label_visibility="collapsed")
    st.caption(f"Sampel dinyatakan **{CLS[1]['short'].lower()}** jika peluangnya ≥ {pct(THR, 0)}.")
    _p_te = model.predict_proba(X_ALL[TE])[:, 1]
    _r = classification_report(Y_ALL[TE], (_p_te >= THR).astype(int))
    st.markdown(f"""
    <div class="thrbox" role="group" aria-label="Efek batas keputusan pada data uji">
      <div class="thrrow"><span>{_html.escape(CLS[1]["short"])} terdeteksi</span><b style="color:{EOSIN}">{pct(_r['sensitivity'])}</b></div>
      <div class="thrbar" role="img" aria-label="Sensitivitas {pct(_r['sensitivity'])}"><i style="width:{_r['sensitivity'] * 100:.0f}%;background:{EOSIN}"></i></div>
      <div class="thrrow"><span>{_html.escape(CLS[0]["short"])} dikenali benar</span><b style="color:{HEMA}">{pct(_r['specificity'])}</b></div>
      <div class="thrbar" role="img" aria-label="Spesifisitas {pct(_r['specificity'])}"><i style="width:{_r['specificity'] * 100:.0f}%;background:{HEMA}"></i></div>
      <div class="thrnote">Pada {len(TE)} sampel uji: {_r['FN']} kasus {_html.escape(CLS[1]["short"].lower())} terlewat, {_r['FP']} salah alarm.</div>
    </div>""", unsafe_allow_html=True)
    if abs(THR - float(config.THRESHOLD)) > 1e-9:
        st.caption(f"Nilai bawaan tim: {pct(config.THRESHOLD, 0)}.")

    LAST_SLOT = st.container()

    with st.expander("Panduan singkat"):
        st.markdown(
            "1. **Periksa sampel** — pilih sampel uji, isi ukuran sendiri, atau unggah CSV.\n"
            "2. Baca hasil, ringkasan alasan, dan timbangan bukti.\n"
            "3. Buka **Bagaimana model menghitung angka ini?** untuk perhitungan Bayes.\n"
            "4. **Akurasi** — seberapa bisa diandalkan hasilnya.")

    st.markdown(f'<div class="warnbox" role="note">{DISCLAIMER}'
                '</div>',
                unsafe_allow_html=True)

    _links = []
    if PI.GITHUB_URL:
        _links.append(f"[Kode sumber]({PI.GITHUB_URL})")
    if PI.DEMO_URL:
        _links.append(f"[Demo]({PI.DEMO_URL})")
    if _links:
        st.markdown(" · ".join(_links))
    st.markdown(
        '<div class="sidefoot">Proyek Akhir Kecerdasan Buatan · DTETI UGM · 2026<br>'
        + "<br>".join(m["name"] for m in PI.TEAM) + "</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
HEADLINE = TITLE if BUILTIN else f"Klasifikasi {DS['name']} dengan Gaussian Naive Bayes"
st.markdown(f"""
<header class="hero">
  <h1>{_html.escape(HEADLINE)}</h1>
  <p>{HERO_SUB}</p>
</header>
""", unsafe_allow_html=True)

(tab_home, tab_check, tab_sim, tab_feat, tab_acc, tab_data, tab_about) = st.tabs(
    ["Mulai di sini", "Periksa sampel", "Simulasi", "Jelajahi ciri",
     "Akurasi", "Data & istilah", "Tentang"])


# =========================================================================== #
# TAB 0 — Mulai di sini
# =========================================================================== #
with tab_home:
    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        st.markdown("## Untuk apa aplikasi ini?")
        if BUILTIN:
            st.markdown(
                "Setelah biopsi jarum halus, sel dari benjolan payudara difoto di bawah mikroskop. "
                "Dari foto itu komputer mengukur bentuk inti sel: ukurannya, kehalusan tepinya, "
                "lekukannya, dan sebagainya. Aplikasi ini memakai ukuran-ukuran tersebut untuk "
                f"menghitung **peluang tumor itu {P1}**, dengan metode probabilistik "
                "**Gaussian Naive Bayes**.")
        else:
            st.markdown(
                f"Dataset **{DS['name']}** berisi {len(df)} kasus dengan {len(FEATS)} ciri numerik "
                f"dan satu label hasil ({PU1} atau {PU0}). Aplikasi memakai ciri-ciri tersebut untuk "
                f"menghitung **peluang sebuah kasus tergolong {P1}**, dengan metode probabilistik "
                "**Gaussian Naive Bayes**.")
        st.markdown(
            "Tujuannya adalah menunjukkan bagaimana AI berbasis probabilitas dapat memberi "
            "keputusan yang **bisa ditelusuri**: setiap hasil disertai penjelasan ciri mana "
            f"yang mendorong ke arah {P0} atau {P1}.")
        if not BUILTIN:
            st.info("Dataset unggahan sedang dipakai. Kembalikan ke dataset bawaan lewat panel "
                    "samping (**Sumber data**).")
        st.markdown("## Cara memakai")
        for n, (t, d) in enumerate([
            ("Pilih sampel", "Di tab <i>Periksa sampel</i>, pilih sampel dari data uji, isi ukuran "
                             "sendiri, atau unggah file CSV berisi banyak sampel."),
            ("Baca hasil", f"Aplikasi menampilkan peluang {P1}, keputusan akhir, dan apakah "
                           "tebakannya cocok dengan diagnosis sebenarnya."),
            ("Pahami alasannya", "Grafik bukti menunjukkan ciri mana yang paling berpengaruh. "
                                 "Panel <i>Bagaimana model menghitung angka ini?</i> membongkar perhitungannya langkah demi langkah."),
        ], 1):
            st.markdown(f'<div class="stepcard"><b>Langkah {n}. {t}</b><br>{d}</div>',
                        unsafe_allow_html=True)
    with c2:
        st.markdown("## Isi aplikasi")
        st.markdown(
            "- **Periksa sampel**: hasil, alasan, kasus serupa, unggah CSV, unduh laporan. "
            "Di dalamnya ada penjelasan perhitungan Bayes langkah demi langkah.\n"
            "- **Simulasi**: ubah nilai ciri, lihat peluang berubah\n"
            f"- **Jelajahi ciri**: pola tiap ciri pada kasus {P0} dan {P1}\n"
            "- **Akurasi**: hasil pengujian model dan keandalan peluang\n"
            "- **Data & istilah**: sumber data dan kamus istilah\n"
            "- **Tentang**: tim, konfigurasi, dan verifikasi implementasi")
        st.markdown("## Sekilas model")
        r = classification_report(Y_ALL[TE], (model.predict_proba(X_ALL[TE])[:, 1] >= THR).astype(int))
        st.metric("Sampel untuk belajar", f"{len(TR)}")
        st.metric("Sampel untuk menguji", f"{len(TE)}", help="Sampel ini tidak pernah dilihat model saat belajar.")
        st.metric("Tebakan benar pada data uji", pct(r["accuracy"]))
        st.metric(f"Kasus {P1} yang berhasil terdeteksi", f"{r['TP']} dari {r['TP'] + r['FN']}")
        st.markdown("## Batasan")
        st.markdown(
            ("- Hanya memakai data dari satu sumber (Wisconsin, AS, 1990-an).\n"
             if BUILTIN else f"- Hanya memakai data dari berkas {DS['name']} yang Anda unggah.\n")
            + f"- Hanya menerima {len(FEATS)} ciri numerik, bukan foto, gejala, atau riwayat pasien.\n"
            "- Hasilnya adalah perkiraan statistik, **bukan diagnosis medis**.")


# =========================================================================== #
# TAB 1 — Periksa sampel
# =========================================================================== #
with tab_check:
    st.markdown("## Periksa sampel")
    mode = st.radio("Pilih cara memasukkan data",
                    ["Sampel dari data uji", "Isi ukuran sendiri", "Unggah file CSV (banyak sampel)"],
                    horizontal=True)

    x_active, true_label, src_text = None, None, ""

    # ---------------- Mode 1: test sample ----------------
    if mode == "Sampel dari data uji":
        te_rows = df.iloc[TE].copy()
        te_rows["opsi"] = [f"Sampel #{int(r.id)} (diagnosis asli: {CLS[int(r.label)]['short']})"
                           for r in te_rows.itertuples()]
        if "pick" not in st.session_state:
            st.session_state.pick = te_rows["opsi"].iloc[0]
        cA, cB = st.columns([3, 1])
        with cB:
            st.write("")
            if st.button("Ambil sampel acak", width="stretch"):
                st.session_state.pick = te_rows["opsi"].sample(1).iloc[0]
        with cA:
            choice = st.selectbox(f"Pilih dari {len(TE)} sampel uji", te_rows["opsi"].tolist(), key="pick")
        row = te_rows.loc[te_rows["opsi"] == choice].iloc[0]
        x_active = row[FEATS].to_numpy(float)
        true_label = int(row["label"])
        src_text = f"Sampel #{int(row['id'])} dari data uji"
        st.caption("Data uji tidak dipakai saat model belajar, jadi hasil di sini adalah ujian yang jujur.")

    # ---------------- Mode 2: manual ----------------
    elif mode == "Isi ukuran sendiri":
        st.markdown("Isi 30 ukuran di bawah. Agar mudah, mulai dari salah satu titik awal lalu ubah nilainya.")
        opts = ["Median semua data", f"Rata-rata kelas {P0}", f"Rata-rata kelas {P1}"]
        preset = st.radio("Titik awal", opts, horizontal=True)
        base = {opts[0]: df[FEATS].median(),
                opts[1]: df.loc[df.label == 0, FEATS].mean(),
                opts[2]: df.loc[df.label == 1, FEATS].mean()}[preset]
        vals = {}
        groups = [("mean", "Nilai rata-rata (mean)"), ("se", "Variasi antar sel (se)"),
                  ("worst", "Nilai paling ekstrem (worst)")]
        for stat, title in groups:
            group = [f for f in FEATS if f.endswith("_" + stat)]
            if not group:
                continue
            with st.expander(f"{title}: {len(group)} ukuran", expanded=(stat == "mean")):
                cols = st.columns(3)
                for i, f in enumerate(group):
                    lo, hi = float(df[f].min()), float(df[f].max())
                    step = float(10 ** (np.floor(np.log10(max(hi - lo, 1e-9))) - 2))
                    vals[f] = cols[i % 3].number_input(
                        nice(f), min_value=0.0, max_value=hi * 2, value=float(base[f]),
                        step=step, format="%.4f", key=f"in_{preset}_{f}",
                        help=f"{plain(f)} Rentang di data: {lo:.4g} sampai {hi:.4g}.")
        x_active = np.array([vals[f] for f in FEATS], float)
        src_text = "Ukuran yang Anda isi"
        out = [nice(f) for f in FEATS if not (df[f].min() <= vals[f] <= df[f].max())]
        if out:
            st.warning("Beberapa nilai berada di luar rentang data latih, sehingga hasil kurang dapat "
                       "dipercaya: " + ", ".join(out))

    # ---------------- Mode 3: batch CSV ----------------
    else:
        st.markdown(
            "Unggah file CSV dengan **30 kolom ukuran** yang namanya sama seperti dataset "
            "(misalnya `radius_mean`, `texture_mean`, ...). Pemisah koma atau titik koma diterima. "
            "Kolom `diagnosis` boleh ada untuk membandingkan hasil.")
        tmpl = df.iloc[TE[:5]][FEATS]
        st.download_button("Unduh contoh file CSV", tmpl.to_csv(index=False).encode(),
                           "contoh_input.csv", "text/csv")
        up = st.file_uploader("Pilih file CSV", type=["csv"])
        if up is not None:
            raw = up.getvalue().decode("utf-8", errors="ignore")
            sep = ";" if raw.splitlines()[0].count(";") > raw.splitlines()[0].count(",") else ","
            data = pd.read_csv(io.StringIO(raw), sep=sep)
            missing = [f for f in FEATS if f not in data.columns]
            if missing:
                st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing[:8])}"
                         f"{' ...' if len(missing) > 8 else ''}. Samakan nama kolom dengan contoh file.")
            else:
                Xb = data[FEATS].apply(pd.to_numeric, errors="coerce")
                bad = Xb.isna().any(axis=1)
                if bad.any():
                    st.warning(f"{int(bad.sum())} baris dilewati karena ada nilai kosong atau bukan angka.")
                Xb = Xb[~bad]
                pb = model.predict_proba(Xb.to_numpy(float))[:, 1]
                res = pd.DataFrame({"Baris": Xb.index + 1, "Peluang positif": pb,
                                    "Hasil": np.where(pb >= THR, CLS[1]["short"], CLS[0]["short"])})
                if "diagnosis" in data.columns:
                    truth = data.loc[~bad, "diagnosis"].astype(str).str.strip().str.upper().str[0]
                    res["Label asli"] = truth.map({"M": CLS[1]["short"], "B": CLS[0]["short"]}).values
                    res["Cocok?"] = np.where(res["Hasil"] == res["Label asli"], "Ya", "Tidak")
                c = st.columns(3)
                c[0].metric("Sampel diproses", len(res))
                c[1].metric(f"Diperkirakan {P1}", int((res["Hasil"] == CLS[1]["short"]).sum()))
                c[2].metric(f"Diperkirakan {P0}", int((res["Hasil"] == CLS[0]["short"]).sum()))
                st.dataframe(res.style.format({"Peluang positif": "{:.2%}"}), hide_index=True, width="stretch")
                st.download_button("Unduh hasil (CSV)", res.to_csv(index=False).encode(),
                                   "hasil_prediksi.csv", "text/csv")

                # ---- Evaluasi batch (hanya jika ada kolom diagnosis) ----
                if "diagnosis" in data.columns:
                    yb_raw = truth.map({"B": 0, "M": 1}).to_numpy(dtype=float)
                    ok = ~np.isnan(yb_raw)
                    if (~ok).any():
                        st.warning(f"{int((~ok).sum())} baris tidak dievaluasi karena nilai kolom "
                                   "diagnosis tidak dikenali (harus Benign/Malignant atau B/M).")
                    yb = yb_raw[ok].astype(int)
                    pb_eval = pb[ok]
                    st.divider()
                    st.markdown("### Evaluasi file yang diunggah")
                    n_over = len(pd.merge(pd.DataFrame(Xb.to_numpy(float)[ok].round(6)),
                                          pd.DataFrame(X_ALL[TR].round(6)), how="inner"))
                    if n_over:
                        st.markdown(
                            f'<div class="warnbox" role="note"><b>{n_over} dari {len(yb)} baris</b> '
                            'juga dipakai saat model belajar, sehingga angka di bawah lebih bagus '
                            'daripada performa sebenarnya pada data baru. Untuk penilaian yang jujur, '
                            'unggah data yang belum pernah dilihat model.</div>', unsafe_allow_html=True)
                    rb = classification_report(yb, (pb_eval >= THR).astype(int))
                    fb, tb, _ = roc_curve_points(yb, pb_eval)
                    Ab = auc(fb, tb)
                    mb = st.columns(4)
                    mb[0].metric("Akurasi", pct(rb["accuracy"]))
                    mb[1].metric("Sensitivitas", pct(rb["sensitivity"]), help=f"Kasus {P1} yang terdeteksi.")
                    mb[2].metric("Spesifisitas", pct(rb["specificity"]), help=f"Kasus {P0} yang dikenali benar.")
                    mb[3].metric("ROC AUC", num(Ab))
                    st.markdown(
                        f"Dari **{rb['TP'] + rb['FN']} kasus {P1}** di file ini, model menemukan "
                        f"**{rb['TP']}** dan melewatkan **{rb['FN']}**. Dari **{rb['TN'] + rb['FP']} kasus {P0}**, "
                        f"model benar pada **{rb['TN']}** dan salah menandai **{rb['FP']}** sebagai {P1}.")
                    e1, e2 = st.columns(2, gap="large")
                    with e1:
                        zb = [[rb["TN"], rb["FP"]], [rb["FN"], rb["TP"]]]
                        tx = [[f"{rb['TN']}<br>benar: {P0}", f"{rb['FP']}<br>salah alarm"],
                              [f"{rb['FN']}<br>terlewat", f"{rb['TP']}<br>benar: {P1}"]]
                        fcm = go.Figure(go.Heatmap(z=zb, x=[f"Diprediksi {P0}", f"Diprediksi {P1}"],
                                                   y=[f"Sebenarnya {P0}", f"Sebenarnya {P1}"], text=tx,
                                                   texttemplate="%{text}", textfont=dict(size=15),
                                                   colorscale=[[0, GLASS], [1, HEMA]], showscale=False))
                        fcm.update_layout(title="Tabel kebenaran (file ini)", height=340,
                                          yaxis_autorange="reversed", margin=dict(t=50, b=20, l=10, r=10))
                        chart(fcm, key="batch_cm")
                    with e2:
                        frc = go.Figure()
                        frc.add_trace(go.Scatter(x=fb, y=tb, mode="lines", line=dict(color=EOSIN, width=3),
                                                 name=f"File ini (AUC {Ab:.3f})"))
                        frc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                                 line=dict(color=MUTED, dash="dash"), name="Tebakan acak"))
                        frc.update_layout(title="Kurva ROC (file ini)", height=340, plot_bgcolor="white",
                                          xaxis_title="Salah alarm (1 − spesifisitas)",
                                          yaxis_title=f"{CLS[1]['short']} terdeteksi (sensitivitas)",
                                          margin=dict(t=50, b=20, l=10, r=10), legend=dict(x=0.4, y=0.08))
                        chart(frc, key="batch_roc")
                    fhb = go.Figure()
                    for cc in (0, 1):
                        fhb.add_trace(go.Histogram(x=pb_eval[yb == cc] * 100, nbinsx=20, opacity=0.6,
                                                   marker_color=CLS[cc]["color"],
                                                   name=f"Sebenarnya {CLS[cc]['short'].lower()}"))
                    fhb.add_vline(x=THR * 100, line_dash="dot", line_color=INK,
                                  annotation_text="batas keputusan")
                    fhb.update_layout(barmode="overlay", height=300, plot_bgcolor="white",
                                      margin=dict(l=10, r=10, t=30, b=30),
                                      xaxis=dict(title=f"Peluang {P1}", ticksuffix="%"),
                                      yaxis_title="Jumlah sampel", legend=dict(orientation="h", y=-0.3))
                    chart(fhb, key="batch_hist")
                    tsb = np.linspace(0.01, 0.99, 99)
                    sb_ = [classification_report(yb, (pb_eval >= v).astype(int))["sensitivity"] for v in tsb]
                    sp_ = [classification_report(yb, (pb_eval >= v).astype(int))["specificity"] for v in tsb]
                    ftb = go.Figure()
                    ftb.add_trace(go.Scatter(x=tsb * 100, y=np.array(sb_) * 100, name="Sensitivitas",
                                             line=dict(color=EOSIN, width=3)))
                    ftb.add_trace(go.Scatter(x=tsb * 100, y=np.array(sp_) * 100, name="Spesifisitas",
                                             line=dict(color=HEMA, width=3, dash="dash")))
                    ftb.add_vline(x=THR * 100, line_dash="dot", line_color=INK,
                                  annotation_text="batas saat ini")
                    ftb.update_layout(height=300, plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=30),
                                      xaxis=dict(title="Batas keputusan", ticksuffix="%"),
                                      yaxis=dict(title="Persentase", ticksuffix="%", range=[0, 103]),
                                      legend=dict(orientation="h", y=-0.3))
                    chart(ftb, key="batch_thr")
                    wrong_b = res[res["Cocok?"] == "Tidak"]
                    st.markdown(f"**Baris yang salah ditebak: {len(wrong_b)}**")
                    if len(wrong_b):
                        show_b = wrong_b.copy()
                        show_b["Peluang positif"] = [pct(v) for v in show_b["Peluang positif"]]
                        st.dataframe(show_b, hide_index=True, width="stretch")
                    st.download_button("Unduh ringkasan evaluasi (CSV)",
                                       pd.DataFrame([{**rb, "auc": Ab, "threshold": THR,
                                                      "n_sampel": int(len(yb))}]).to_csv(index=False).encode(),
                                       "evaluasi_file.csv", "text/csv")
                else:
                    st.info("Tambahkan kolom `diagnosis` (Benign/Malignant atau B/M) pada file untuk "
                            "melihat confusion matrix, kurva ROC, dan metrik lainnya.")
        st.info("Penjelasan per sampel tersedia di mode *Sampel dari data uji* atau *Isi ukuran sendiri*.")

    # ---------------- Result + explanation (single sample) ----------------
    if x_active is not None:
        E = explain(x_active)
        p_mal = float(E["p"][1])
        pred = int(p_mal >= THR)
        st.divider()
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            st.markdown("### Hasil")
            if true_label is None:
                pill = '<span class="pill na">Diagnosis asli tidak diketahui</span>'
            elif true_label == pred:
                pill = f'<span class="pill ok">✓ Cocok dengan diagnosis asli ({CLS[true_label]["short"]})</span>'
            else:
                pill = f'<span class="pill no">✗ Tidak cocok: diagnosis asli {CLS[true_label]["short"]}</span>'
            st.markdown(f"""
            <div class="verdict" style="--c:{CLS[pred]['color']}" role="status" aria-live="polite"
                 aria-label=f"Hasil: {CLS[pred]['name']}. Peluang {P1} {pct(p_mal)}, peluang {P0} {pct(1 - p_mal)}.">
              <div class="label">{src_text}. Perkiraan model:</div>
              <div class="big"><span aria-hidden="true">{CLS[pred]['icon']}</span> {CLS[pred]['name']}</div>
              <div class="prob">Peluang {P1} <b>{pct(p_mal)}</b>, peluang {P0} <b>{pct(1 - p_mal)}</b></div>
              {pill}
            </div>""", unsafe_allow_html=True)
            st.markdown(
                f"<p class='muted' style='margin-top:.7rem'>Keputusan diambil dengan membandingkan "
                f"peluang {P1} ({pct(p_mal)}) dengan batas keputusan ({pct(THR, 0)}) yang dapat diubah "
                f"di panel samping.</p>", unsafe_allow_html=True)
        with right:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[p_mal * 100], y=[""], orientation="h", marker_color=EOSIN,
                                 name=CLS[1]["short"], hovertemplate=CLS[1]["short"] + " %{x:.1f}%<extra></extra>"))
            fig.add_trace(go.Bar(x=[(1 - p_mal) * 100], y=[""], orientation="h", marker_color=HEMA,
                                 name=CLS[0]["short"], hovertemplate=CLS[0]["short"] + " %{x:.1f}%<extra></extra>"))
            fig.add_vline(x=THR * 100, line_width=3, line_dash="dot", line_color=INK,
                          annotation_text=f"batas {THR * 100:.0f}%", annotation_position="top")
            fig.update_layout(barmode="stack", height=150, margin=dict(l=10, r=10, t=35, b=25),
                              xaxis=dict(range=[0, 100], ticksuffix="%"), showlegend=True,
                              legend=dict(orientation="h", y=-0.45), plot_bgcolor="white",
                              title=dict(text="Pembagian peluang", font_size=14))
            chart(fig)
            tb = E["tbl"].sort_values("Bukti")
            to_m = tb[tb["Bukti"] > 0].tail(3)["Ciri"].tolist()[::-1]
            to_b = tb[tb["Bukti"] < 0].head(3)["Ciri"].tolist()
            st.markdown("**Ringkasan alasan**")
            st.markdown(
                (f"- Paling mendorong ke arah **{P1}**: {', '.join(to_m)}.\n" if to_m else "")
                + (f"- Paling mendorong ke arah **{P0}**: {', '.join(to_b)}." if to_b else ""))

        st.markdown("### Mengapa model memutuskan begitu?")
        st.markdown(
            f"Model menimbang bukti dari setiap ciri. Titik awalnya adalah seberapa umum kasus {P1} "
            "di data latih. Setiap ciri lalu menggeser timbangan: **ke kanan** jika nilainya lebih mirip "
            f"kasus {P1}, **ke kiri** jika lebih mirip kasus {P0}. Posisi akhir menentukan hasil.")
        if len(FEATS) > 6:
            k = st.slider("Jumlah ciri yang ditampilkan satu per satu", 5, len(FEATS),
                          min(10, len(FEATS)))
        else:
            k = len(FEATS)
        t = E["tbl"].reindex(E["tbl"]["Bukti"].abs().sort_values(ascending=False).index)
        top, rest = t.head(k), t.iloc[k:]
        labels = ["Titik awal (prior)"] + top["Ciri"].tolist()
        values = [E["prior_lr"]] + top["Bukti"].tolist()
        measures = ["absolute"] + ["relative"] * len(top)
        if len(rest):
            labels.append(f"{len(rest)} ciri lainnya")
            values.append(float(rest["Bukti"].sum()))
            measures.append("relative")
        labels.append("Hasil akhir")
        values.append(0)
        measures.append("total")
        wf = go.Figure(go.Waterfall(
            orientation="h", y=labels, x=values, measure=measures,
            increasing=dict(marker=dict(color=EOSIN)), decreasing=dict(marker=dict(color=HEMA)),
            totals=dict(marker=dict(color=INK)), connector=dict(line=dict(color=LINE, width=1)),
            hovertemplate="%{y}<br>geser %{x:+.2f}<extra></extra>"))
        wf.add_vline(x=logit(THR), line_dash="dot", line_color=MUTED, line_width=2,
                     annotation_text="batas keputusan", annotation_position="top right")
        wf.add_vline(x=0, line_color=LINE)
        wf.update_layout(height=max(360, 34 * len(labels) + 90), margin=dict(l=10, r=20, t=30, b=40),
                         yaxis=dict(autorange="reversed"), plot_bgcolor="white",
                         xaxis_title="← JINAK     skor bukti (log-odds)     GANAS →")
        chart(wf)
        st.caption(f"Skor bukti adalah log(P(ciri | {P1}) / P(ciri | {P0})). Skor 0 artinya ciri itu "
                   f"sama-sama cocok untuk kedua kelas. Batang merah muda mendorong ke {P1}, biru ke {P0}.")

        st.markdown("### Lihat satu ciri lebih dekat")
        order = t["feature"].tolist()
        if st.session_state.get("feat_zoom") not in order:
            st.session_state.pop("feat_zoom", None)
        fsel = st.selectbox("Pilih ciri", order, format_func=nice, key="feat_zoom")
        j = FEATS.index(fsel)
        xv = float(x_active[j])
        lo = min(float(X_ALL[:, j].min()), xv)
        hi = max(float(X_ALL[:, j].max()), xv)
        grid = np.linspace(lo, hi, 400)
        fz = go.Figure()
        for c in (0, 1):
            fz.add_trace(go.Scatter(x=grid, y=gauss_pdf(grid, model.mean_[c, j], model.var_[c, j]),
                                    mode="lines", fill="tozeroy", line=dict(color=CLS[c]["color"], width=2.5),
                                    opacity=0.55, name=f"Pola tumor {CLS[c]['short'].lower()} (hasil belajar)"))
        fz.add_vline(x=xv, line_color=INK, line_width=3,
                     annotation_text=f"sampel ini: {xv:.4g}", annotation_position="top")
        fz.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=30), plot_bgcolor="white",
                         legend=dict(orientation="h", y=-0.2), xaxis_title=nice(fsel),
                         yaxis_title="kepadatan peluang")
        c1, c2 = st.columns([1.6, 1], gap="large")
        with c1:
            chart(fz)
        with c2:
            d_b = gauss_pdf(xv, model.mean_[0, j], model.var_[0, j])
            d_m = gauss_pdf(xv, model.mean_[1, j], model.var_[1, j])
            ratio = np.exp(E["llr"][j])
            side = P1 if ratio > 1 else P0
            times = ratio if ratio > 1 else 1 / ratio
            st.markdown(f"**{nice(fsel)}**")
            st.markdown(plain(fsel))
            st.markdown(
                f"Saat belajar, model merangkum ciri ini sebagai kurva lonceng (Gaussian) untuk tiap kelas. "
                f"Pada nilai sampel ini, kurva **{side}** lebih tinggi sekitar **{fmt_times(times)} kali**. "
                f"Karena itu ciri ini mendorong keputusan ke arah {side}.")
            st.markdown(
                f"<p class='muted'>Rata-rata {P0} {model.mean_[0, j]:.4g}, rata-rata {P1} "
                f"{model.mean_[1, j]:.4g}. Tinggi kurva: {P0} {d_b:.3g}, {P1} {d_m:.3g}.</p>",
                unsafe_allow_html=True)

        with st.expander("Tabel lengkap bukti untuk semua ciri"):
            show = t[["Ciri", "Nilai", "log P(x|negatif)", "log P(x|positif)", "Bukti"]]
            st.dataframe(show.style.format({"Nilai": "{:.4g}", "log P(x|negatif)": "{:.3f}",
                                            "log P(x|positif)": "{:.3f}", "Bukti": "{:+.3f}"}),
                         hide_index=True, width="stretch")

        st.markdown("### Kasus serupa di data latih")
        st.markdown("Sebagai pembanding, berikut 5 sampel latih yang ukurannya paling mirip dengan sampel ini. "
                    "Daftar ini hanya referensi dan tidak dipakai dalam perhitungan Naive Bayes.")
        sim = similar_cases(x_active)
        cs1, cs2 = st.columns([1.4, 1])
        with cs1:
            st.dataframe(sim.style.format({"Jarak kemiripan": "{:.2f}"}), hide_index=True, width="stretch")
        with cs2:
            nm = int((sim["Label asli"] == CLS[1]["short"]).sum())
            st.metric(f"Kasus serupa yang {P1}", f"{nm} dari {len(sim)}")
            st.caption(f"Jarak dihitung dari {len(FEATS)} ciri yang sudah distandardisasi. "
                       "Makin kecil, makin mirip.")

        st.markdown("### Simpan hasil")
        st.download_button("Unduh laporan hasil (HTML, bisa dicetak ke PDF)",
                           report_html(src_text, E, THR, true_label).encode("utf-8"),
                           file_name=f"hasil_{src_text.replace(' ', '_').replace('#', '')}.html",
                           mime="text/html")

        with st.expander("Bagaimana model menghitung angka ini? (langkah demi langkah)"):
            st.markdown("Model ini memakai **aturan Bayes**: memperbarui keyakinan awal dengan bukti baru. "
                        "Berikut empat langkahnya, lengkap dengan angka sebenarnya dari model.")
            E2 = E

            n_b, n_m = int((Y_ALL[TR] == 0).sum()), int((Y_ALL[TR] == 1).sum())
            pri = np.exp(model.log_prior_)

            st.markdown("### Langkah 1. Belajar dari data")
            st.markdown(
                f"Model mempelajari **{len(TR)} sampel latih** ({n_b} {P0}, {n_m} {P1}). Ia menyimpan dua hal:")
            c1, c2 = st.columns([1, 1.4], gap="large")
            with c1:
                st.markdown(f"**a. Keyakinan awal (prior).** Tanpa melihat ukuran apa pun, peluang sebuah sampel "
                            f"{P1} adalah P({PU1}) = {n_m}/{len(TR)} = **{num(pri[1])}**, dan P({PU0}) = **{num(pri[0])}**.")
                st.markdown("**b. Pola tiap ciri.** Untuk setiap ciri dan setiap kelas, model menghitung "
                            "rata-rata (μ) dan varians (σ²). Keduanya membentuk kurva lonceng Gaussian yang "
                            "menggambarkan nilai yang lazim untuk kelas itu.")
            with c2:
                par = pd.DataFrame({"Ciri": [nice(f) for f in FEATS],
                                    f"μ {P0}": model.mean_[0], f"σ² {P0}": model.var_[0],
                                    f"μ {P1}": model.mean_[1], f"σ² {P1}": model.var_[1]})
                st.dataframe(par.style.format({c: "{:.4g}" for c in par.columns if c != "Ciri"}),
                             hide_index=True, width="stretch", height=260)
                st.caption(f"Total {2 * len(FEATS) * 2} parameter + 2 prior dipelajari dari data latih.")

            st.markdown("### Langkah 2. Menilai sampel baru, satu ciri demi satu ciri")
            st.markdown(
                f"Untuk setiap ciri, model bertanya: *seberapa lazim nilai ini jika kelasnya {P0}, dan jika {P1}?* "
                "Jawabannya adalah tinggi kurva Gaussian pada nilai tersebut, yaitu likelihood P(xᵢ | kelas):")
            st.latex(r"P(x_i \mid C) = \frac{1}{\sqrt{2\pi\sigma^2_{C,i}}}\,"
                     r"\exp\!\left(-\frac{(x_i-\mu_{C,i})^2}{2\sigma^2_{C,i}}\right)")
            t2 = E2["tbl"].reindex(E2["tbl"]["Bukti"].abs().sort_values(ascending=False).index).head(5)
            st.dataframe(t2[["Ciri", "Nilai", "log P(x|negatif)", "log P(x|positif)", "Bukti"]].style.format(
                {"Nilai": "{:.4g}", "log P(x|negatif)": "{:.3f}", "log P(x|positif)": "{:.3f}", "Bukti": "{:+.3f}"}),
                hide_index=True, width="stretch")
            st.caption("Lima ciri paling berpengaruh untuk sampel ini. Angka disimpan dalam bentuk log agar "
                       "perkalian banyak angka kecil tidak menjadi nol di komputer.")

            st.markdown("### Langkah 3. Menggabungkan semua bukti dengan aturan Bayes")
            st.markdown("Naive Bayes menganggap setiap ciri memberi bukti secara terpisah bila kelasnya diketahui "
                        "(inilah asumsi *naive*). Dengan begitu, bukti semua ciri cukup dikalikan, atau dijumlahkan "
                        "dalam bentuk log:")
            st.latex(r"P(C \mid x_1,\dots,x_n) \;\propto\; P(C)\,\prod_{i=1}^{n} P(x_i \mid C)")
            s_llr = float(E2["llr"].sum())
            st.latex(
                r"\underbrace{\log\frac{P(G\mid x)}{P(J\mid x)}}_{%s} = "
                r"\underbrace{\log\frac{P(G)}{P(J)}}_{%s} + \underbrace{\sum_i \log\frac{P(x_i\mid G)}{P(x_i\mid J)}}_{%s}"
                % (f"{E2['total']:+.3f}", f"{E2['prior_lr']:+.3f}", f"{s_llr:+.3f}"))
            p2 = float(E2["p"][1])
            st.markdown(f"Skor akhir **{E2['total']:+.3f}** diubah kembali menjadi peluang: "
                        f"P({PU1} | x) = 1 / (1 + e^(−skor)) = **{pct(p2, 2)}**.")

            st.markdown("### Langkah 4. Mengambil keputusan")
            pred2 = int(p2 >= THR)
            st.markdown(f"Peluang {P1} {pct(p2)} {'≥' if pred2 else '<'} batas keputusan {pct(THR, 0)}, "
                        f"maka hasilnya **{CLS[pred2]['name']}**.")
            st.markdown(
                f"Batas keputusan tidak harus 50%. Dalam konteks medis, melewatkan kasus {P1} (false negative) "
                f"dan salah menandai kasus {P0} (false positive) punya akibat berbeda. Grafik berikut menunjukkan "
                "pengaruh batas keputusan pada data uji:")
            p_te = model.predict_proba(X_ALL[TE])[:, 1]
            ts = np.linspace(0.01, 0.99, 99)
            sens = [classification_report(Y_ALL[TE], (p_te >= v).astype(int))["sensitivity"] for v in ts]
            spec = [classification_report(Y_ALL[TE], (p_te >= v).astype(int))["specificity"] for v in ts]
            ft = go.Figure()
            ft.add_trace(go.Scatter(x=ts * 100, y=np.array(sens) * 100, name=f"Kasus {P1} terdeteksi (sensitivitas)",
                                    line=dict(color=EOSIN, width=3)))
            ft.add_trace(go.Scatter(x=ts * 100, y=np.array(spec) * 100, name=f"Kasus {P0} terdeteksi (spesifisitas)",
                                    line=dict(color=HEMA, width=3, dash="dash")))
            ft.add_vline(x=THR * 100, line_dash="dot", line_color=INK, annotation_text="batas saat ini")
            ft.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=30), plot_bgcolor="white",
                             xaxis=dict(title="Batas keputusan", ticksuffix="%"),
                             yaxis=dict(title="Persentase", ticksuffix="%", range=[0, 103]),
                             legend=dict(orientation="h", y=-0.3))
            chart(ft)

        st.session_state["E"] = E
        st.session_state["src_text"] = src_text
        st.session_state["x_active"] = x_active
        hist = st.session_state.setdefault("history", [])
        entry = {"Waktu": _dt.datetime.now().strftime("%H:%M:%S"), "Sumber": src_text,
                 "Peluang positif": round(p_mal, 4), "Batas": THR, "Hasil": CLS[pred]["short"],
                 "Label asli": CLS[true_label]["short"] if true_label is not None else "-"}
        if not hist or (hist[-1]["Sumber"], hist[-1]["Peluang positif"], hist[-1]["Batas"]) != \
                (entry["Sumber"], entry["Peluang positif"], entry["Batas"]):
            hist.append(entry)

    hist = st.session_state.get("history", [])
    if hist:
        with st.expander(f"Riwayat pemeriksaan sesi ini ({len(hist)})"):
            hd = pd.DataFrame(hist)
            hd["Peluang positif"] = [pct(v) for v in hd["Peluang positif"]]
            st.dataframe(hd.iloc[::-1], hide_index=True, width="stretch")
            ch1, ch2 = st.columns(2)
            ch1.download_button("Unduh riwayat (CSV)", pd.DataFrame(hist).to_csv(index=False).encode(),
                                "riwayat_pemeriksaan.csv", "text/csv")
            if ch2.button("Hapus riwayat"):
                st.session_state["history"] = []
                st.rerun()


# =========================================================================== #
# TAB — Simulasi
# =========================================================================== #
with tab_sim:
    st.markdown("## Simulasi")
    st.markdown(f"Ubah beberapa ukuran dan lihat bagaimana peluang {P1} berubah. Fitur ini membantu memahami "
                "ciri mana yang membuat model berubah pikiran.")
    base_opt = ["Sampel yang sedang diperiksa", f"Rata-rata kasus {P0}", f"Rata-rata kasus {P1}"]
    base_choice = st.radio("Mulai dari", base_opt, horizontal=True, key="sim_base")
    if base_choice == base_opt[0] and st.session_state.get("x_active") is not None:
        x0 = np.array(st.session_state["x_active"], float)
        base_name = st.session_state.get("src_text", "sampel aktif")
    elif base_choice == base_opt[2]:
        x0 = df.loc[df.label == 1, FEATS].mean().to_numpy(float)
        base_name = f"rata-rata kasus {P1}"
    else:
        if base_choice == base_opt[0]:
            st.caption("Belum ada sampel yang diperiksa, jadi dipakai sampel uji pertama.")
            x0 = X_ALL[TE[0]].copy()
            base_name = f"Sampel #{int(df.iloc[TE[0]]['id'])}"
        else:
            x0 = df.loc[df.label == 0, FEATS].mean().to_numpy(float)
            base_name = f"rata-rata kasus {P0}"
    E0 = explain(x0)
    default_feats = E0["tbl"].reindex(E0["tbl"]["Bukti"].abs().sort_values(ascending=False).index)["feature"].head(5).tolist()
    _mk = f"sim_feats_{base_name}"
    if any(v not in FEATS for v in st.session_state.get(_mk, [])):
        st.session_state.pop(_mk, None)
    chosen = st.multiselect("Ciri yang ingin diubah (awalnya: 5 ciri paling berpengaruh)", FEATS,
                            default=default_feats, format_func=nice, key=f"sim_feats_{base_name}")
    x1 = x0.copy()
    if chosen:
        cols = st.columns(2)
        for i, f in enumerate(chosen):
            j = FEATS.index(f)
            lo, hi = float(df[f].min()), float(df[f].max())
            v0 = float(min(max(x0[j], lo), hi))
            x1[j] = cols[i % 2].slider(nice(f), lo, hi, v0, (hi - lo) / 200, format="%.4g",
                                       key=f"sim_{base_name}_{f}", help=plain(f))
    p0, p1 = float(E0["p"][1]), float(model.predict_proba(x1[None, :])[0, 1])
    d0, d1 = int(p0 >= THR), int(p1 >= THR)
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Sebelum ({base_name})", pct(p0), help=f"Peluang {P1} sebelum diubah.")
    m2.metric("Sesudah diubah", pct(p1), f"{(p1 - p0) * 100:+.1f} poin persen".replace(".", ","))
    m3.metric("Keputusan", CLS[d1]["short"], "berubah" if d0 != d1 else "tetap", delta_color="off")
    if d0 != d1:
        st.success(f"Keputusan berubah dari {CLS[d0]['short'].lower()} menjadi {CLS[d1]['short'].lower()}.")

    st.markdown("### Bagaimana peluang berubah jika satu ciri digeser?")
    if st.session_state.get("sim_sweep") not in FEATS:
        st.session_state.pop("sim_sweep", None)
    fsw = st.selectbox("Pilih ciri untuk digeser dari nilai terkecil sampai terbesar", FEATS,
                       index=FEATS.index(chosen[0]) if chosen else 0, format_func=nice, key="sim_sweep")
    j = FEATS.index(fsw)
    grid = np.linspace(float(df[fsw].min()), float(df[fsw].max()), 200)
    Xg = np.repeat(x1[None, :], len(grid), axis=0)
    Xg[:, j] = grid
    pg = model.predict_proba(Xg)[:, 1]
    fs = go.Figure()
    fs.add_trace(go.Scatter(x=grid, y=pg * 100, mode="lines", line=dict(color=EOSIN, width=3),
                            name=f"Peluang {P1}"))
    fs.add_hline(y=THR * 100, line_dash="dot", line_color=INK, annotation_text="batas keputusan")
    fs.add_vline(x=x1[j], line_color=HEMA, line_width=2, annotation_text="nilai sekarang")
    fs.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=30), plot_bgcolor="white",
                     xaxis_title=nice(fsw), yaxis=dict(title=f"Peluang {P1}", ticksuffix="%", range=[-2, 102]),
                     showlegend=False)
    chart(fs)
    st.caption("Ciri lain ditahan pada nilai sekarang. Garis yang melompat tajam menunjukkan model sangat "
               "sensitif terhadap ciri tersebut di sekitar titik itu.")


# =========================================================================== #
# TAB — Jelajahi ciri
# =========================================================================== #
@st.cache_data(show_spinner=False)
def single_feature_auc(ds_key: str):
    out = []
    for j, f in enumerate(FEATS):
        fp, tp, _ = roc_curve_points(Y_ALL, X_ALL[:, j])
        a = auc(fp, tp)
        out.append({"feature": f, "Ciri": nice(f), "AUC": max(a, 1 - a),
                    "Arah": f"lebih besar → cenderung {P1}" if a >= 0.5 else f"lebih kecil → cenderung {P1}"})
    return pd.DataFrame(out).sort_values("AUC", ascending=False).reset_index(drop=True)


with tab_feat:
    st.markdown("## Jelajahi ciri")
    st.markdown(f"Lihat bagaimana setiap ukuran berbeda antara kasus {P0} dan {P1}, dan kurva Gaussian yang "
                "dipelajari model untuk ciri tersebut.")
    if st.session_state.get("explore_feat") not in FEATS:
        st.session_state.pop("explore_feat", None)
    fx = st.selectbox("Pilih ciri", FEATS, format_func=nice, key="explore_feat")
    j = FEATS.index(fx)
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        xtr, ytr = X_ALL[TR, j], Y_ALL[TR]
        grid = np.linspace(xtr.min(), xtr.max(), 300)
        fh = go.Figure()
        for c in (0, 1):
            fh.add_trace(go.Histogram(x=xtr[ytr == c], histnorm="probability density", nbinsx=35,
                                      marker_color=CLS[c]["color"], opacity=0.35,
                                      name=f"Data latih {CLS[c]['short'].lower()}"))
            fh.add_trace(go.Scatter(x=grid, y=gauss_pdf(grid, model.mean_[c, j], model.var_[c, j]), mode="lines",
                                    line=dict(color=CLS[c]["color"], width=3),
                                    name=f"Kurva Gaussian {CLS[c]['short'].lower()}"))
        fh.update_layout(barmode="overlay", height=380, margin=dict(l=10, r=10, t=30, b=30),
                         plot_bgcolor="white", xaxis_title=nice(fx), yaxis_title="kepadatan",
                         legend=dict(orientation="h", y=-0.25))
        chart(fh)
        st.caption("Batang = data latih yang sebenarnya. Garis = kurva Gaussian yang dipakai model. "
                   "Makin mirip keduanya, makin sesuai asumsi Gaussian untuk ciri ini.")
    with c2:
        st.markdown(f"**{nice(fx)}**")
        st.markdown(plain(fx))
        sa = single_feature_auc(RAW_KEY)
        rowf = sa.loc[sa["feature"] == fx].iloc[0]
        st.metric("Daya pisah ciri ini sendirian (AUC)", num(rowf["AUC"]),
                  help=f"1 = memisahkan {P0} dan {P1} dengan sempurna; 0,5 = tidak membantu sama sekali.")
        st.caption(f"Arah: nilai {rowf['Arah']}. Peringkat {int(sa.index[sa['feature'] == fx][0]) + 1} dari {len(sa)}.")
        st.dataframe(pd.DataFrame({"": ["μ (rata-rata)", "σ (simpangan)"],
                                   f"{PU0}": [model.mean_[0, j], np.sqrt(model.var_[0, j])],
                                   f"{PU1}": [model.mean_[1, j], np.sqrt(model.var_[1, j])]})
                     .style.format({f"{PU0}": "{:.4g}", f"{PU1}": "{:.4g}"}), hide_index=True, width="stretch")
        corr = pd.Series(np.corrcoef(X_ALL.T)[j], index=FEATS).drop(fx)
        top = corr.abs().sort_values(ascending=False).head(3)
        st.markdown("**Paling berkaitan dengan ciri ini**")
        st.markdown("\n".join(f"- {nice(k)} (r = {corr[k]:+.2f})" for k in top.index))

    st.markdown("### Peringkat daya pisah semua ciri")
    sa = single_feature_auc(RAW_KEY)
    fr = go.Figure(go.Bar(x=sa["AUC"], y=sa["Ciri"], orientation="h",
                          marker_color=[EOSIN if f == fx else "#AEB4D0" for f in sa["feature"]],
                          hovertemplate="%{y}: AUC %{x:.3f}<extra></extra>"))
    fr.update_layout(height=720, margin=dict(l=10, r=10, t=20, b=30), plot_bgcolor="white",
                     yaxis=dict(autorange="reversed"), xaxis=dict(title="AUC satu ciri", range=[0.5, 1]))
    chart(fr)
    st.caption(f"Setiap batang menunjukkan seberapa baik satu ciri, jika dipakai sendirian, membedakan {P0} dan "
               f"{P1} pada seluruh data. Model Naive Bayes sendiri memakai semua ciri yang dipilih sekaligus.")


# =========================================================================== #
# TAB 3 — Seberapa akurat?
# =========================================================================== #
with tab_acc:
    st.markdown("## Seberapa akurat model ini?")
    yte = Y_ALL[TE]
    p_te = model.predict_proba(X_ALL[TE])[:, 1]
    yhat = (p_te >= THR).astype(int)
    r = classification_report(yte, yhat)
    fpr, tpr, _ = roc_curve_points(yte, p_te)
    A = auc(fpr, tpr)
    st.markdown(f"Model diuji pada **{len(TE)} sampel** yang tidak pernah dilihatnya saat belajar "
                f"({int(yte.sum())} {P1}, {int((1 - yte).sum())} {P0}), dengan batas keputusan {pct(THR, 0)}.")
    m = st.columns(4)
    m[0].metric("Akurasi", pct(r["accuracy"]), help="Persentase semua tebakan yang benar.")
    m[1].metric("Sensitivitas", pct(r["sensitivity"]), help=f"Dari semua kasus {P1}, berapa yang terdeteksi.")
    m[2].metric("Spesifisitas", pct(r["specificity"]), help=f"Dari semua kasus {P0}, berapa yang dikenali {P0}.")
    m[3].metric("ROC AUC", num(A), help="Kemampuan membedakan kedua kelas di semua batas keputusan (1 = sempurna).")
    st.markdown(
        f"Dalam bahasa sehari-hari: dari **{r['TP'] + r['FN']} kasus {P1}**, model menemukan **{r['TP']}** "
        f"dan melewatkan **{r['FN']}**. Dari **{r['TN'] + r['FP']} kasus {P0}**, model benar pada "
        f"**{r['TN']}** dan salah menandai **{r['FP']}** sebagai {P1}.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        z = [[r["TN"], r["FP"]], [r["FN"], r["TP"]]]
        txt = [[f"{r['TN']}<br>benar: {P0}", f"{r['FP']}<br>salah alarm"],
               [f"{r['FN']}<br>terlewat", f"{r['TP']}<br>benar: {P1}"]]
        cm = go.Figure(go.Heatmap(z=z, x=[f"Diprediksi {P0}", f"Diprediksi {P1}"],
                                  y=[f"Sebenarnya {P0}", f"Sebenarnya {P1}"], text=txt,
                                  texttemplate="%{text}", textfont=dict(size=16),
                                  colorscale=[[0, GLASS], [1, HEMA]], showscale=False))
        cm.update_layout(title="Tabel kebenaran (confusion matrix)", height=360,
                         yaxis_autorange="reversed", margin=dict(t=50, b=20, l=10, r=10))
        chart(cm)
    with c2:
        rc = go.Figure()
        rc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=EOSIN, width=3),
                                name=f"Model (AUC {A:.3f})"))
        rc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dash"),
                                name="Tebakan acak"))
        rc.update_layout(title="Kurva ROC", height=360, xaxis_title="Salah alarm (1 − spesifisitas)",
                         yaxis_title=f"{PU1} terdeteksi (sensitivitas)", plot_bgcolor="white",
                         margin=dict(t=50, b=20, l=10, r=10), legend=dict(x=0.45, y=0.08))
        chart(rc)

    st.markdown("### Hasil pada lipatan data berbeda (5-fold cross-validation)")
    st.markdown("Supaya hasil tidak bergantung pada satu pembagian data saja, seluruh data dibagi menjadi "
                "5 bagian. Model dilatih 5 kali, setiap kali diuji pada bagian yang berbeda.")
    cvd = cv_results(DS["name"], DS["feats"], THR, RAW_KEY)
    summ = cvd.drop(columns="Fold").agg(["mean", "std"]).T
    cc = st.columns(4)
    for i, col in enumerate(["Akurasi", "Sensitivitas", "Spesifisitas", "AUC"]):
        mu, sd = summ.loc[col, "mean"], summ.loc[col, "std"]
        if col == "AUC":
            cc[i].metric("AUC rata-rata", num(mu), help="Rata-rata dari 5 fold.")
            cc[i].caption(f"simpangan ± {num(sd)}")
        else:
            cc[i].metric(f"{col} rata-rata", pct(mu), help="Rata-rata dari 5 fold.")
            cc[i].caption(f"simpangan ± {pct(sd)}")
    with st.expander("Lihat hasil tiap fold"):
        st.dataframe(cvd.style.format({c: "{:.3f}" for c in cvd.columns if c != "Fold"}),
                     hide_index=True, width="stretch")

    st.markdown("### Apakah peluang dari model bisa dipercaya apa adanya?")
    st.markdown(f"Sampel uji dikelompokkan menurut peluang {P1} yang diberikan model, lalu dibandingkan dengan "
                f"persentase yang benar-benar {P1} di tiap kelompok. Titik di garis diagonal berarti peluang "
                "model sesuai kenyataan.")
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p_te, edges) - 1, 0, 9)
    rel = pd.DataFrame({"bin": idx, "p": p_te, "y": yte}).groupby("bin").agg(
        rata_peluang=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size")).reset_index()
    fcal = go.Figure()
    fcal.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=MUTED, dash="dash"),
                              name="Sesuai sempurna"))
    fcal.add_trace(go.Scatter(x=rel["rata_peluang"] * 100, y=rel[f"frac_pos"] * 100, mode="markers+lines",
                              marker=dict(size=8 + rel["n"] / rel["n"].max() * 22, color=EOSIN),
                              line=dict(color=EOSIN), name="Model",
                              text=[f"{n} sampel" for n in rel["n"]],
                              hovertemplate="peluang rata-rata %{x:.1f}%<br>benar-benar " + P1 + " %{y:.1f}%<br>%{text}<extra></extra>"))
    fcal.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=30), plot_bgcolor="white",
                       xaxis=dict(title=f"Peluang {P1} menurut model", ticksuffix="%", range=[-3, 103]),
                       yaxis=dict(title=f"Persentase yang benar-benar {P1}", ticksuffix="%", range=[-3, 103]),
                       legend=dict(orientation="h", y=-0.25))
    c1, c2 = st.columns([1.5, 1])
    with c1:
        chart(fcal)
    with c2:
        tb = rel.assign(Kelompok=[f"{int(edges[b] * 100)}–{int(edges[b + 1] * 100)}%" for b in rel["bin"]])
        st.dataframe(pd.DataFrame({"Peluang model": tb["Kelompok"], "Jumlah sampel": tb["n"],
                                   f"Benar-benar {P1}": [pct(v) for v in tb["frac_pos"]]}),
                     hide_index=True, width="stretch")
        st.caption("Ukuran titik sebanding dengan jumlah sampel di kelompok itu.")

    wrong = np.flatnonzero(yhat != yte)
    st.markdown(f"### Sampel yang salah ditebak ({len(wrong)})")
    if len(wrong):
        wdf = pd.DataFrame({"Sampel": [f"#{int(df.iloc[TE[i]]['id'])}" for i in wrong],
                            "Diagnosis asli": [CLS[int(yte[i])]["short"] for i in wrong],
                            "Tebakan model": [CLS[int(yhat[i])]["short"] for i in wrong],
                            f"Peluang {P1}": p_te[wrong]})
        wdf[f"Peluang {P1}"] = [pct(v) for v in wdf[f"Peluang {P1}"]]
        st.dataframe(wdf, hide_index=True, width="stretch")
        st.caption("Pilih nomor sampel ini di tab Periksa sampel untuk melihat alasan kesalahannya.")

# =========================================================================== #
# TAB 4 — Data & istilah
# =========================================================================== #
with tab_data:
    st.markdown("## Data yang dipakai")
    counts = df["label"].value_counts()
    c = st.columns(4)
    c[0].metric("Jumlah sampel", len(df))
    c[1].metric("Jumlah ciri per sampel", len(FEATS))
    c[2].metric(f"Kasus {P0}", int(counts.get(0, 0)))
    c[3].metric(f"Kasus {P1}", int(counts.get(1, 0)))
    if BUILTIN:
        st.markdown(
            "**Sumber:** Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). *Breast Cancer Wisconsin "
            "(Diagnostic)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B")
        st.markdown(
            "Setiap baris adalah satu citra digital hasil biopsi jarum halus (FNA) dari massa payudara. "
            "Dari citra itu, **10 ciri inti sel** diukur, dan masing-masing dirangkum dengan tiga cara "
            "(**mean**, **se**, **worst**), sehingga total ada 30 ukuran. Label diagnosis: **Malignant** "
            f"({P1}) atau **Benign** ({P0}). Nomor sampel ditambahkan oleh aplikasi karena file tidak memuat ID.")
        figs = ROOT / "figures"
        cA, cB = st.columns(2)
        if (figs / "class_distribution.png").exists():
            cA.image(str(figs / "class_distribution.png"), width="stretch")
        if (figs / "correlation_heatmap.png").exists():
            cB.image(str(figs / "correlation_heatmap.png"), width="stretch")
        if (figs / "feature_distributions.png").exists():
            st.image(str(figs / "feature_distributions.png"), width="stretch")
    else:
        st.markdown(
            f"Dataset diunggah oleh pengguna dengan nama **{DS['name']}**. Kolom label: "
            f"**{DS['label_col']}**, dengan nilai **{DS['pos']}** diperlakukan sebagai kasus positif "
            f"({P1}). Cantumkan sumber asli dataset ini di laporan.")
        if B["dropped"]:
            st.caption(f"{B['dropped']} baris dilewati karena ada nilai kosong atau bukan angka.")
        cA, cB = st.columns(2)
        with cA:
            fb1 = go.Figure(go.Bar(x=[PU0, PU1],
                                   y=[int(counts.get(0, 0)), int(counts.get(1, 0))],
                                   marker_color=[HEMA, EOSIN]))
            fb1.update_layout(title="Distribusi kelas", height=320, plot_bgcolor="white",
                              margin=dict(t=50, b=30, l=10, r=10), yaxis_title="Jumlah sampel")
            chart(fb1, key="ds_class")
        with cB:
            cm_ = np.corrcoef(X_ALL, rowvar=False)
            fh1 = go.Figure(go.Heatmap(z=np.abs(cm_), x=[nice(f) for f in FEATS],
                                       y=[nice(f) for f in FEATS], colorscale="RdPu", zmin=0, zmax=1))
            fh1.update_layout(title="Korelasi absolut antar ciri", height=320,
                              margin=dict(t=50, b=30, l=10, r=10))
            chart(fh1, key="ds_corr")
        st.markdown("### Ringkasan tiap ciri")
        desc = pd.DataFrame({"Ciri": [nice(f) for f in FEATS],
                             "Rata-rata": X_ALL.mean(axis=0), "Simpangan baku": X_ALL.std(axis=0),
                             "Min": X_ALL.min(axis=0), "Maks": X_ALL.max(axis=0),
                             f"Rata-rata {P0}": X_ALL[Y_ALL == 0].mean(axis=0),
                             f"Rata-rata {P1}": X_ALL[Y_ALL == 1].mean(axis=0)})
        st.dataframe(desc.style.format({c: "{:.4g}" for c in desc.columns if c != "Ciri"}),
                     hide_index=True, width="stretch", height=300)

    st.markdown("## Kamus istilah")
    st.dataframe(pd.DataFrame({"Kolom": FEATS, "Nama": [nice(f) for f in FEATS],
                               "Artinya": [plain(f) for f in FEATS]}),
                 hide_index=True, width="stretch", height=300)
    gl = {}
    if BUILTIN:
        gl.update({
            f"Benign / {P0}": "Tumor yang tidak menyebar ke jaringan lain.",
            f"Malignant / {P1}": "Tumor kanker yang dapat tumbuh dan menyebar.",
            "FNA (fine needle aspiration)": "Pengambilan sampel sel dari benjolan dengan jarum tipis."})
    gl.update({
        "Prior": "Keyakinan awal sebelum melihat ukuran apa pun, diambil dari proporsi kelas di data latih.",
        "Likelihood": "Seberapa lazim sebuah nilai ukuran untuk kelas tertentu.",
        "Posterior": "Peluang akhir setelah semua bukti digabung dengan aturan Bayes.",
        "Threshold / batas keputusan": f"Nilai peluang {P1} minimum agar sampel dinyatakan {P1}.",
        "Sensitivitas": f"Persentase kasus {P1} yang berhasil terdeteksi.",
        "Spesifisitas": f"Persentase kasus {P0} yang dikenali sebagai {P0}.",
    })
    st.dataframe(pd.DataFrame({"Istilah": gl.keys(), "Arti": gl.values()}), hide_index=True, width="stretch")
    with st.expander("Lihat data mentah"):
        st.dataframe(df.drop(columns=["label"]), hide_index=True, width="stretch")


# =========================================================================== #
# TAB — Tentang proyek
# =========================================================================== #
def team_sections():
    card = ROOT / "docs" / "model_card.md"
    if not card.exists():
        return []
    import re
    text = re.sub(r"<!--.*?-->", "", card.read_text(encoding="utf-8"), flags=re.S)
    out = []
    for part in text.split("### ")[1:]:
        head, _, body = part.partition("\n")
        if body.strip() and "[DIISI TIM]" not in body:
            out.append((head.strip(), body.strip()))
    return out


with tab_about:
    st.markdown("## Tentang proyek")
    st.markdown(f"**{PI.TITLE}**")
    st.markdown(f"Proyek akhir mata kuliah {PI.COURSE}, {PI.YEAR}"
                + (f", Kelompok {PI.GROUP}." if PI.GROUP else "."))
    st.markdown("### Tim")
    has_role = any(m["role"] for m in PI.TEAM)
    team_df = pd.DataFrame([{"Nama": m["name"], "NIM": m["nim"], **({"Peran": m["role"]} if has_role else {})}
                            for m in PI.TEAM])
    st.dataframe(team_df, hide_index=True, width="stretch")

    st.markdown("### Konfigurasi model yang dipakai aplikasi")
    st.markdown(
        f"- Ciri: **{len(FEATS)}** dari 30 ukuran\n"
        f"- Prior: P({P0}) = **{num(np.exp(model.log_prior_[0]))}**, P({P1}) = **{num(np.exp(model.log_prior_[1]))}** "
        f"({'dari proporsi data latih' if config.PRIORS is None else 'ditetapkan tim'})\n"
        f"- Batas keputusan bawaan: **{pct(config.THRESHOLD, 0)}**\n"
        f"- Data latih/uji: **{len(TR)}/{len(TE)}** sampel (terstratifikasi, seed {config.RANDOM_SEED})")

    for head, body in team_sections():
        st.markdown(f"### {head}")
        st.markdown(body)

    st.markdown("### Verifikasi implementasi")
    try:
        from sklearn.naive_bayes import GaussianNB
        sk = GaussianNB(priors=config.PRIORS).fit(X_ALL[TR], Y_ALL[TR])
        p_sk = sk.predict_proba(X_ALL[TE])[:, 1]
        p_me = model.predict_proba(X_ALL[TE])[:, 1]
        agree = float(((p_sk >= THR) == (p_me >= THR)).mean())
        st.markdown(f"Model ditulis dari nol dengan NumPy. Sebagai pengecekan, hasilnya dibandingkan dengan "
                    f"`GaussianNB` dari scikit-learn pada {len(TE)} sampel uji: prediksi sama **{pct(agree)}**, "
                    f"selisih peluang terbesar **{np.abs(p_sk - p_me).max():.1e}**.")
    except ImportError:
        st.markdown("Pasang scikit-learn untuk menampilkan perbandingan dengan implementasi pustaka.")

    st.markdown("### Sumber data")
    if BUILTIN:
        st.markdown("Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). *Breast Cancer Wisconsin "
                    "(Diagnostic)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B")
        st.markdown("Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction "
                    "for breast tumor diagnosis. *Proc. IS&T/SPIE Electronic Imaging*, 1905, 861–870.")
    else:
        st.markdown(f"Dataset aktif: **{DS['name']}**, diunggah oleh pengguna "
                    f"({len(df)} sampel, {len(FEATS)} ciri, kolom label `{DS['label_col']}`). "
                    "Sitasi sumber dataset wajib dicantumkan sendiri di laporan.")
    st.markdown("### Teknologi")
    st.markdown("Python dan NumPy (model Gaussian Naive Bayes ditulis dari nol), pandas, Plotly, Streamlit.")
    links = []
    if PI.GITHUB_URL:
        links.append(f"[Kode sumber di GitHub]({PI.GITHUB_URL})")
    if PI.DEMO_URL:
        links.append(f"[Demo daring]({PI.DEMO_URL})")
    if links:
        st.markdown(" · ".join(links))
    st.markdown('<div class="warnbox" role="note">Prototipe akademik untuk mata kuliah Kecerdasan Buatan. Tidak untuk '
                'keputusan klinis.</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Panel "sampel terakhir" di sidebar (diisi setelah tab Periksa sampel dihitung)
# --------------------------------------------------------------------------- #
_lastE = st.session_state.get("E")
if _lastE is not None:
    _pm = float(_lastE["p"][1])
    _pr = int(_pm >= THR)
    with LAST_SLOT:
        st.markdown("#### Sampel terakhir diperiksa")
        st.markdown(f"""
        <div class="lastbox" style="--c:{CLS[_pr]['color']}" role="status"
             aria-label=f"Sampel terakhir diperiksa: {CLS[_pr]['name']}, peluang {P1} {pct(_pm)}">
          <div class="lastsrc">{st.session_state.get('src_text', 'Sampel')}</div>
          <div class="lastres"><span aria-hidden="true">{CLS[_pr]['icon']}</span> {CLS[_pr]['short']} · {pct(_pm)}</div>
        </div>""", unsafe_allow_html=True)
