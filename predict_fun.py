import json, joblib, numpy as np, pandas as pd
from pathlib import Path

# ─── 1) carga de artefactos y metadatos ───────────────────────────────
SAVE_DIR = Path("models")
with open(SAVE_DIR/"metadata.json") as f:
    meta = json.load(f)

ALPHA     = {k: tuple(v) for k,v in meta["ALPHA"].items()}
DELTAS    = {k: tuple(v) for k,v in meta["deltas"].items()}
EXTREME_K = meta["EXTREME_K"]
FEATURES  = meta["FEATURES"]

def load_pipe(bucket, alpha):
    return joblib.load(SAVE_DIR/f"{bucket}_{alpha:.2f}.joblib")

# ─── 2) tablas auxiliares ─────────────────────────────────────────────
CSV_PATH = "./data/dataset.csv"
ART_CSV  = "./data/articulos_raw.csv"


# histórico breve para lags/rolling (solo para qty_base)
_hist = pd.read_csv(CSV_PATH, usecols=["Year","Week","Product","Deposit","UOM","Quantity"])
_hist["UOM"]      = _hist["UOM"].str.upper().str.strip()

hier_cols = ["CodigoProducto","CodigoTipo","CodigoRubro"]
art = (pd.read_csv(ART_CSV, usecols=hier_cols)
         .rename(columns={"CodigoProducto":"Product"})
         .drop_duplicates("Product"))

# asegurarnos de que Product sea int64
art["Product"] = pd.to_numeric(art["Product"], errors="coerce")
art = art.dropna(subset=["Product"])
art["Product"] = art["Product"].astype("int64")

# cuantiles fijos para bucket
q33, q66, q99 = _hist["Quantity"].quantile([.33,.66,.99])
def _bucket(q):
    if q < 3:      return "micro"
    if q < q33:    return "small"
    if q < q66:    return "medium"
    if q < 2000:   return "large"
    if q < q99:    return "xlarge"
    return "extreme"

# ─── 3) función predict_interval ───────────────────────────────────────
def predict_interval(year:int, week:int, product:int, deposit:int):
    # 3.1) filtramos histórico para UOM/factor
    sub = _hist[( _hist.Product==product ) & ( _hist.Deposit==deposit )]
    if sub.empty:
        raise ValueError(f"No hay histórico para Product={product} Deposit={deposit}")

    uom    = sub["UOM"].mode().iat[0]

    # 3.2) fila base
    row = pd.DataFrame([{
        "Year": year,
        "Week": week,
        "Product": product,
        "Deposit": deposit,
        "UOM": uom
    }])
    # <<< aquí forzamos el tipo de Product a int64 >>>
    row["Product"] = row["Product"].astype("int64")
    # 3.3) merge con art para incluir CodigoTipo, CodigoRubro
    row = row.merge(art, on="Product", how="left")
    # opcional: si faltan jerarquías, podrías imputar un valor por defecto aquí

    # 3.4) calendar features
    row["date"]      = pd.to_datetime(
        row["Year"].astype(str)+"-W"+row["Week"].astype(str)+"-1",
        format="%G-W%V-%u"
    )
    row["week_sin"]  = np.sin(2*np.pi * row["Week"]/52)
    row["week_cos"]  = np.cos(2*np.pi * row["Week"]/52)
    row["month_end"]   = row["date"].dt.is_month_end.astype("int8")
    row["quarter_end"] = row["date"].dt.is_quarter_end.astype("int8")
    row["year_end"]    = row["date"].dt.is_year_end.astype("int8")
    row["peak_OctJan"] = row["date"].dt.month.isin([10,11,12,1]).astype("int8")
    doy = row["date"].dt.dayofyear
    row["doy_sin"] = np.sin(2*np.pi * doy/365)
    row["doy_cos"] = np.cos(2*np.pi * doy/365)

    # 3.5) history features (Lag_1, Lag_52, Rolling4) desde _hist
    h = sub.set_index(["Year","Week"])["Quantity"].sort_index()
    lag1   = h.get((year, week-1), h.get((year-1,52), h.median()))
    lag52  = h.get((year-1, week), lag1)
    vals   = []
    for w in range(week-4, week):
        yr, wk = (year, w) if w>=1 else (year-1, w+52)
        v = h.get((yr, wk), None)
        if v is not None: vals.append(v)
    rolling4 = np.mean(vals) if vals else lag1
    row["Lag_1"]    = lag1
    row["Lag_52"]   = lag52
    row["Rolling4"] = rolling4

    # 3.6) asignar bucket (usamos lag1 como proxy de qty_base)
    b = _bucket(lag1)

    # 3.7) predecir cuantiles y aplicar CQR
    αlo, αhi = ALPHA[b]
    Δlo, Δhi = DELTAS[b]
    X = row[FEATURES]

    lo  = np.expm1(load_pipe(b, αlo ).predict(X)[0] - Δlo)
    med = np.expm1(load_pipe(b, 0.50).predict(X)[0]        )
    hi  = np.expm1(load_pipe(b, αhi ).predict(X)[0] + Δhi)
    if b == "extreme":
        hi = max(hi, med*(1+EXTREME_K))

    return {
        "p10":     float(max(lo,  0.0)),
        "p50":     float(med),
        "p90":     float(hi),
    }
