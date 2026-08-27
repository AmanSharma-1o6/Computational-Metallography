import os, re
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.linear_model import LinearRegression
import xgboost as xgb

BASE = r"C:\projects\computational_metallography"
RAW_DIR = os.path.join(BASE, "data", "raw_micrographs")
PROPS = os.path.join(BASE, "data", "properties", "properties.csv")
MODEL_DIR = os.path.join(BASE, "models")

st.set_page_config(page_title="Metallography Analyzer", layout="wide")
st.title("🔬 Computational Metallography")
st.write("Upload micrograph → grain size (ASTM E112) → yield strength prediction")

FEATURES = ["d_inv_sqrt", "C_pct", "Mn_pct", "Ni_pct", "Cr_pct"]

@st.cache_resource
def load_models():
    hp = joblib.load(os.path.join(MODEL_DIR, "hall_petch.pkl"))
    xg = joblib.load(os.path.join(MODEL_DIR, "strength_model.pkl"))
    return hp, xg

hp_model, xgb_model = load_models()

def enhance(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(blur)

def segment(img, low, high, crop):
    if crop:
        img = img[0:int(img.shape[0]*0.92), :].copy()
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    edges = cv2.Canny(blur, low, high)
    k = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            closed[labels == i] = 0
    return img, closed

def measure(b, upp, n_lines=15):
    h, w = b.shape
    binary = (b > 127).astype(np.int32)
    intercepts = []
    for i in range(n_lines):
        y = int(h*(i+1)/(n_lines+1))
        grains = int(np.sum(np.abs(np.diff(binary[y, :])))) // 2
        if grains >= 2:
            intercepts.append(w*upp/grains)
    if len(intercepts) < 5:
        return None, None
    return float(np.mean(intercepts)), float(np.std(intercepts))

# ══════ OPTION A: controlled retraining ══════
def retrain_models():
    df_all = pd.read_csv(PROPS)
    n = len(df_all)
    hp_new = LinearRegression().fit(df_all[["d_inv_sqrt"]], df_all["yield_MPa"])
    xg_new = xgb.XGBRegressor(n_estimators=200, max_depth=3,
                              learning_rate=0.1, random_state=42)
    xg_new.fit(df_all[FEATURES], df_all["yield_MPa"])
    joblib.dump(hp_new, os.path.join(MODEL_DIR, "hall_petch.pkl"))
    joblib.dump(xg_new, os.path.join(MODEL_DIR, "strength_model.pkl"))

    pred_hp  = hp_new.predict(df_all[["d_inv_sqrt"]])
    pred_xgb = xg_new.predict(df_all[FEATURES])
    mae_hp   = np.mean(np.abs(pred_hp  - df_all["yield_MPa"]))
    mae_xgb  = np.mean(np.abs(pred_xgb - df_all["yield_MPa"]))
    return n, hp_new.intercept_, hp_new.coef_[0], mae_hp, mae_xgb

st.sidebar.header("🔄 Model management")
df_now = pd.read_csv(PROPS)
st.sidebar.metric("Samples in training data", len(df_now))
with st.sidebar.expander("📋 View training samples"):
    st.dataframe(df_now[["sample_id", "d_um", "yield_MPa"]], hide_index=True)
    st.caption(f"Grain sizes: {df_now['d_um'].min():.1f}–{df_now['d_um'].max():.1f} µm | "
               f"Strengths: {df_now['yield_MPa'].min():.0f}–{df_now['yield_MPa'].max():.0f} MPa")

if st.sidebar.button("🔄 Retrain models on ALL data"):
    with st.spinner("Retraining..."):
        n_s, s0, ky, mae_hp, mae_xgb = retrain_models()
    st.sidebar.success(f"✓ Trained on {n_s} samples!")
    st.sidebar.markdown(
        f"**σ₀** = {s0:.0f} MPa | **k_y** = {ky:.2f}\n\n"
        f"MAE — Hall-Petch: {mae_hp:.0f} MPa | XGBoost: {mae_xgb:.0f} MPa")
    st.toast("Models updated! Clear cache...", icon="🔄")
    load_models.clear()          # force reload of new .pkl files
    st.rerun()
# ═════════════════════════════════════════════

st.sidebar.header("📏 Scale calibration")
bar_px = st.sidebar.number_input("Scale bar length (pixels)", 1, 10000, 100)
bar_um = st.sidebar.number_input("Scale bar real length (µm)", 0.01, 10000.0, 19.0, step=0.01)

st.sidebar.header("🎛️ Segmentation tuning")
low  = st.sidebar.slider("Canny low", 10, 200, 50)
high = st.sidebar.slider("Canny high", 50, 400, 150)
crop = st.sidebar.checkbox("Crop bottom 8% (image has scale bar?)", True)

uploaded = st.file_uploader("Upload micrograph (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("Could not read that image file.")
        st.stop()

    nums = [int(m.group(1)) for f in os.listdir(RAW_DIR)
            if (m := re.match(r"sample_(\d+)", f))]
    next_n = max(nums, default=0) + 1
    new_name = f"sample_{next_n:02d}.jpg"
    cv2.imwrite(os.path.join(RAW_DIR, new_name), img)
    st.success(f"✓ Saved to raw_micrographs as {new_name}")

    enh = enhance(img)
    img_c, b = segment(enh, low, high, crop)
    upp = bar_um / bar_px
    d, sd = measure(b, upp)

    col1, col2 = st.columns(2)
    with col1:
        st.image(enh, caption="Enhanced", use_container_width=True)
    with col2:
        overlay = cv2.cvtColor(img_c, cv2.COLOR_GRAY2BGR)
        overlay[b > 0] = [0, 0, 255]
        cov = (b > 0).mean()*100
        st.image(overlay, caption=f"Boundaries — coverage {cov:.1f}% (healthy: 3–25%)",
                 channels="BGR", use_container_width=True)

    if d is None:
        st.warning("⚠ Too few boundaries — drag the sidebar sliders!")
    else:
        G = np.log2((25400/d)**2/6.45)
        m1, m2 = st.columns(2)
        m1.metric("Grain size (ASTM E112)", f"{d:.1f} ± {sd:.1f} µm")
        m2.metric("ASTM grain number", f"G ≈ {G:.1f}")

        d_inv_sqrt = 1.0/np.sqrt(d*1e-6)
        pred_hp = hp_model.predict(pd.DataFrame([[d_inv_sqrt]], columns=["d_inv_sqrt"]))[0]

        st.subheader("🧪 Chemistry (for XGBoost)")
        c1, c2, c3, c4 = st.columns(4)
        C  = c1.number_input("C %",  0.0, 5.0,  0.20, 0.01)
        Mn = c2.number_input("Mn %", 0.0, 20.0, 0.60, 0.01)
        Ni = c3.number_input("Ni %", 0.0, 20.0, 0.00, 0.01)
        Cr = c4.number_input("Cr %", 0.0, 30.0, 0.00, 0.01)

        X = pd.DataFrame([[d_inv_sqrt, C, Mn, Ni, Cr]], columns=FEATURES)
        pred_xgb = xgb_model.predict(X)[0]

        p1, p2 = st.columns(2)
        p1.metric("Hall-Petch prediction", f"{pred_hp:.0f} MPa")
        p2.metric("XGBoost prediction", f"{pred_xgb:.0f} MPa")
        st.sidebar.subheader("➕ Add sample to training data")
        actual = st.sidebar.number_input("Actual yield strength (MPa)", 0.0, 5000.0, 0.0, 1.0)
        if st.sidebar.button("Append to properties.csv"):
            sid = new_name.replace(".jpg", "")
            if actual <= 0:
                st.sidebar.error("Enter the real strength first.")
            else:
                df_old = pd.read_csv(PROPS)
                if sid in df_old["sample_id"].values:
                    st.sidebar.warning("Already in dataset.")
                else:
                    new_row = pd.DataFrame([{"sample_id": sid, "d_um": round(d, 2),
                        "C_pct": C, "Mn_pct": Mn, "Ni_pct": Ni, "Cr_pct": Cr,
                        "yield_MPa": actual, "d_inv_sqrt": d_inv_sqrt}])
                    pd.concat([df_old, new_row], ignore_index=True).to_csv(PROPS, index=False)
                    st.sidebar.success(f"✓ Added! Dataset now has {len(df_old)+1} samples.")
                    st.sidebar.info("💡 Click '🔄 Retrain models' to learn from it.")
