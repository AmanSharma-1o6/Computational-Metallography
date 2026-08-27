# 🔬 Computational Metallography — Grain Size & Yield Strength Predictor

End-to-end ML system that estimates **grain size (ASTM E112)** from steel
micrographs and predicts **yield strength** using both the Hall-Petch
relationship and an XGBoost model trained on chemistry + microstructure features.

![pipeline](docs/pipeline.png)

## ✨ Features
- 📤 Upload micrograph → automatic segmentation (CLAHE + Canny + morphology)
- 📏 Interactive scale-bar calibration
- 📐 ASTM E112 line-intercept grain size measurement with live-tunable parameters
- 🧪 Dual prediction: Hall-Petch linear fit vs XGBoost (chemistry-aware)
- ➕ Human-in-the-loop data collection: append validated samples via UI
- 🔄 Controlled retraining with before/after MAE quality metrics
