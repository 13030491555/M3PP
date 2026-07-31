# M3PP

Code for **Microstructure-Informed Multimodal Stacking Learning for Mechanical Property Prediction of Recycled QT-450 Ductile Cast Iron**.

## Input

The model table must contain 327 complete records, three target columns, and the predictors listed in a feature manifest.

Feature manifest format:

```csv
feature_name,modality,order
C,composition,1
Si,composition,2
...
Tapping temperature,processing,1
...
Average graphite radius,microstructure,1
...
```

Required modality counts are 38 composition features, 24 processing features, and 6 microstructure descriptors.

Default target names:

- `Yield strength(MPa)`
- `Tensile strength(MPa)`
- `Total elongation(%)`

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src/run_m3pp.py \
  --data final_dataset.xlsx \
  --feature-manifest feature_manifest.csv \
  --output-dir outputs
```

Add `--shap` to calculate SHAP attribution files after model evaluation.

## Image descriptors

```bash
python src/m3pp/image_features.py \
  --unetched sample_unetched.tif \
  --etched sample_etched.tif \
  --pixel-size-um 0.5 \
  --output descriptors.json
```

The training workflow uses three repeated 5-fold splits with seeds 42, 2023, and 2024. Preprocessing is fitted inside each training fold. XGBoost and TabNet predictions are cross-fitted before Ridge regression is trained.
