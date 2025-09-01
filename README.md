# M3PP: Multimodal Microstructure-Informed Property Prediction Framework

An open-source framework for predicting mechanical properties of (recycled) ductile cast iron using integrated Composition–Process–Microstructure (C–P–M) data.  
Focus: clean code structure, reproducible pipelines, explainable stacked ensemble modeling (XGBoost + TabNet), and modular extensibility.

---

## Key Features
- Multimodal fusion: composition + process parameters + engineered microstructure descriptors.
- Stacked ensemble: complementary tabular learners combined via a lightweight meta model.
- Deterministic, scriptable preprocessing & feature extraction.
- Interpretability: SHAP-based global and local explanations; grouped by feature domain.
- Reproducibility: environment file, reproducible run script, versioned outputs.
- Extensibility: plug-in architecture for new features, models, or deployment backends.

---

## Why Multimodal?
Mechanical properties of cast iron depend jointly on chemistry, process control, and microstructure morphology. Pure composition-only models often under-capture variability. Adding engineered microstructure descriptors and selected process factors improves stability, calibration, and robustness (especially for challenging targets).  
