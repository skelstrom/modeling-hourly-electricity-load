![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)


## 📊 Hourly Electricity Load Forecasting

### Time Series, Machine Learning, and Deep Learning Comparison

This project evaluates multiple modeling approaches for **hourly electricity load forecasting**, with emphasis on both predictive performance and operational suitability.

**Evaluation dimensions include:**

- Forecast accuracy  
- Uncertainty calibration  
- Residual behavior  
- Computational efficiency  
- Interpretability  
- Deployability  

Three modeling paradigms are compared:

- **SARIMAX** — classical statistical time series modeling  
- **Gradient Boosting** — tree-based machine learning  
- **Temporal Fusion Transformer (TFT)** — deep learning for time series  

Rather than focusing solely on accuracy, this work examines how models **behave, scale, and explain themselves** in a real-world operational forecasting context.


## 🔍 Key Findings

- Hourly electricity demand exhibits strong autoregressive structure  
- SARIMAX consistently outperforms more complex models for short-horizon forecasting  
- Models with explicit temporal inductive bias are best suited to this problem  
- More complex models are not always more deployable, despite greater flexibility  
- Classical models remain competitive when the data-generating process is well understood


## Repository Structure

```text
artifacts/          # Saved figures for reports / GitHub
data/
├── raw/            # Original source datasets (immutable)
├── interim/        # Cleaned / partially transformed datasets
└── processed/      # Final feature-engineered modeling datasets
models/             # Trained model artifacts (ignored)
notebooks/          # Analysis notebooks (executed, with outputs)
src/                # Reusable code (config, helpers, pipelines)
requirements.txt    # Reproducible environment
.gitignore
README.md
```


## 🔄 Data Lineage

```text
data/raw/
   ↓
data/interim/        (cleaning, alignment, joins)
   ↓
data/processed/     (feature engineering, lags, rolling statistics)
   ↓
model training & evaluation
```
- raw/ data is immutable

- interim/ captures intermediate transformation steps

- processed/ contains modeling-ready features

This separation ensures transparent data lineage and enables partial pipeline re-runs without recomputing everything.

## ▶️ How to Reproduce the Analysis

```bash
### 1. Clone the repository

git clone https://github.com/skelstrom/modeling-hourly-electricity-load.git
cd modeling-hourly-electricity-load


### 2. Create and activate a virtual environment

   python -m venv .venv
   source .venv/bin/activate        # macOS / Linux
   .venv\Scripts\activate         # Windows

### 3. Install dependencies (via notebook)

   ⚠️ Important: Dependencies are installed via the environment setup notebook.

   Run: notebooks/00_Environment_Setup.ipynb

   This notebook installs all required libraries and system dependencies.

### 4. Run the modeling notebook

   notebooks/02_Modeling_and_Evaluation.ipynb

   All datasets and trained models load automatically from disk.
 ```

## 🧠 Model Evaluation Dimensions
```markdown

### Accuracy
- MAE  
- RMSE  
- MASE  

### Uncertainty
- Coverage (90%)  
- Weighted Interval Score (WIS)  

### Efficiency
- Inference runtime  
- Model Efficiency Ratio (MER)  

### Interpretability
- **SARIMAX:** standardized coefficients  
- **Gradient Boosting:** SHAP values  
- **TFT:** attention mechanisms
```

## 🚀 Deployment Perspective

This project explicitly evaluates **deployability**, demonstrating that:

- Simpler models can outperform deep learning when structure is well understood  
- Interpretability and stability matter as much as raw accuracy  
- SARIMAX remains a strong baseline for operational electricity load forecasting  

## 📌 Notes

- Raw data and trained models are intentionally tracked for reproducibility  
- Secrets and API keys are excluded via `.gitignore`  
- Figures used in the analysis are exported to `artifacts/`  

## 📝 License

MIT License
 
