---
title: Credit Risk - Probability of Default
emoji: 📊
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
license: mit
---

# Credit Risk — Probability of Default

An interactive web app that predicts the **probability of default** for a borrower using an **XGBoost** classifier trained on the German Credit Dataset.

## Features

- Inputs: Age, Sex, Job, Housing, Saving Accounts, Checking Account, Credit Amount, Duration, Purpose
- Output: Probability of default (%), creditworthiness score, risk tier (Low / Medium / High)
- Clean dark finance-themed UI

## Files required in this Space

Upload these files alongside `app.py`:

| File | Description |
|------|-------------|
| `XGBclassifier.pkl` | Trained XGBoost model |
| `Sex_encoder.pkl` | LabelEncoder for Sex |
| `Job_encoder.pkl` | LabelEncoder for Job |
| `Housing_encoder.pkl` | LabelEncoder for Housing |
| `Saving accounts_encoder.pkl` | LabelEncoder for Saving accounts |
| `Checking account_encoder.pkl` | LabelEncoder for Checking account |
| `Purpose_encoder.pkl` | LabelEncoder for Purpose |

## How to export your model from the notebook

Add this cell at the end of your Colab notebook and run it:

```python
import joblib

# Save the XGBoost model
joblib.dump(best_xgb, "XGBclassifier.pkl")

# Save all encoders (they were already saved during feature engineering,
# but re-save here to be sure)
for col in ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
    joblib.dump(encoders[col], f"{col}_encoder.pkl")

# Download from Colab
from google.colab import files
files.download("XGBclassifier.pkl")
for col in ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
    files.download(f"{col}_encoder.pkl")
```

## Deployment on Hugging Face Spaces

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Gradio** as the SDK
3. Upload all `.pkl` files + `app.py` + `requirements.txt`
4. The Space will build and launch automatically
