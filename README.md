---
title: Mobile Money Credit Scorer
emoji: 📱
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# 📱 Mobile Money Credit Scorer

A machine learning app that scores mobile money customers and produces a loan decision.

## Features
- Enter a customer's cash in/out, transaction activity, utility rate, account age, and balance
- Get a **credit score (300–850)** with a visual gauge
- Get a **loan decision**: Low / Medium / High Risk or Reject
- Try built-in example profiles

## How it works
1. Features are engineered from raw inputs (spending ratio, activity score, stability index)
2. A Logistic Regression model predicts the **probability of default**
3. The probability is converted to a credit score: `score = 850 − prob × 550`
4. The score maps to a loan decision:
   - **≥ 750** → LOW RISK — HIGH LOAN ✅
   - **≥ 650** → MEDIUM RISK — MEDIUM LOAN ⚠️
   - **≥ 550** → HIGH RISK — SMALL LOAN 🔶
   - **< 550** → REJECT ❌

## Deploy to Hugging Face Spaces

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Gradio** as the SDK
3. Upload `app.py` and `requirements.txt`
4. The Space will build and launch automatically — no other config needed
