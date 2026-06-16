import gradio as gr
import numpy as np
import joblib
import os

# ── Load model & encoders ──────────────────────────────────────────────
# Job is already numeric in the German Credit dataset (0,1,2,3)
# so no LabelEncoder was saved for it — we handle it directly.
MODEL_PATH  = "XGBclassifier.pkl"
ENCODER_DIR = "."

CAT_COLS_ENCODED = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]

model    = joblib.load(MODEL_PATH)
encoders = {
    col: joblib.load(os.path.join(ENCODER_DIR, f"{col}_encoder.pkl"))
    for col in CAT_COLS_ENCODED
}

# ── Category options ───────────────────────────────────────────────────
def get_choices(col):
    return list(encoders[col].classes_)

SEX_CHOICES      = get_choices("Sex")
HOUSING_CHOICES  = get_choices("Housing")
SAVING_CHOICES   = get_choices("Saving accounts")
CHECKING_CHOICES = get_choices("Checking account")
PURPOSE_CHOICES  = get_choices("Purpose")

# Job values: 0 = unskilled non-resident, 1 = unskilled resident,
#             2 = skilled, 3 = highly skilled
JOB_CHOICES = ["0 — Unskilled non-resident", "1 — Unskilled resident",
               "2 — Skilled", "3 — Highly skilled"]

# ── Prediction function ────────────────────────────────────────────────
def predict_default(age, sex, job_label, housing, saving, checking,
                    credit_amount, duration, purpose):

    # Job: extract the leading integer
    job_enc = int(job_label.split("—")[0].strip())

    # Encode categoricals
    sex_enc      = int(encoders["Sex"].transform([sex])[0])
    housing_enc  = int(encoders["Housing"].transform([housing])[0])
    saving_enc   = int(encoders["Saving accounts"].transform([saving])[0])
    checking_enc = int(encoders["Checking account"].transform([checking])[0])
    purpose_enc  = int(encoders["Purpose"].transform([purpose])[0])

    # Feature order must match training exactly:
    # ['Age','Sex','Job','Housing','Saving accounts','Checking account',
    #  'Credit amount','Duration','Purpose']
    X = np.array([[age, sex_enc, job_enc, housing_enc,
                   saving_enc, checking_enc, credit_amount, duration, purpose_enc]],
                 dtype=float)

    prob_bad  = float(model.predict_proba(X)[0][1])
    prob_good = 1.0 - prob_bad

    if prob_bad < 0.30:
        tier  = "🟢 Low Risk"
        color = "#2ecc71"
    elif prob_bad < 0.60:
        tier  = "🟡 Medium Risk"
        color = "#f39c12"
    else:
        tier  = "🔴 High Risk"
        color = "#e74c3c"

    result_html = f"""
    <div style="
        background: linear-gradient(135deg, #0a0f1e 0%, #0f1729 100%);
        border: 1px solid #1e2d4a;
        border-radius: 12px;
        padding: 28px 32px;
        font-family: 'Segoe UI', sans-serif;
        color: #f4f4f0;
    ">
        <h2 style="margin-top:0; color:#c9a84c; letter-spacing:1px;">Credit Risk Assessment</h2>
        <hr style="border-color:#1e2d4a; margin-bottom:20px;">

        <div style="display:flex; gap:32px; flex-wrap:wrap; margin-bottom:24px;">
            <div style="flex:1; min-width:160px; text-align:center;">
                <div style="font-size:13px; color:#8899aa; margin-bottom:6px;">PROBABILITY OF DEFAULT</div>
                <div style="font-size:48px; font-weight:700; color:{color};">{prob_bad*100:.1f}%</div>
            </div>
            <div style="flex:1; min-width:160px; text-align:center;">
                <div style="font-size:13px; color:#8899aa; margin-bottom:6px;">CREDITWORTHINESS</div>
                <div style="font-size:48px; font-weight:700; color:#2ecc71;">{prob_good*100:.1f}%</div>
            </div>
        </div>

        <div style="background:#1e2d4a; border-radius:8px; height:14px; margin-bottom:18px; overflow:hidden;">
            <div style="
                width:{prob_bad*100:.1f}%;
                height:100%;
                background: linear-gradient(90deg, {color}, #c0392b);
                border-radius:8px;
            "></div>
        </div>

        <div style="
            background:#162040;
            border-left: 4px solid {color};
            border-radius: 4px;
            padding: 14px 18px;
            font-size: 20px;
            font-weight: 600;
        ">
            {tier}
        </div>

        <div style="margin-top:18px; font-size:12px; color:#556677; text-align:right;">
            Powered by XGBoost · German Credit Dataset
        </div>
    </div>
    """
    return result_html


# ── UI ─────────────────────────────────────────────────────────────────
css = """
body { background-color: #0a0f1e !important; }
.gradio-container { background: #0a0f1e !important; }
"""

with gr.Blocks(css=css, title="Credit Risk Predictor") as demo:

    gr.Markdown("""
    # 📊 Credit Risk — Probability of Default
    ### XGBoost Model · German Credit Dataset
    Fill in the borrower's profile below and click **Predict**.
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 👤 Borrower Profile")
            age     = gr.Slider(18, 80, value=35, step=1, label="Age")
            sex     = gr.Dropdown(SEX_CHOICES,     value=SEX_CHOICES[0],     label="Sex")
            job     = gr.Dropdown(JOB_CHOICES,     value=JOB_CHOICES[1],     label="Job")
            housing = gr.Dropdown(HOUSING_CHOICES, value=HOUSING_CHOICES[0], label="Housing")

        with gr.Column(scale=1):
            gr.Markdown("#### 💳 Financial Details")
            saving        = gr.Dropdown(SAVING_CHOICES,   value=SAVING_CHOICES[0],   label="Saving Accounts")
            checking      = gr.Dropdown(CHECKING_CHOICES, value=CHECKING_CHOICES[0], label="Checking Account")
            credit_amount = gr.Number(value=2000, label="Credit Amount (DM)")
            duration      = gr.Slider(6, 72, value=24, step=1, label="Duration (months)")
            purpose       = gr.Dropdown(PURPOSE_CHOICES,  value=PURPOSE_CHOICES[0],  label="Purpose")

        with gr.Column(scale=1):
            gr.Markdown("#### 📈 Result")
            output = gr.HTML()
            with gr.Row():
                predict_btn = gr.Button("⚡ Predict", variant="primary")
                clear_btn   = gr.Button("↺ Reset",   variant="secondary")

    predict_btn.click(
        fn=predict_default,
        inputs=[age, sex, job, housing, saving, checking, credit_amount, duration, purpose],
        outputs=output
    )
    clear_btn.click(fn=lambda: "", inputs=[], outputs=output)

    gr.Markdown("""
    ---
    > **Disclaimer:** Educational / research purposes only.
    """)

demo.launch()
