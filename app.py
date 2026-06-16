import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Training data (from notebook) ──────────────────────────────────────────
data = {
    "cash_in":      [650, 200, 480, 900, 150, 700, 300, 1000, 250, 550],
    "cash_out":     [320, 250, 300, 500, 140, 650, 180,  400, 270, 200],
    "transactions": [ 45,  12,  35,  60,   8,  50,  20,   70,  15,  40],
    "utility_rate": [0.95,0.40,0.85,0.98,0.60,0.70,0.90,0.99,0.50,0.92],
    "account_age":  [ 24,   4,  18,  36,   6,  12,  10,  48,   5,  20],
    "balance":      [150,  20,  90, 300,  25,  80,  60, 500,  30, 110],
    "default":      [  0,   1,   0,   0,   1,   1,   0,   0,   1,   0],
}

df = pd.DataFrame(data)

# ── Feature engineering ─────────────────────────────────────────────────────
df["spending_ratio"]  = df["cash_out"] / (df["cash_in"] + 1)
df["activity_score"]  = df["transactions"] / 50
df["stability"]       = df["account_age"] * df["balance"]

FEATURE_COLS = [
    "cash_in", "cash_out", "transactions", "utility_rate",
    "account_age", "balance", "spending_ratio", "activity_score", "stability",
]

X = df[FEATURE_COLS]
y = df["default"]

# ── Train model ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(random_state=42, max_iter=500)
model.fit(X_scaled, y)

# ── Score helpers ────────────────────────────────────────────────────────────
def prob_to_score(prob: float) -> int:
    prob = max(0.0, min(1.0, prob))
    return int(850 - prob * 550)

def decision_from_score(score: int) -> tuple[str, str]:
    if score >= 750:
        return "✅ LOW RISK — HIGH LOAN",   "#22c55e"
    elif score >= 650:
        return "⚠️ MEDIUM RISK — MEDIUM LOAN", "#f59e0b"
    elif score >= 550:
        return "🔶 HIGH RISK — SMALL LOAN",  "#f97316"
    else:
        return "❌ REJECT",                  "#ef4444"

# ── Score gauge HTML ─────────────────────────────────────────────────────────
def score_gauge_html(score: int, color: str) -> str:
    pct = (score - 300) / 550 * 100   # 300–850 → 0–100%
    return f"""
    <div style="font-family:sans-serif; padding:16px 0;">
      <div style="font-size:13px; color:#6b7280; margin-bottom:4px;">CREDIT SCORE</div>
      <div style="font-size:56px; font-weight:800; color:{color}; line-height:1;">{score}</div>
      <div style="font-size:12px; color:#9ca3af; margin-bottom:12px;">Range 300 – 850</div>
      <div style="background:#e5e7eb; border-radius:999px; height:12px; overflow:hidden;">
        <div style="width:{pct:.1f}%; background:{color}; height:100%;
                    border-radius:999px; transition:width .4s ease;"></div>
      </div>
      <div style="display:flex; justify-content:space-between;
                  font-size:11px; color:#9ca3af; margin-top:4px;">
        <span>300</span><span>575</span><span>850</span>
      </div>
    </div>
    """

# ── Main prediction function ─────────────────────────────────────────────────
def predict(cash_in, cash_out, transactions, utility_rate, account_age, balance):
    row = pd.DataFrame([{
        "cash_in":      cash_in,
        "cash_out":     cash_out,
        "transactions": transactions,
        "utility_rate": utility_rate,
        "account_age":  account_age,
        "balance":      balance,
    }])

    row["spending_ratio"]  = row["cash_out"] / (row["cash_in"] + 1)
    row["activity_score"]  = row["transactions"] / 50
    row["stability"]       = row["account_age"] * row["balance"]

    row_scaled = scaler.transform(row[FEATURE_COLS])
    prob       = model.predict_proba(row_scaled)[0][1]
    score      = prob_to_score(prob)
    label, color = decision_from_score(score)

    gauge   = score_gauge_html(score, color)
    prob_str = f"{prob * 100:.1f}%"
    risk_html = f'<div style="font-size:22px; font-weight:700; color:{color}; padding:8px 0;">{label}</div>'

    return gauge, prob_str, risk_html

# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Mobile Money Credit Scorer",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
    #title  { text-align:center; margin-bottom:4px; }
    #sub    { text-align:center; color:#6b7280; margin-top:0; font-size:14px; }
    .card   { background:#f9fafb; border-radius:12px; padding:20px; }
    footer  { display:none !important; }
    """
) as demo:

    gr.HTML("<h1 id='title'>📱 Mobile Money Credit Scorer</h1>")
    gr.HTML("<p id='sub'>Enter a customer's financial profile to get their credit score and loan decision.</p>")

    with gr.Row():
        # ── Left: inputs ────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="card"):
            gr.Markdown("### 📋 Customer Profile")

            with gr.Row():
                cash_in  = gr.Number(label="Cash In (USD)",  value=500, minimum=0)
                cash_out = gr.Number(label="Cash Out (USD)", value=300, minimum=0)

            with gr.Row():
                transactions = gr.Slider(label="Monthly Transactions", minimum=1, maximum=150, step=1, value=30)
                utility_rate = gr.Slider(label="Utility Rate (0–1)",    minimum=0, maximum=1,   step=0.01, value=0.80)

            with gr.Row():
                account_age = gr.Slider(label="Account Age (months)", minimum=1, maximum=120, step=1, value=12)
                balance     = gr.Number(label="Current Balance (USD)", value=100, minimum=0)

            btn = gr.Button("🔍 Score This Customer", variant="primary", size="lg")

        # ── Right: outputs ───────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="card"):
            gr.Markdown("### 📊 Scoring Result")

            gauge_out  = gr.HTML(label="Credit Score")
            risk_out   = gr.HTML(label="Decision")
            prob_out   = gr.Textbox(label="Default Probability", interactive=False)

    # ── Examples ────────────────────────────────────────────────────────────
    gr.Markdown("### 💡 Try these examples")
    gr.Examples(
        examples=[
            [800, 300, 55, 0.95, 30, 250],   # Strong profile
            [400, 380, 20, 0.65, 8,  40],    # Medium profile
            [100, 300,  5, 0.35,  2,  10],   # Weak profile
        ],
        inputs=[cash_in, cash_out, transactions, utility_rate, account_age, balance],
        label="Click a row to load it, then press Score",
    )

    btn.click(
        fn=predict,
        inputs=[cash_in, cash_out, transactions, utility_rate, account_age, balance],
        outputs=[gauge_out, prob_out, risk_out],
    )

if __name__ == "__main__":
    demo.launch()
