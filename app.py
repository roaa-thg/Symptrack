import base64
import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_logo_base64():
    for name in ["image (1).png", "logo.png"]:
        if os.path.exists(name):
            with open(name, "rb") as f:
                return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    return ""

LOGO_URI = get_logo_base64()


MODEL_PATH = "./clinicalbert_final_v2"
MAX_LENGTH = 80
CONFIDENCE_THRESHOLD = 0.65
DEFAULT_LOW_CONF_LABEL = "ER"

ID2LABEL = {0: "ER", 1: "Doctor", 2: "Self-care"}


TRIAGE_META = {
    "ER": {
        "badge": "Urgent Care / Emergency",
        "badge_bg": "#FFE3E1",
        "badge_text": "#C1352E",
        "border_color": "#FF5C5C",
        "headline": "Immediate Medical Evaluation Advised",
        "action": "Proceed to the nearest emergency department or call 997 immediately.",
        "note": "Symptoms match clinical patterns that may require urgent intervention.",
    },
    "Doctor": {
        "badge": "Clinical Consultation",
        "badge_bg": "#FFEAD9",
        "badge_text": "#B85C12",
        "border_color": "#FF9D52",
        "headline": "Physician Review Recommended",
        "action": "Schedule an appointment with a clinic or general practitioner soon.",
        "note": "Non-emergency assessment suggested for proper diagnostic checkup.",
    },
    "Self-care": {
        "badge": "Home Monitoring",
        "badge_bg": "#E8F5DA",
        "badge_text": "#3D6E16",
        "border_color": "#5EA829",
        "headline": "Rest and Routine Monitoring",
        "action": "Stay rested, drink fluids, and monitor for changes. Seek care if symptoms worsen.",
        "note": "Reported indicators are typical of mild conditions managed with self-care.",
    },
}

EMPTY_STATE = """
<div class="st-empty">
    <p>Describe what you're experiencing to view triage guidance.</p>
</div>
"""

DIST_EMPTY = '<p class="st-dist-empty">The likelihood breakdown will appear here once you check your symptoms.</p>'

TRIAGE_ORDER = ["ER", "Doctor", "Self-care"]


def build_distribution_html(prob_dict):
    if not prob_dict:
        return DIST_EMPTY
    rows = []
    for label in TRIAGE_ORDER:
        pct = prob_dict.get(label, 0.0) * 100
        color = TRIAGE_META[label]["border_color"]
        rows.append(f"""
        <div class="st-dist-row">
            <div class="st-dist-label">
                <span>{label}</span>
                <span class="st-dist-pct">{pct:.1f}%</span>
            </div>
            <div class="st-dist-track">
                <div class="st-dist-fill" style="width:{pct:.1f}%; background:{color};"></div>
            </div>
        </div>
        """)
    return f'<div class="st-dist">{"".join(rows)}</div>'

# 3. Model Pipeline Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    model_loaded = True
except Exception as e:
    model_loaded = False

# 4. Inference Logic
def run_triage(symptoms_text):
    if not symptoms_text.strip():
        return EMPTY_STATE, DIST_EMPTY

    if not model_loaded:
        err = """
        <div class="st-error">
            <strong>Configuration notice:</strong> local model weights not found in <code>./clinicalbert_final_v2</code>.
        </div>
        """
        fallback = {"ER": 0.33, "Doctor": 0.33, "Self-care": 0.34}
        return err, build_distribution_html(fallback)

    inputs = tokenizer(
        symptoms_text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()

    prob_dict = {ID2LABEL[i]: float(prob) for i, prob in enumerate(probabilities)}
    top_idx = int(torch.argmax(logits, dim=-1).item())
    top_prob = prob_dict[ID2LABEL[top_idx]]

    if top_prob < CONFIDENCE_THRESHOLD:
        decision = DEFAULT_LOW_CONF_LABEL
        confidence_text = f"Safety protocol applied (certainty {top_prob*100:.1f}% &lt; 65%)"
    else:
        decision = ID2LABEL[top_idx]
        confidence_text = f"Certainty: {top_prob*100:.1f}%"

    meta = TRIAGE_META[decision]

    card = f"""
    <div class="st-card" style="border-left-color: {meta['border_color']};">
        <div class="st-card-top">
            <span class="st-badge" style="background:{meta['badge_bg']}; color:{meta['badge_text']};">
                {meta['badge']}
            </span>
            <span class="st-confidence">{confidence_text}</span>
        </div>
        <h2 class="st-headline">{meta['headline']}</h2>
        <p class="st-note">{meta['note']}</p>
        <div class="st-action-box">
            <p class="st-action-label">Recommended next step</p>
            <p class="st-action-text">{meta['action']}</p>
        </div>
    </div>
    """
    return card, build_distribution_html(prob_dict)


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --st-coral: #FF5C5C;
    --st-orange: #FF9D52;
    --st-green: #5EA829;
    --st-ink: #17140F;
    --st-paper: #FFFFFF;
    --st-mist: #F6F4F0;
    --st-line: #E4E0D8;
    --st-muted: #6B6558;
    --st-hero-bg: #17140F;
    --st-hero-line: rgba(255,255,255,0.12);
    --st-hero-muted: rgba(245,243,238,0.62);
}

body, html, .gradio-container {
    background-color: var(--st-mist) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--st-ink) !important;
}
.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
    padding: 0 20px 32px !important;
}

/* ---- Header ---- */
.st-header {
    position: sticky;
    top: 0;
    z-index: 20;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    background: var(--st-mist);
    padding: 18px 0 16px;
    border-bottom: 1px solid var(--st-line);
    margin-bottom: 0;
}
/* Thin tri-color rule sitting between the header and the hero band */
.st-brand-rule {
    height: 4px;
    background: linear-gradient(to right,
        var(--st-coral) 0%, var(--st-coral) 33.3%,
        var(--st-orange) 33.3%, var(--st-orange) 66.6%,
        var(--st-green) 66.6%, var(--st-green) 100%);
}

/* ---- Hero: full-bleed dark band, breaks out of the centered column ---- */
.st-hero-band {
    position: relative;
    margin: 0 -20px;
    background: var(--st-hero-bg);
    overflow: hidden;
    padding: 46px 20px 54px;
    margin-bottom: 34px;
}
.st-hero-shapes {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    pointer-events: none;
}
.st-pent {
    position: absolute;
    clip-path: polygon(28% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 28%);
}
.st-pent.p1 { width: 210px; height: 210px; top: -70px; right: 14%; background: var(--st-coral); opacity: 0.9; transform: rotate(8deg); }
.st-pent.p2 { width: 170px; height: 170px; top: -40px; right: 4%; background: var(--st-orange); opacity: 0.85; transform: rotate(8deg); }
.st-pent.p3 { width: 150px; height: 150px; top: 30px; right: -30px; background: var(--st-green); opacity: 0.85; transform: rotate(8deg); }
.st-pulse {
    position: absolute;
    left: 0;
    bottom: 0;
    width: 140%;
    height: 22px;
    opacity: 0.4;
}
.st-hero-inner {
    position: relative;
    max-width: 1080px;
    margin: 0 auto;
}
.st-hero-inner h1 {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    line-height: 1.12;
    margin: 0 0 14px 0;
    color: #FDFCFA;
    max-width: 15ch;
}
.st-hero-inner p {
    font-size: 1rem;
    color: var(--st-hero-muted);
    line-height: 1.65;
    max-width: 52ch;
    margin: 0 0 26px 0;
}
.st-steps {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
}
.st-step {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    background: rgba(255,255,255,0.06);
    border: 1px solid var(--st-hero-line);
    border-radius: 999px;
    padding: 7px 14px 7px 8px;
    font-size: 0.85rem;
    color: #F3F1EC;
    font-weight: 500;
}
.st-step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    font-size: 0.72rem;
    font-weight: 700;
    color: #17140F;
}
.st-step:nth-child(1) .st-step-num { background: var(--st-coral); }
.st-step:nth-child(3) .st-step-num { background: var(--st-orange); }
.st-step:nth-child(5) .st-step-num { background: var(--st-green); }
.st-step-connector {
    color: var(--st-hero-line);
    font-size: 1rem;
}
@media (max-width: 680px) {
    .st-hero-inner h1 { font-size: 1.9rem; }
    .st-pent.p1, .st-pent.p2, .st-pent.p3 { opacity: 0.55; }
}
.st-header img {
    height: 44px;
    width: auto;
    object-fit: contain;
    display: block;
}
.st-tagline {
    font-size: 0.85rem;
    color: var(--st-muted);
    margin-top: 2px;
}
.st-emergency {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--st-ink);
    color: #FFFFFF !important;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 10px 18px;
    border-radius: 999px;
    text-decoration: none;
    white-space: nowrap;
    box-shadow: 0 2px 8px rgba(255, 92, 92, 0.25);
}
.st-emergency .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--st-coral);
    margin-right: 2px;
}

/* ---- Section labels ---- */
.st-label {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--st-ink);
    margin: 0 0 10px 2px;
}
.st-helper {
    font-size: 0.85rem;
    color: var(--st-muted);
    line-height: 1.5;
    margin: 8px 2px 0;
}
.st-disclaimer {
    font-size: 0.82rem;
    color: var(--st-muted);
    line-height: 1.5;
    margin: 12px 2px 0;
    padding: 10px 14px;
    background: var(--st-paper);
    border: 1.5px solid var(--st-line);
    border-radius: 6px;
}

/* ---- Inputs & buttons ---- */
textarea {
    border-radius: 8px !important;
    border: 1.5px solid var(--st-line) !important;
    background-color: var(--st-paper) !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    color: var(--st-ink) !important;
}
textarea:focus {
    border-color: var(--st-coral) !important;
    box-shadow: 0 0 0 3px rgba(255, 92, 92, 0.15) !important;
    outline: none !important;
}
button.primary {
    background-color: var(--st-coral) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: 1.5px solid var(--st-coral) !important;
    transition: background-color 0.15s ease, transform 0.05s ease !important;
}
button.primary:hover {
    background-color: #E84A4A !important;
}
button.primary:active {
    transform: scale(0.98);
}
button.secondary {
    border-radius: 8px !important;
    background-color: var(--st-paper) !important;
    color: var(--st-ink) !important;
    border: 1.5px solid var(--st-ink) !important;
    font-weight: 600 !important;
}

/* ---- Result states ---- */
.st-empty {
    background: var(--st-paper);
    border: 1.5px dashed var(--st-line);
    border-radius: 6px;
    padding: 40px 20px;
    text-align: center;
    color: var(--st-muted);
}
.st-empty p { margin: 0; font-size: 0.98rem; }

.st-error {
    background: #FFE3E1;
    border: 1.5px solid var(--st-coral);
    border-radius: 6px;
    padding: 18px 20px;
    color: #A0281F;
    font-size: 0.92rem;
}

@keyframes st-fade-up {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: none; }
}
.st-card {
    background: var(--st-paper);
    border: 1.5px solid var(--st-ink);
    border-left: 6px solid;
    border-radius: 6px;
    padding: 26px;
    animation: st-fade-up 0.35s ease;
}
.st-card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 14px;
}
.st-badge {
    font-weight: 700;
    font-size: 0.78rem;
    padding: 5px 12px;
    border-radius: 999px;
}
.st-confidence {
    font-size: 0.82rem;
    color: var(--st-muted);
    font-weight: 500;
}
.st-headline {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    margin: 0 0 10px 0;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--st-ink);
    letter-spacing: -0.2px;
}
.st-note {
    margin: 0 0 18px 0;
    font-size: 0.96rem;
    color: #4B4B4B;
    line-height: 1.6;
}
.st-action-box {
    background: var(--st-mist);
    border: 1.5px solid var(--st-line);
    border-radius: 6px;
    padding: 14px 16px;
}
.st-action-label {
    margin: 0;
    font-size: 0.85rem;
    color: var(--st-ink);
    font-weight: 700;
}
.st-action-text {
    margin: 4px 0 0 0;
    font-size: 0.96rem;
    color: #333333;
    line-height: 1.5;
}

.st-dist-empty {
    font-size: 0.85rem;
    color: var(--st-muted);
    line-height: 1.5;
    margin: 4px 2px 0;
}
.st-dist {
    display: flex;
    flex-direction: column;
    gap: 14px;
    background: var(--st-paper);
    border: 1.5px solid var(--st-line);
    border-radius: 6px;
    padding: 20px 22px;
}
.st-dist-row { width: 100%; }
.st-dist-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.86rem;
    font-weight: 600;
    color: var(--st-ink);
    margin-bottom: 6px;
}
.st-dist-pct {
    font-weight: 700;
    color: var(--st-muted);
}
.st-dist-track {
    height: 9px;
    background: var(--st-mist);
    border: 1px solid var(--st-line);
    border-radius: 999px;
    overflow: hidden;
}
.st-dist-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease;
}

.st-footer {
    margin-top: 40px;
    padding-top: 18px;
    border-top: 1px solid var(--st-line);
    text-align: center;
    font-size: 0.8rem;
    color: var(--st-muted);
}
"""

HEAD_HTML = f'<link rel="icon" type="image/png" href="{LOGO_URI}">' if LOGO_URI else ""

with gr.Blocks(css=custom_css, title="Symptrack", head=HEAD_HTML) as demo:

    
    gr.HTML(
        f"""
        <div class="st-header">
            <div style="display:flex; align-items:center; gap:14px;">
                <img src="{LOGO_URI}" alt="Symptrack logo"/>
                <span class="st-tagline">Clinical triage guidance, not a diagnosis</span>
            </div>
            <a href="tel:997" class="st-emergency">
                <span class="dot"></span>
                Emergency 997
            </a>
        </div>
        <div class="st-brand-rule"></div>
        <div class="st-hero-band">
            <div class="st-hero-shapes">
                <span class="st-pent p1"></span>
                <span class="st-pent p2"></span>
                <span class="st-pent p3"></span>
                <svg class="st-pulse" viewBox="0 0 800 24" preserveAspectRatio="none">
                    <polyline points="0,12 120,12 145,4 170,20 195,12 260,12 285,6 305,18 325,12 800,12"
                        fill="none" stroke="#F5F3EE" stroke-width="1.5"/>
                </svg>
            </div>
            <div class="st-hero-inner">
                <h1>Know your next step, faster.</h1>
                <p>Describe how you feel in your own words. Symptrack points you toward emergency care, a doctor visit, or rest at home, so you're not left guessing what to do.</p>
                <div class="st-steps">
                    <span class="st-step"><span class="st-step-num">1</span>Describe your symptoms</span>
                    <span class="st-step-connector">›</span>
                    <span class="st-step"><span class="st-step-num">2</span>Symptrack reads the pattern</span>
                    <span class="st-step-connector">›</span>
                    <span class="st-step"><span class="st-step-num">3</span>Get clear next-step guidance</span>
                </div>
            </div>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # Left Panel: User Input
        with gr.Column(scale=5):
            gr.HTML('<p class="st-label">Describe your symptoms</p>')
            symptom_input = gr.Textbox(
                show_label=False,
                placeholder="Describe symptoms in your own words — onset, severity, location, sensations...",
                lines=5,
                elem_id="symptom-input",
            )
            gr.HTML(
                '<p class="st-helper">Mentioning when it started, how it has changed, and where it hurts helps the match quality.</p>'
                '<p class="st-disclaimer">Symptrack gives triage guidance based on described symptoms. It cannot diagnose you and is not a substitute for professional medical advice. If you are in doubt, seek care.</p>'
            )

            with gr.Row():
                clear_btn = gr.Button("Reset", variant="secondary")
                submit_btn = gr.Button("Check guidance", variant="primary")

            gr.HTML('<p class="st-label" style="margin-top:22px;">Quick scenarios</p>')
            gr.Examples(
                examples=[
                    ["I have sharp chest pressure radiating to my left jaw, accompanied by sudden shortness of breath and cold sweat."],
                    ["I've had a persistent sore throat and dry cough for four days, with mild fatigue and a 37.8°C temperature."],
                    ["I have a mild localized skin rash on my forearms after gardening, with occasional dry itching but no pain."],
                ],
                inputs=symptom_input,
                label=None,
            )

        
        with gr.Column(scale=5):
            gr.HTML('<p class="st-label">Assessment guidance</p>')
            result_card = gr.HTML(EMPTY_STATE)
            gr.HTML('<p class="st-label" style="margin-top:22px;">Likelihood breakdown</p>')
            prob_label = gr.HTML(DIST_EMPTY)

    gr.HTML(
        """
        <footer class="st-footer">
            Symptrack is an academic research prototype built on ClinicalBERT. This tool provides triage sorting indicators, not medical diagnoses.
        </footer>
        """
    )

    submit_btn.click(
        fn=run_triage,
        inputs=symptom_input,
        outputs=[result_card, prob_label],
    )

    clear_btn.click(
        fn=lambda: ("", EMPTY_STATE, DIST_EMPTY),
        inputs=[],
        outputs=[symptom_input, result_card, prob_label],
    )

    
    demo.load(
        None,
        None,
        None,
        js="""
        () => {
            const ta = document.querySelector('#symptom-input textarea');
            if (ta) ta.focus();
        }
        """,
    )

if __name__ == "__main__":
    from pyngrok import ngrok

public_url = ngrok.connect(7860)
print(f"Public demo URL: {public_url}")

demo.launch(share=False)