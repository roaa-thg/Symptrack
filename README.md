# Symptrack — AI Symptom Triage

**🌐 Website:** https://roaa-thg.github.io/Symptrack/

Symptrack reads a patient's symptoms written in plain language and sorts them into one of three triage levels:

| 🔴 Emergency Room | 🟠 Doctor | 🟢 Self-care |
|---|---|---|
| May need immediate care | Needs attention, not urgent | Can be managed at home |

It is built on a fine-tuned **ClinicalBERT** model, with a confidence threshold that defaults uncertain cases to the safest category (ER).

---

## Tools & Libraries

| Purpose | Tools / Libraries |
|---|---|
| Model | `transformers` (Bio_ClinicalBERT), `torch` |
| Interface | `gradio` |
| Data & metrics | `pandas`, `numpy`, `scikit-learn` (splits, macro-F1, Cohen's Kappa, calibration) |
| EDA / plots | `matplotlib`, `seaborn` |
| Dataset download | `kagglehub` |
| Baseline comparison | `tensorflow` / `keras` (GRU), GloVe & BioWordVec embeddings |
| Environment | Python 3.10+, Jupyter / Google Colab |

---

## How it works

1. **Describe your symptoms** — type how you feel, like a chat message.
2. **Symptrack reads the pattern** — a fine-tuned ClinicalBERT model analyzes the text.
3. **Get clear guidance** — a triage category + a medical disclaimer.

---

## Results

| Metric | Score |
|---|---|
| Macro-F1 | **0.979** |
| ER Recall | 0.966 |
| Calibration (ECE) | 0.017 |
| Accuracy | 0.983 |

### Model comparison

| Model | Macro-F1 | ER Recall |
|---|---|---|
| GRU (no embeddings) | 0.255 | 0.000 |
| GRU + GloVe | 0.931 | 0.897 |
| GRU + BioWordVec | 0.926 | 0.931 |
| **ClinicalBERT** | **0.979** | **0.966** |

ClinicalBERT won because contextual, medically pre-trained understanding handled short and ambiguous inputs that the static-embedding GRUs mis-triaged.

---

## Repository contents

| File | Description |
|---|---|
| `app.py` | Gradio interface for live triage |
| `Symptrack_Main_ClinicalBERT_CLEAN.ipynb` | Full pipeline: cleaning → split → training → evaluation → calibration |
| `Symptrack_Model_Comparison_CLEAN.ipynb` | GRU baselines vs. ClinicalBERT |
| `config.json`, `inference_config.json` | Model + inference settings (threshold 0.65, default = ER) |
| `requirements.txt` | Dependencies |
| `Symptrack.html` | Landing page |

---

## How we built it

We mapped 24 disease labels into three triage categories using a **CTAS-based protocol** (labels checked independently by three annotators, Cohen's Kappa = 0.76), fine-tuned ClinicalBERT with a weighted loss to handle class imbalance, added short balanced examples to fix short-input generalization, and set a confidence threshold that sends uncertain cases to ER.

**Team AI-titans — Symptrack Capstone Project**
