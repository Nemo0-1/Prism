# 🧠 PRISM: Real-Time Mechanistic Interpretability & Activation Steering Engine

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TransformerLens](https://img.shields.io/badge/TransformerLens-1.17-green.svg)](https://github.com/TransformerLensOrg/TransformerLens)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **PRISM** is an open-source mechanistic interpretability engine designed to inspect, decompose, and steer internal latent representations inside Large Language Models (LLMs) during inference—without retraining or fine-tuning model weights.

---

## 🔬 Key Breakthroughs & Empirical Findings

### 1. Logit Lens Layer-by-Layer Decomposition
Using residual stream un-embedding at each transformer layer, we mapped how fact retrieval emerges across attention layers for factual prompts (e.g., *"The Eiffel Tower is located in the city of"*):
- **Layers 0–4 (Syntactic Phase):** Focuses on grammatical structure (`"the"`, `73.3%` confidence).
- **Layers 5–8 (Categorical Phase):** Shifts toward geographic semantic clusters (`"England"`, `"Chicago"`).
- **Layers 9–11 (Factual Locking):** Settles on the target city (`"London"`, native hallucination baseline).

### 2. Real-Time Latent Steering
By computing the concept direction vector in vocabulary space:
$$\vec{v}_{\text{steering}} = W_U[\text{"Paris"}] - W_U[\text{"London"}]$$
and injecting $\vec{v}_{\text{steering}}$ into Layer 6 residual activations ($\text{resid\_post}$), we alter the model's factual output deterministically in real-time.

### 3. Optimal Multiplier Window Discovery
Empirical sweep across multiplier values ($0.0 \le \lambda \le 25.0$):

| Multiplier ($\lambda$) | Generated Completion | Interpretation |
|---|---|---|
| `0.0` | `London, and is` | Baseline Hallucination |
| `2.0 – 5.0` | **`Paris, France.`** | **Optimal Factual Correction Window** |
| `15.0+` | `Paris Paris Paris` | Semantic Over-Steering Collapse |

---

## 🛠 Project Structure

prism/
├── docs/
│   └── steering_curve.html    # Interactive Plotly probability response curve
├── scripts/
│   ├── hello_prism.py          # Environment smoke test
│   ├── weekend1_extract.py     # Residual stream activation extraction
│   ├── weekend1_logit_lens.py  # Layer-by-layer logit lens projection
│   ├── weekend1_steering.py    # Real-time Layer 6 vector injection hook
│   ├── weekend1_sweep.py       # Steering strength multiplier sweep
│   └── weekend1_visualizer.py  # Interactive probability visualizer
├── environment.yml             # Conda environment definition
└── README.md

---

## 🚀 Quickstart

1. Clone repository:
   git clone https://github.com/Nemo0-1/prism.git
   cd prism

2. Activate environment:
   mamba env create -f environment.yml
   mamba activate prism

3. Run activation steering experiment:
   python scripts/weekend1_sweep.py

---

## 📊 Interactive Visualization
To view the interactive probability shift graph, open https://prism-steering-1.vercel.app/
---

## 🎯 Next Research Milestones
- [x] Layer-wise activation extraction & logit lens projection.
- [x] Deterministic single-concept activation steering hooks.
- [ ] Sparse Autoencoders (SAEs) for disentangling monosemantic features.
- [ ] Automated concept steering web dashboard.