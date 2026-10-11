import os
import warnings

warnings.filterwarnings("ignore")

import plotly.graph_objects as go
import torch
from transformer_lens import HookedTransformer

print("📊 PRISM — Generating Interactive Steering Probability Chart...")
model = HookedTransformer.from_pretrained("gpt2", device="cpu")

prompt = "The Eiffel Tower is located in the city of"

# Concept vector: (Paris - London)
paris_id = model.to_single_token(" Paris")
london_id = model.to_single_token(" London")
steering_vector = model.W_U[:, paris_id] - model.W_U[:, london_id]

strengths = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    15.0,
    20.0,
]
paris_probs = []
london_probs = []

for mult in strengths:

    def steer_hook(activations, hook):
        activations[:, -1, :] += steering_vector * mult
        return activations

    with model.hooks(fwd_hooks=[("blocks.6.hook_resid_post", steer_hook)]):
        logits = model(prompt)
        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)

        paris_probs.append(probs[paris_id].item() * 100)
        london_probs.append(probs[london_id].item() * 100)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=strengths,
        y=paris_probs,
        mode="lines+markers",
        name="Paris Probability (%)",
        line=dict(color="#00FF7F", width=4),
        marker=dict(size=10),
    )
)

fig.add_trace(
    go.Scatter(
        x=strengths,
        y=london_probs,
        mode="lines+markers",
        name="London Probability (%)",
        line=dict(color="#FF4500", width=4, dash="dash"),
        marker=dict(size=10),
    )
)

fig.update_layout(
    title="<b>PRISM Engine: Activation Steering Impact</b><br><sup>Real-Time Shift in Latent Token Probabilities at Layer 6</sup>",
    xaxis_title="Steering Multiplier Strength",
    yaxis_title="Next Token Probability (%)",
    template="plotly_dark",
    font=dict(family="Monospace", size=14),
    hovermode="x unified",
    paper_bgcolor="#111111",
    plot_bgcolor="#1e1e1e",
)

os.makedirs("docs", exist_ok=True)
output_path = "docs/steering_curve.html"
fig.write_html(output_path)
print(f"✅ Chart successfully saved to: {output_path}")