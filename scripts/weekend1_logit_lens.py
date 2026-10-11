import torch
from transformer_lens import HookedTransformer

print("Loading GPT-2...")
model = HookedTransformer.from_pretrained("gpt2", device="cpu")

prompt = "The Eiffel Tower is located in the city of"
print(f"\n🎯 Prompt: '{prompt}'\n")

# Run model and cache all internal layers
logits, cache = model.run_with_cache(prompt)

print("=" * 65)
print("🧠 LOGIT LENS: What is the AI guessing at EACH layer?")
print("=" * 65)

# Inspect the final token position across all 12 layers
for layer in range(model.cfg.n_layers):
    # 1. Grab full residual stream tensor [batch=1, seq=10, d_model=768]
    resid_post = cache["resid_post", layer]

    # 2. Project layer activations through LayerNorm & Unembedding matrix
    scaled_resid = model.ln_final(resid_post)
    layer_logits = model.unembed(scaled_resid)  # Shape: [1, seq, d_vocab]

    # 3. Focus on the very LAST token position (" of")
    last_token_logits = layer_logits[0, -1]

    # 4. Compute top prediction & confidence
    probs = torch.softmax(last_token_logits, dim=-1)
    top_token_id = last_token_logits.argmax(dim=-1).item()
    top_token_str = model.to_string(top_token_id)
    confidence = probs[top_token_id].item() * 100

    print(
        f"Layer {layer:2d} | Top Prediction: '{top_token_str.strip():15}' | Confidence: {confidence:5.1f}%"
    )

print("=" * 65)