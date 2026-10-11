"""Smoke test: verify the ML stack is wired correctly."""

import torch
from transformer_lens import HookedTransformer

print("=" * 60)
print("🧠 PRISM — Environment Smoke Test")
print("=" * 60)

print(f"\n✅ PyTorch version:  {torch.__version__}")
print(f"✅ CUDA available:   {torch.cuda.is_available()}")

print("\n⏳ Loading GPT-2 small via TransformerLens...")
print("   (First run downloads ~500MB — be patient)")
model = HookedTransformer.from_pretrained("gpt2")

print(f"\n✅ Model loaded:     {model.cfg.model_name}")
print(f"   Layers:           {model.cfg.n_layers}")
print(f"   Hidden dim:       {model.cfg.d_model}")
print(f"   Attention heads:  {model.cfg.n_heads}")

prompt = "The future of artificial intelligence is"
print(f"\n🎯 Prompt: '{prompt}'")

output = model.generate(prompt, max_new_tokens=20, temperature=0.7)
print(f"\n📝 Output: {output}")

print("\n🚀 Environment is ready. Welcome to mechanistic interpretability.")