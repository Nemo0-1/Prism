import warnings

warnings.filterwarnings("ignore")  # Mute all warnings

import torch
from transformer_lens import HookedTransformer

print("🧠 PRISM — Activation Steering Strength Sweep")
print("=" * 70)
print("Loading model...")
model = HookedTransformer.from_pretrained("gpt2", device="cpu")

prompt = "The Eiffel Tower is located in the city of"

# Calculate the concept vector: (Paris - London)
paris_id = model.to_single_token(" Paris")
london_id = model.to_single_token(" London")
steering_vector = model.W_U[:, paris_id] - model.W_U[:, london_id]

print(f"\n🎯 Prompt: '{prompt}'")
print("=" * 70)
print(f"{'STRENGTH':<12} | {'GENERATED OUTPUT'}")
print("-" * 70)

# Sweep steering intensity from 0.0 to 25.0
strengths = [0.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0]

for multiplier in strengths:

    # Define dynamic hook with current strength multiplier
    def steer_hook(activations, hook):
        activations[:, -1, :] += steering_vector * multiplier
        return activations

    # Run inference with modified activations
    with model.hooks(fwd_hooks=[("blocks.6.hook_resid_post", steer_hook)]):
        output = model.generate(
            prompt, max_new_tokens=4, temperature=0.0, verbose=False
        )

    # Clean up output string for pretty display
    completion = output[len(prompt) :].replace("\n", " ")
    print(f"Multiplier {multiplier:<4.1f} | {completion}")

print("=" * 70)
print("🔬 OBSERVE THE TIPPING POINT: Watch 'London' transition to 'Paris'!")