import torch
from transformer_lens import HookedTransformer

print("Loading GPT-2...")
model = HookedTransformer.from_pretrained("gpt2", device="cpu")

prompt = "The Eiffel Tower is located in the city of"

# 1. Run standard inference (Unsteered)
unsteered_output = model.generate(
    prompt, max_new_tokens=5, temperature=0.0, verbose=False
)
print("=" * 65)
print(f"🔴 ORIGINAL (UNSTEERED) OUTPUT: {unsteered_output}")
print("=" * 65)

# 2. Extract direction vectors for 'Paris' vs 'London'
paris_token_id = model.to_single_token(" Paris")
london_token_id = model.to_single_token(" London")

# Get unembedding vectors (directions in 768-dimensional space)
paris_vector = model.W_U[:, paris_token_id]
london_vector = model.W_U[:, london_token_id]

# The Steering Vector = (Direction of Paris) - (Direction of London)
steering_vector = paris_vector - london_vector


# 3. Create a Hook Function that injects the steering vector at Layer 6
def steer_brain_hook(activation_tensor, hook):
    steering_strength = 20.0
    # Inject the vector into the last token position's residual stream
    activation_tensor[:, -1, :] += steering_vector * steering_strength
    return activation_tensor


# 4. Run inference WITH the hook active at Layer 6 using context manager
with model.hooks(fwd_hooks=[("blocks.6.hook_resid_post", steer_brain_hook)]):
    steered_output = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=0.0,
        verbose=False,
    )

print(f"🟢 STEERED (BRAIN-SURGERY) OUTPUT: {steered_output}")
print("=" * 65)
print(
    "🚀 SUCCESS: You just altered an LLM's internal cognition in real-time!"
)