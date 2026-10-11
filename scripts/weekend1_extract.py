import torch
import plotly.express as px
from transformer_lens import HookedTransformer

# 1. Load our model
print("Loading model...")
model = HookedTransformer.from_pretrained("gpt2", device="cpu")

# 2. Define a sentence to "scan"
prompt = "The Eiffel Tower is located in the city of"

# 3. Run the model and capture EVERY internal thought (activation)
# 'run_with_cache' is the magic function of TransformerLens
print(f"Scanning prompt: {prompt}")
logits, cache = model.run_with_cache(prompt)

# 4. Let's look at the 'Residual Stream' at Layer 6
# This is the "main bus" of the model's brain
layer_to_examine = 6
resid_post = cache["resid_post", layer_to_examine] 

print(f"\n✅ Captured activations at Layer {layer_to_examine}!")
print(f"Shape of activations: {resid_post.shape}") 
# [batch, position, d_model] -> [1, 10 tokens, 768 dimensions]

# 5. Let's find out which token the model thinks is most important
# We'll visualize the "internal energy" (Norm) of each token's vector
token_labels = model.to_str_tokens(prompt)
norms = resid_post[0].norm(dim=-1).detach().numpy()

print("\nInternal 'Energy' per Token:")
for token, norm in zip(token_labels, norms):
    print(f"Token: {token:12} | Magnitude: {norm:.2f}")

print("\n🚀 Next Step: We will find the 'Paris' neuron.")