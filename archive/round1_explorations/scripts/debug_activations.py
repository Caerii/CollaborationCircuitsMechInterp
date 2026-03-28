"""Debug script to understand the activation data structure."""

import torch
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

print("Loading activations...")
data = torch.load(DATA_DIR / "activations.pt", map_location="cpu")

print(f"\nKeys in data: {data.keys()}")
print(f"\nNumber of labels: {len(data['labels'])}")
print(f"Label distribution: {np.unique(data['labels'], return_counts=True)}")

activations = data['activations']
print(f"\nActivation layers: {list(activations.keys())}")

# Check the actual values
for layer_key in list(activations.keys())[:3]:
    acts = activations[layer_key]
    print(f"\nLayer {layer_key}:")
    print(f"  Type: {type(acts)}")
    print(f"  Shape: {acts.shape if hasattr(acts, 'shape') else 'N/A'}")
    print(f"  Dtype: {acts.dtype if hasattr(acts, 'dtype') else 'N/A'}")
    
    if hasattr(acts, 'numpy'):
        arr = acts.float().numpy()
    else:
        arr = np.array(acts, dtype=np.float32)
    
    print(f"  Min: {arr.min():.6f}")
    print(f"  Max: {arr.max():.6f}")
    print(f"  Mean: {arr.mean():.6f}")
    print(f"  Std: {arr.std():.6f}")
    print(f"  Non-zero elements: {(arr != 0).sum()} / {arr.size}")
    
    # Check per-sample variance
    sample_stds = arr.std(axis=1)
    print(f"  Per-sample std - min: {sample_stds.min():.6f}, max: {sample_stds.max():.6f}")
    
    # Check first few activations
    print(f"  First activation sample[:10]: {arr[0, :10]}")

# Check labels
labels = np.array(data['labels'])
print(f"\nFirst 10 labels: {labels[:10]}")
print(f"Last 10 labels: {labels[-10:]}")

