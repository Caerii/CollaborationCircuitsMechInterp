"""
Linear probing pipeline for entity classification.

We train linear probes at each layer to predict:
1. Entity type (User/Self/Other) - who is the current speaker?
2. Target (User/Other) - who is being addressed?

This tells us WHERE and HOW the model represents entity information.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import json
from pathlib import Path

from .config import EXP_CFG, RESULTS_DIR, ENTITY_TYPES


@dataclass
class ProbeResult:
    """Results from training a single probe."""
    layer: int
    task: str  # "entity_type" or "target"
    train_accuracy: float
    test_accuracy: float
    confusion_matrix: np.ndarray
    class_accuracies: Dict[str, float]
    

class LinearProbe(nn.Module):
    """Simple linear probe for classification."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProbe(nn.Module):
    """MLP probe with one hidden layer (for comparison)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProbingPipeline:
    """
    Complete pipeline for probing entity representations.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 3,  # User, Self, Other
        probe_type: str = "linear",  # "linear" or "mlp"
        mlp_hidden: int = 256,
        device: str = "cuda"
    ):
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.probe_type = probe_type
        self.mlp_hidden = mlp_hidden
        self.device = device
        
        self.probes: Dict[int, nn.Module] = {}  # layer -> probe
        self.results: Dict[int, ProbeResult] = {}
        
    def create_probe(self, layer: int) -> nn.Module:
        """Create a probe for a specific layer."""
        if self.probe_type == "linear":
            probe = LinearProbe(self.hidden_dim, self.num_classes)
        else:
            probe = MLPProbe(self.hidden_dim, self.mlp_hidden, self.num_classes)
        
        self.probes[layer] = probe.to(self.device)
        return self.probes[layer]
    
    def train_probe(
        self,
        layer: int,
        activations: torch.Tensor,  # [n_samples, hidden_dim]
        labels: torch.Tensor,        # [n_samples]
        task_name: str = "entity_type",
        epochs: int = EXP_CFG.epochs,
        lr: float = EXP_CFG.learning_rate,
        batch_size: int = EXP_CFG.batch_size,
        verbose: bool = True
    ) -> ProbeResult:
        """
        Train a probe on activations for a specific layer.
        """
        # Create probe if not exists
        if layer not in self.probes:
            self.create_probe(layer)
        
        probe = self.probes[layer]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            activations.numpy(),
            labels.numpy(),
            test_size=1 - EXP_CFG.train_split,
            random_state=EXP_CFG.seed,
            stratify=labels.numpy()
        )
        
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.long, device=self.device)
        y_test = torch.tensor(y_test, dtype=torch.long, device=self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        
        # Training loop
        probe.train()
        n_batches = (len(X_train) + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]
            
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = probe(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / n_batches
                print(f"  Layer {layer}, Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluation
        probe.eval()
        with torch.no_grad():
            train_preds = probe(X_train).argmax(dim=1).cpu().numpy()
            test_preds = probe(X_test).argmax(dim=1).cpu().numpy()
        
        train_acc = accuracy_score(y_train.cpu().numpy(), train_preds)
        test_acc = accuracy_score(y_test.cpu().numpy(), test_preds)
        conf_mat = confusion_matrix(y_test.cpu().numpy(), test_preds)
        
        # Per-class accuracy
        class_names = list(ENTITY_TYPES.keys())
        class_accs = {}
        for idx, name in enumerate(class_names):
            mask = y_test.cpu().numpy() == idx
            if mask.sum() > 0:
                class_accs[name] = (test_preds[mask] == idx).mean()
            else:
                class_accs[name] = 0.0
        
        result = ProbeResult(
            layer=layer,
            task=task_name,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            confusion_matrix=conf_mat,
            class_accuracies=class_accs
        )
        
        self.results[layer] = result
        
        if verbose:
            print(f"  Layer {layer}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")
            print(f"  Per-class: {class_accs}")
        
        return result
    
    def train_all_layers(
        self,
        layer_activations: Dict[int, torch.Tensor],  # layer -> [n_samples, hidden_dim]
        labels: torch.Tensor,
        task_name: str = "entity_type",
        **kwargs
    ) -> Dict[int, ProbeResult]:
        """Train probes for all layers."""
        print(f"\n{'='*50}")
        print(f"Training probes for task: {task_name}")
        print(f"{'='*50}")
        
        results = {}
        for layer in sorted(layer_activations.keys()):
            print(f"\nLayer {layer}:")
            result = self.train_probe(
                layer=layer,
                activations=layer_activations[layer],
                labels=labels,
                task_name=task_name,
                **kwargs
            )
            results[layer] = result
        
        return results
    
    def get_probe_direction(self, layer: int) -> Optional[torch.Tensor]:
        """
        Get the learned probe direction (for linear probes).
        This can be used for activation steering.
        
        Returns weight matrix [num_classes, hidden_dim]
        """
        if layer not in self.probes:
            return None
        
        probe = self.probes[layer]
        if isinstance(probe, LinearProbe):
            return probe.linear.weight.detach().cpu()
        else:
            return None  # MLP doesn't have a simple direction
    
    def save_results(self, path: Optional[Path] = None):
        """Save probe results to JSON."""
        if path is None:
            path = RESULTS_DIR / "probe_results.json"
        
        # Convert results to serializable format
        data = {}
        for layer, result in self.results.items():
            data[str(layer)] = {
                "layer": result.layer,
                "task": result.task,
                "train_accuracy": result.train_accuracy,
                "test_accuracy": result.test_accuracy,
                "confusion_matrix": result.confusion_matrix.tolist(),
                "class_accuracies": result.class_accuracies
            }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {path}")
    
    def save_probes(self, path: Optional[Path] = None):
        """Save trained probe weights."""
        if path is None:
            path = RESULTS_DIR / "probes.pt"
        
        state = {layer: probe.state_dict() for layer, probe in self.probes.items()}
        torch.save(state, path)
        print(f"Probes saved to {path}")
    
    def load_probes(self, path: Optional[Path] = None):
        """Load trained probe weights."""
        if path is None:
            path = RESULTS_DIR / "probes.pt"
        
        state = torch.load(path)
        for layer, weights in state.items():
            if layer not in self.probes:
                self.create_probe(layer)
            self.probes[layer].load_state_dict(weights)
        
        print(f"Probes loaded from {path}")


def compute_representation_similarity(
    activations1: torch.Tensor,
    activations2: torch.Tensor,
    method: str = "cosine"
) -> float:
    """
    Compute similarity between two sets of activations.
    
    Args:
        activations1: [n1, hidden_dim]
        activations2: [n2, hidden_dim]
        method: "cosine" or "cka"
        
    Returns:
        Similarity score
    """
    if method == "cosine":
        # Mean activation similarity
        mean1 = activations1.mean(dim=0)
        mean2 = activations2.mean(dim=0)
        cos_sim = torch.nn.functional.cosine_similarity(
            mean1.unsqueeze(0), 
            mean2.unsqueeze(0)
        )
        return cos_sim.item()
    
    elif method == "cka":
        # Centered Kernel Alignment
        # Simplified implementation
        def centering_matrix(n):
            return torch.eye(n) - torch.ones(n, n) / n
        
        n1, n2 = activations1.size(0), activations2.size(0)
        
        # Compute Gram matrices
        K = activations1 @ activations1.T
        L = activations2 @ activations2.T
        
        # Center
        H1 = centering_matrix(n1)
        H2 = centering_matrix(n2)
        K_centered = H1 @ K @ H1
        L_centered = H2 @ L @ H2
        
        # If different sizes, need to handle (simplified: use mean)
        if n1 != n2:
            return compute_representation_similarity(activations1, activations2, "cosine")
        
        # HSIC
        hsic_kl = (K_centered * L_centered).sum()
        hsic_kk = (K_centered * K_centered).sum()
        hsic_ll = (L_centered * L_centered).sum()
        
        cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-8)
        return cka.item()
    
    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_entity_separation(
    layer_activations: Dict[int, torch.Tensor],
    labels: torch.Tensor,
    method: str = "cosine"
) -> Dict[int, Dict[str, float]]:
    """
    Analyze how well entity representations are separated at each layer.
    
    Returns similarity matrix between entity types for each layer.
    """
    entity_names = list(ENTITY_TYPES.keys())
    results = {}
    
    for layer, acts in layer_activations.items():
        # Split by entity type
        entity_acts = {}
        for name, idx in ENTITY_TYPES.items():
            mask = labels == idx
            entity_acts[name] = acts[mask]
        
        # Compute pairwise similarities
        similarities = {}
        for i, name1 in enumerate(entity_names):
            for j, name2 in enumerate(entity_names):
                if i <= j:  # Upper triangle
                    sim = compute_representation_similarity(
                        entity_acts[name1],
                        entity_acts[name2],
                        method=method
                    )
                    key = f"{name1}_vs_{name2}"
                    similarities[key] = sim
        
        results[layer] = similarities
    
    return results

