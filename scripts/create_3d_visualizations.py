"""
3D Animated Visualizations for Entity Representations
======================================================

Creates compelling animated GIFs showing:
1. Rotating 3D PCA projections of entity clusters
2. Layer-by-layer evolution of representations
3. Entity separation trajectories across layers

These visualizations make the science more intuitive and compelling!
"""

import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
VIZ_DIR = RESULTS_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("CREATING 3D ANIMATED VISUALIZATIONS")
print("=" * 70)


def load_activations():
    """Load activation data with proper label mapping."""
    print("\nLoading activations...")
    data = torch.load(DATA_DIR / "activations.pt", map_location="cpu")
    
    # Convert labels
    label_map = {0: "user", 1: "self", 2: "other"}
    raw_labels = data["labels"]
    if isinstance(raw_labels, torch.Tensor):
        raw_labels = raw_labels.numpy()
    labels = np.array([label_map[int(l)] for l in raw_labels])
    
    # Normalize activation keys
    activations = {}
    for k, v in data["activations"].items():
        layer_idx = int(k) if isinstance(k, str) else k
        if isinstance(v, torch.Tensor):
            activations[layer_idx] = v.float().numpy()
        else:
            activations[layer_idx] = np.array(v, dtype=np.float32)
    
    print(f"  Loaded {len(labels)} samples across {len(activations)} layers")
    return activations, labels


def create_rotating_3d_pca(activations, labels, layer, output_path, n_frames=120):
    """Create a rotating 3D PCA visualization for a single layer."""
    
    print(f"\nCreating rotating 3D PCA for Layer {layer}...")
    
    acts = activations[layer]
    
    # PCA to 3D
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(acts)
    
    # Colors with nice gradients
    colors = {
        "user": "#2ecc71",   # Green
        "self": "#3498db",   # Blue  
        "other": "#e74c3c"   # Red
    }
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each entity type
    scatters = {}
    for entity in ["user", "self", "other"]:
        mask = labels == entity
        scatters[entity] = ax.scatter(
            coords_3d[mask, 0], coords_3d[mask, 1], coords_3d[mask, 2],
            c=colors[entity], label=entity.capitalize(), alpha=0.6, s=40,
            edgecolors='white', linewidth=0.5
        )
    
    # Styling
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=10)
    ax.set_title(f'Entity Representations - Layer {layer}\n3D PCA Projection', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    
    # Set consistent axis limits
    max_range = np.abs(coords_3d).max() * 1.1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    # Dark background for better contrast
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3, color='white')
    
    # White text
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(colors='white')
    for text in ax.legend_.get_texts():
        text.set_color('white')
    
    def animate(frame):
        ax.view_init(elev=20, azim=frame * 3)  # Rotate 3 degrees per frame
        return scatters.values()
    
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
    anim.save(output_path, writer='pillow', fps=24, dpi=100)
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_layer_evolution_animation(activations, labels, layers, output_path, n_frames_per_layer=30):
    """Create animation showing how representations evolve across layers."""
    
    print(f"\nCreating layer evolution animation...")
    
    colors = {
        "user": "#2ecc71",
        "self": "#3498db", 
        "other": "#e74c3c"
    }
    
    # Compute 3D PCA for each layer
    all_pca_coords = {}
    for layer in layers:
        acts = activations[layer]
        pca = PCA(n_components=3)
        all_pca_coords[layer] = pca.fit_transform(acts)
    
    # Find global axis limits
    all_coords = np.concatenate(list(all_pca_coords.values()))
    max_range = np.abs(all_coords).max() * 1.1
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dark theme
    ax.set_facecolor('#0f0f23')
    fig.patch.set_facecolor('#0f0f23')
    
    total_frames = len(layers) * n_frames_per_layer
    
    def animate(frame):
        ax.clear()
        
        # Determine which layer and rotation angle
        layer_idx = frame // n_frames_per_layer
        rotation_frame = frame % n_frames_per_layer
        
        if layer_idx >= len(layers):
            layer_idx = len(layers) - 1
        
        layer = layers[layer_idx]
        coords_3d = all_pca_coords[layer]
        
        # Progress through transition
        if layer_idx < len(layers) - 1 and rotation_frame > n_frames_per_layer * 0.7:
            # Blend towards next layer
            next_layer = layers[layer_idx + 1]
            next_coords = all_pca_coords[next_layer]
            blend = (rotation_frame - n_frames_per_layer * 0.7) / (n_frames_per_layer * 0.3)
            coords_3d = coords_3d * (1 - blend) + next_coords * blend
        
        # Plot
        for entity in ["user", "self", "other"]:
            mask = labels == entity
            ax.scatter(
                coords_3d[mask, 0], coords_3d[mask, 1], coords_3d[mask, 2],
                c=colors[entity], label=entity.capitalize(), alpha=0.7, s=35,
                edgecolors='white', linewidth=0.3
            )
        
        # Styling
        ax.set_xlabel('PC1', fontsize=10, color='white')
        ax.set_ylabel('PC2', fontsize=10, color='white')
        ax.set_zlabel('PC3', fontsize=10, color='white')
        ax.set_title(f'Entity Representation Evolution\nLayer {layer}', 
                    fontsize=14, fontweight='bold', color='white')
        
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.2, color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper left', fontsize=9, facecolor='#1a1a2e', labelcolor='white')
        
        # Slow rotation
        ax.view_init(elev=25, azim=frame * 2)
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=80, blit=False)
    anim.save(output_path, writer='pillow', fps=15, dpi=100)
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_centroid_trajectory(activations, labels, layers, output_path):
    """Create animation showing entity centroid trajectories through layers."""
    
    print(f"\nCreating centroid trajectory animation...")
    
    colors = {
        "user": "#2ecc71",
        "self": "#3498db",
        "other": "#e74c3c"
    }
    
    # Compute centroids for each layer
    centroids = {entity: [] for entity in ["user", "self", "other"]}
    
    # Use consistent PCA across all layers
    all_acts = np.concatenate([activations[l] for l in layers])
    pca = PCA(n_components=3)
    pca.fit(all_acts)
    
    for layer in layers:
        acts = activations[layer]
        coords_3d = pca.transform(acts)
        
        for entity in ["user", "self", "other"]:
            mask = labels == entity
            centroid = coords_3d[mask].mean(axis=0)
            centroids[entity].append(centroid)
    
    # Convert to arrays
    for entity in centroids:
        centroids[entity] = np.array(centroids[entity])
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dark theme
    ax.set_facecolor('#0f0f23')
    fig.patch.set_facecolor('#0f0f23')
    
    # Find axis limits
    all_centroids = np.concatenate(list(centroids.values()))
    max_range = np.abs(all_centroids).max() * 1.3
    
    n_frames = 180
    
    def animate(frame):
        ax.clear()
        
        # How many layers to show (gradually reveal trajectory)
        n_layers_shown = min(len(layers), 1 + frame // 15)
        
        for entity in ["user", "self", "other"]:
            traj = centroids[entity][:n_layers_shown]
            
            # Plot trajectory line
            if len(traj) > 1:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                       c=colors[entity], linewidth=3, alpha=0.8)
            
            # Plot points with size indicating layer depth
            for i, point in enumerate(traj):
                size = 50 + i * 30  # Larger for deeper layers
                alpha = 0.3 + 0.7 * (i / max(1, len(traj) - 1))
                ax.scatter(point[0], point[1], point[2],
                          c=colors[entity], s=size, alpha=alpha,
                          edgecolors='white', linewidth=1)
            
            # Label the latest point
            if len(traj) > 0:
                ax.text(traj[-1, 0], traj[-1, 1], traj[-1, 2] + max_range * 0.05,
                       entity.capitalize(), color=colors[entity], fontsize=10, fontweight='bold')
        
        # Add layer labels
        for i in range(n_layers_shown):
            # Add small text near user centroid showing layer number
            pos = centroids["user"][i]
            ax.text(pos[0] + max_range * 0.08, pos[1], pos[2],
                   f'L{layers[i]}', color='white', fontsize=7, alpha=0.6)
        
        ax.set_xlabel('PC1', fontsize=10, color='white')
        ax.set_ylabel('PC2', fontsize=10, color='white')
        ax.set_zlabel('PC3', fontsize=10, color='white')
        ax.set_title('Entity Centroid Trajectories Through Layers\n(Small->Large = Early->Late layers)', 
                    fontsize=12, fontweight='bold', color='white')
        
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.2, color='white')
        ax.tick_params(colors='white')
        
        ax.view_init(elev=20, azim=frame * 2)
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
    anim.save(output_path, writer='pillow', fps=20, dpi=100)
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_separation_sphere(activations, labels, layer, output_path):
    """Project representations onto a sphere to visualize angular separation."""
    
    print(f"\nCreating sphere projection for Layer {layer}...")
    
    acts = activations[layer]
    
    # PCA to 3D, then normalize to unit sphere
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(acts)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(coords_3d, axis=1, keepdims=True)
    sphere_coords = coords_3d / (norms + 1e-8)
    
    colors = {
        "user": "#2ecc71",
        "self": "#3498db",
        "other": "#e74c3c"
    }
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw reference sphere (wireframe)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1, linewidth=0.5)
    
    # Plot points on sphere
    for entity in ["user", "self", "other"]:
        mask = labels == entity
        ax.scatter(
            sphere_coords[mask, 0], sphere_coords[mask, 1], sphere_coords[mask, 2],
            c=colors[entity], label=entity.capitalize(), alpha=0.7, s=40,
            edgecolors='white', linewidth=0.5
        )
    
    # Draw centroid vectors
    for entity in ["user", "self", "other"]:
        mask = labels == entity
        centroid = sphere_coords[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)  # Normalize
        ax.quiver(0, 0, 0, centroid[0], centroid[1], centroid[2],
                 color=colors[entity], arrow_length_ratio=0.1, linewidth=3, alpha=0.9)
    
    # Dark theme
    ax.set_facecolor('#0f0f23')
    fig.patch.set_facecolor('#0f0f23')
    
    ax.set_xlabel('X', fontsize=10, color='white')
    ax.set_ylabel('Y', fontsize=10, color='white')
    ax.set_zlabel('Z', fontsize=10, color='white')
    ax.set_title(f'Angular Separation on Unit Sphere - Layer {layer}\n(Arrows show entity centroid directions)', 
                fontsize=12, fontweight='bold', color='white')
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')
    ax.legend(loc='upper left', fontsize=10, facecolor='#1a1a2e', labelcolor='white')
    
    n_frames = 120
    
    def animate(frame):
        ax.view_init(elev=20, azim=frame * 3)
        return []
    
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
    anim.save(output_path, writer='pillow', fps=24, dpi=100)
    plt.close()
    
    print(f"  Saved: {output_path}")


def main():
    # Load data
    activations, labels = load_activations()
    layers = sorted(activations.keys())
    
    # Key layers for visualization
    key_layers = [0, 20, 35]
    key_layers = [l for l in key_layers if l in layers]
    
    print(f"\nCreating visualizations for layers: {key_layers}")
    print(f"Output directory: {VIZ_DIR}")
    
    # 1. Rotating 3D PCA for each key layer
    for layer in key_layers:
        create_rotating_3d_pca(
            activations, labels, layer,
            VIZ_DIR / f"rotating_3d_layer_{layer}.gif"
        )
    
    # 2. Layer evolution animation
    create_layer_evolution_animation(
        activations, labels, layers,
        VIZ_DIR / "layer_evolution.gif",
        n_frames_per_layer=25
    )
    
    # 3. Centroid trajectory animation
    create_centroid_trajectory(
        activations, labels, layers,
        VIZ_DIR / "centroid_trajectories.gif"
    )
    
    # 4. Sphere projections for key layers
    for layer in [0, 20]:
        if layer in layers:
            create_separation_sphere(
                activations, labels, layer,
                VIZ_DIR / f"sphere_projection_layer_{layer}.gif"
            )
    
    print("\n" + "=" * 70)
    print("3D VISUALIZATIONS COMPLETE!")
    print("=" * 70)
    print(f"\nAll visualizations saved to: {VIZ_DIR}")
    print("\nFiles created:")
    for f in sorted(VIZ_DIR.glob("*.gif")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

