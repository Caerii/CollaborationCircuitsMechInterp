"""
3D Visualization for Activation Geometry

Beautiful animated visualizations for MATS submissions and papers.
Shows how representations evolve through layers in an intuitive way.

Key visualizations:
- Rotating 3D PCA projections showing cluster separation
- Sphere projections showing angular separation
- Layer evolution animations
- Centroid trajectory animations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# Beautiful color palette
COLORS = {
    "user": "#2ecc71",   # Green
    "self": "#3498db",   # Blue  
    "other": "#e74c3c",  # Red
    "neutral": "#95a5a6", # Gray
}


@dataclass
class VisualizationConfig:
    """Configuration for 3D visualizations."""
    figsize: Tuple[int, int] = (10, 8)
    dpi: int = 100
    fps: int = 30
    rotation_frames: int = 120  # 4 seconds at 30fps
    elev_angle: float = 20.0
    point_size: int = 50
    centroid_size: int = 200
    alpha: float = 0.6
    save_gif: bool = True


class Visualization3D:
    """
    Create beautiful 3D visualizations of activation geometry.
    
    Example:
        viz = Visualization3D(output_dir="figures/")
        
        # Create rotating 3D PCA projection
        viz.rotating_pca_projection(
            activations_by_class={"user": user_acts, "self": self_acts, "other": other_acts},
            title="Layer 20 Entity Representations",
            save_name="layer_20_3d"
        )
        
        # Create layer evolution animation
        viz.layer_evolution(
            activations_by_layer_and_class=layer_data,
            save_name="layer_evolution"
        )
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        config: Optional[VisualizationConfig] = None
    ):
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or VisualizationConfig()
        
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not installed. Install with: pip install matplotlib")
    
    def _check_matplotlib(self):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required. pip install matplotlib")
    
    def rotating_pca_projection(
        self,
        activations_by_class: Dict[str, np.ndarray],
        title: str = "3D PCA Projection",
        save_name: Optional[str] = None
    ):
        """
        Create a rotating 3D PCA projection GIF.
        
        Shows how different classes (user/self/other) cluster in activation space.
        
        Args:
            activations_by_class: Dict mapping class name to (n_samples, hidden_dim) array
            title: Title for the plot
            save_name: Filename to save (without extension)
            
        Returns:
            Animation object
        """
        self._check_matplotlib()
        
        # Combine all activations for PCA
        all_acts = np.vstack(list(activations_by_class.values()))
        labels = []
        for class_name, acts in activations_by_class.items():
            labels.extend([class_name] * len(acts))
        
        # Fit PCA
        pca = PCA(n_components=3)
        projected = pca.fit_transform(all_acts)
        
        # Split back by class
        projected_by_class = {}
        idx = 0
        for class_name, acts in activations_by_class.items():
            projected_by_class[class_name] = projected[idx:idx + len(acts)]
            idx += len(acts)
        
        # Create figure
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each class
        scatters = []
        for class_name, points in projected_by_class.items():
            color = COLORS.get(class_name, COLORS["neutral"])
            scatter = ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=color, s=self.config.point_size, alpha=self.config.alpha,
                label=class_name.capitalize(), edgecolors='white', linewidths=0.5
            )
            scatters.append(scatter)
            
            # Add centroid
            centroid = points.mean(axis=0)
            ax.scatter(
                [centroid[0]], [centroid[1]], [centroid[2]],
                c=color, s=self.config.centroid_size, marker='*',
                edgecolors='black', linewidths=1
            )
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
        ax.set_title(title)
        ax.legend(loc='upper left')
        
        # Animation function
        def rotate(frame):
            ax.view_init(elev=self.config.elev_angle, azim=frame * 3)
            return scatters
        
        anim = animation.FuncAnimation(
            fig, rotate, frames=self.config.rotation_frames,
            interval=1000//self.config.fps, blit=False
        )
        
        if save_name and self.config.save_gif:
            path = self.output_dir / f"{save_name}.gif"
            anim.save(path, writer='pillow', fps=self.config.fps)
            print(f"Saved: {path}")
        
        plt.close(fig)
        return anim
    
    def sphere_projection(
        self,
        activations_by_class: Dict[str, np.ndarray],
        title: str = "Angular Separation on Unit Sphere",
        save_name: Optional[str] = None
    ):
        """
        Project class centroids onto unit sphere to show angular separation.
        
        Arrows from origin show centroid directions - angle between arrows = angular separation.
        
        Args:
            activations_by_class: Dict mapping class name to activations
            title: Title for the plot
            save_name: Filename to save
            
        Returns:
            Animation object
        """
        self._check_matplotlib()
        
        # Compute centroids
        centroids = {}
        for class_name, acts in activations_by_class.items():
            centroid = acts.mean(axis=0)
            # Normalize to unit sphere
            centroids[class_name] = centroid / np.linalg.norm(centroid)
        
        # PCA on centroids for 3D projection
        centroid_matrix = np.vstack(list(centroids.values()))
        pca = PCA(n_components=3)
        projected = pca.fit_transform(centroid_matrix)
        
        # Normalize projected centroids to unit sphere
        projected_centroids = {}
        for i, class_name in enumerate(centroids.keys()):
            vec = projected[i]
            projected_centroids[class_name] = vec / np.linalg.norm(vec)
        
        # Create figure
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw unit sphere wireframe
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, alpha=0.1, color='gray')
        
        # Draw arrows from origin to centroids
        arrows = []
        for class_name, vec in projected_centroids.items():
            color = COLORS.get(class_name, COLORS["neutral"])
            arrow = ax.quiver(
                0, 0, 0, vec[0], vec[1], vec[2],
                color=color, arrow_length_ratio=0.1, linewidth=3,
                label=class_name.capitalize()
            )
            arrows.append(arrow)
            
            # Add point at end
            ax.scatter([vec[0]], [vec[1]], [vec[2]], c=color, s=100, marker='o')
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(title)
        ax.legend(loc='upper left')
        
        # Animation function
        def rotate(frame):
            ax.view_init(elev=self.config.elev_angle, azim=frame * 3)
            return []
        
        anim = animation.FuncAnimation(
            fig, rotate, frames=self.config.rotation_frames,
            interval=1000//self.config.fps, blit=False
        )
        
        if save_name and self.config.save_gif:
            path = self.output_dir / f"{save_name}.gif"
            anim.save(path, writer='pillow', fps=self.config.fps)
            print(f"Saved: {path}")
        
        plt.close(fig)
        return anim
    
    def layer_evolution(
        self,
        activations_by_layer_and_class: Dict[int, Dict[str, np.ndarray]],
        title: str = "Representation Evolution Through Layers",
        save_name: Optional[str] = None
    ):
        """
        Animate how representations evolve through layers.
        
        Shows the U-shaped separation curve in action - classes similar at input,
        diverge in middle layers, partially reconverge at output.
        
        Args:
            activations_by_layer_and_class: Dict[layer -> Dict[class -> activations]]
            title: Title for the plot
            save_name: Filename to save
            
        Returns:
            Animation object
        """
        self._check_matplotlib()
        
        layers = sorted(activations_by_layer_and_class.keys())
        n_layers = len(layers)
        
        # Fit global PCA on all data
        all_acts = []
        for layer_data in activations_by_layer_and_class.values():
            for acts in layer_data.values():
                all_acts.append(acts)
        all_acts = np.vstack(all_acts)
        
        pca = PCA(n_components=3)
        pca.fit(all_acts)
        
        # Project each layer's data
        projected_by_layer = {}
        for layer, layer_data in activations_by_layer_and_class.items():
            projected_by_layer[layer] = {}
            for class_name, acts in layer_data.items():
                projected_by_layer[layer][class_name] = pca.transform(acts)
        
        # Create figure
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Animation function
        def animate(frame):
            ax.clear()
            
            layer_idx = frame % n_layers
            layer = layers[layer_idx]
            layer_data = projected_by_layer[layer]
            
            for class_name, points in layer_data.items():
                color = COLORS.get(class_name, COLORS["neutral"])
                ax.scatter(
                    points[:, 0], points[:, 1], points[:, 2],
                    c=color, s=self.config.point_size, alpha=self.config.alpha,
                    label=class_name.capitalize()
                )
                
                # Add centroid
                centroid = points.mean(axis=0)
                ax.scatter(
                    [centroid[0]], [centroid[1]], [centroid[2]],
                    c=color, s=self.config.centroid_size, marker='*',
                    edgecolors='black', linewidths=1
                )
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title(f"{title}\nLayer {layer}")
            ax.legend(loc='upper left')
            
            # Set consistent axis limits
            ax.set_xlim([all_acts[:, 0].min(), all_acts[:, 0].max()])
            ax.set_ylim([all_acts[:, 1].min(), all_acts[:, 1].max()])
            
            return []
        
        # Slower animation - 1 second per layer
        frames_per_layer = self.config.fps
        total_frames = n_layers * frames_per_layer
        
        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames,
            interval=1000//self.config.fps, blit=False
        )
        
        if save_name and self.config.save_gif:
            path = self.output_dir / f"{save_name}.gif"
            anim.save(path, writer='pillow', fps=self.config.fps)
            print(f"Saved: {path}")
        
        plt.close(fig)
        return anim
    
    def centroid_trajectories(
        self,
        activations_by_layer_and_class: Dict[int, Dict[str, np.ndarray]],
        title: str = "Centroid Trajectories Through Layers",
        save_name: Optional[str] = None
    ):
        """
        Show how class centroids move through layers.
        
        Lines connect same class across layers. Shows the divergence and
        reconvergence pattern.
        
        Args:
            activations_by_layer_and_class: Dict[layer -> Dict[class -> activations]]
            title: Title for the plot
            save_name: Filename to save
            
        Returns:
            Figure object
        """
        self._check_matplotlib()
        
        layers = sorted(activations_by_layer_and_class.keys())
        
        # Compute centroids per layer
        centroids_by_layer = {}
        for layer, layer_data in activations_by_layer_and_class.items():
            centroids_by_layer[layer] = {}
            for class_name, acts in layer_data.items():
                centroids_by_layer[layer][class_name] = acts.mean(axis=0)
        
        # Fit global PCA
        all_centroids = []
        for layer_data in centroids_by_layer.values():
            for centroid in layer_data.values():
                all_centroids.append(centroid)
        
        pca = PCA(n_components=3)
        pca.fit(np.vstack(all_centroids))
        
        # Project centroids
        projected_centroids = {}
        for layer, layer_data in centroids_by_layer.items():
            projected_centroids[layer] = {}
            for class_name, centroid in layer_data.items():
                projected_centroids[layer][class_name] = pca.transform(centroid.reshape(1, -1))[0]
        
        # Create figure
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        class_names = list(activations_by_layer_and_class[layers[0]].keys())
        
        for class_name in class_names:
            color = COLORS.get(class_name, COLORS["neutral"])
            
            # Get trajectory
            trajectory = np.array([projected_centroids[l][class_name] for l in layers])
            
            # Plot line
            ax.plot(
                trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color=color, linewidth=2, alpha=0.7
            )
            
            # Plot points with size increasing by layer
            for i, layer in enumerate(layers):
                size = 30 + i * 20  # Larger for later layers
                ax.scatter(
                    [trajectory[i, 0]], [trajectory[i, 1]], [trajectory[i, 2]],
                    c=color, s=size, alpha=0.8
                )
            
            # Label start and end
            ax.text(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                   f'{class_name} L0', fontsize=8)
            ax.text(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                   f'{class_name} L{layers[-1]}', fontsize=8)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_name:
            path = self.output_dir / f"{save_name}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        return fig

