# Experiment 04: 3D Animated Visualizations

## Purpose

**Make the science visually intuitive and compelling.**

These animated 3D visualizations show the geometric structure of entity representations in a way that's immediately understandable.

---

## Visualizations Created

### 1. Rotating 3D PCA Projections

| File | Description | Key Insight |
|------|-------------|-------------|
| `rotating_3d_layer_0.gif` | Layer 0 (input) | Self and Other OVERLAP, User separate |
| `rotating_3d_layer_20.gif` | Layer 20 (peak) | ALL THREE clusters distinct |
| `rotating_3d_layer_35.gif` | Layer 35 (output) | Partial re-convergence |

**Colors**: 🟢 Green = User | 🔵 Blue = Self | 🔴 Red = Other

### 2. Sphere Projections

| File | Description | Key Insight |
|------|-------------|-------------|
| `sphere_projection_layer_0.gif` | Angular separation at layer 0 | Self/Other arrows nearly PARALLEL |
| `sphere_projection_layer_20.gif` | Angular separation at layer 20 | All arrows point DIFFERENT directions |

**How to read**: Arrows show entity centroid directions from origin. Angle between arrows = angular separation.

### 3. Layer Evolution Animation

| File | Description |
|------|-------------|
| `layer_evolution.gif` | Watch representations transform through all 10 layers |

Animation progresses through layers 0→4→8→12→16→20→24→28→32→35.

### 4. Centroid Trajectories

| File | Description |
|------|-------------|
| `centroid_trajectories.gif` | Shows how entity CENTROIDS move through layers |

Lines connect same entity across layers. Small points = early layers, Large points = late layers.

---

## Visual Evidence Summary

### Layer 0: Self ≈ Other
At input, the blue (Self) and red (Other) arrows point in almost the same direction.

### Layer 20: Maximum Separation
By layer 20, all three arrows point in distinct directions.

### The U-Shape in Motion
The `layer_evolution.gif` shows clusters separate then partially reconverge.

---

## How to Regenerate

```bash
python scripts/create_3d_visualizations.py
```

---

## Files

All visualizations in `figures/`:
- `rotating_3d_layer_0.gif`
- `rotating_3d_layer_20.gif`
- `rotating_3d_layer_35.gif`
- `sphere_projection_layer_0.gif`
- `sphere_projection_layer_20.gif`
- `layer_evolution.gif`
- `centroid_trajectories.gif`

