# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Ghost2Real** is a neural mesh super-resolution system that uses deep learning to enhance low-poly 3D meshes to high-poly quality. The project implements a custom TensorFlow neural network trained to learn geometric details and reconstruct high-resolution mesh vertices through multi-scale progressive refinement.

This project is part of the **GhostObjects** repository, which combines Unity 3D development with machine learning for mesh enhancement.

**Core Component**:
- `Ghost2Real2.py` - Production version with multi-scale LOD system, research-focused GUI interface, and complete ML pipeline

## Running the Application

### GUI Mode (preferred if tkinter available)
```bash
python BallReconstructor.py
```
Launches research-focused interface with configurable training parameters and visual comparisons.

### Console Mode
```bash
python BallReconstructor.py
```
Falls back to console mode if tkinter unavailable. Runs 150 epochs on default mesh and displays results.

### Programmatic Usage
```python
from BallReconstructor import MLLODSystem

# Initialize with custom mesh file
lod_system = MLLODSystem("path/to/your/mesh.obj")

# Train the model (customizable epochs)
lod_system.train_ml_model(epochs=200)

# Visualize results
lod_system.show_visual_results()
```

## Architecture

### Neural Network Model (`MeshSuperResNet`)

Advanced encoder-decoder with learnable scaling parameters:

1. **Local Encoder** (512-dim hidden)
   - 3-layer dense network with LayerNormalization and Dropout (0.1)
   - Extracts local geometric features from low-resolution vertices
   - Output scaled by learnable `feature_scale` parameter (trainable)

2. **Global Encoder** (256-dim output)
   - Processes global statistics: max, mean, and std of local features
   - Captures overall mesh shape context
   - Provides global refinement guidance

3. **Position Decoder** (5-layer network)
   - Combines: interpolated local features + global context + query positions
   - 512→512→512→256 dense layers with LayerNorm and Dropout
   - Final output: 3D displacement vectors (tanh activation)
   - Displacement scaled by learnable `displacement_scale` parameter (trainable)

### Multi-Scale Training Pipeline

The system uses **progressive multi-scale refinement** through 3 cascading scales:

#### Scale 1: Ultra-Low → Low
- Input: ~5% face count
- Output: ~12.5% face count
- Purpose: Learn coarse geometric structure

#### Scale 2: Low → Medium
- Input: ~12.5% face count
- Output: ~25% face count
- Purpose: Refine intermediate details

#### Scale 3: Medium → High (weighted 2x)
- Input: ~25% face count (Medium Base, training input)
- Output: 100% face count (Original, training target)
- **Special features:**
  - k-NN increased to 16 neighbors for better context
  - Feature-aware loss weighting (curvature-based)
  - Laplacian smoothness regularization (0.01 weight)
  - Per-vertex outlier clamping at 95th percentile loss

**Training Details:**
- Optimizer: Adam with exponential learning rate decay (LR=0.001, decay=0.95 every 100 steps)
- Loss: Feature-weighted MSE + smoothness regularization (final scale only)
- Gradient clipping: L2 norm ≤ 1.0 per step
- All scales normalized to single coordinate space
- Template generation: 2% Gaussian noise initialization

### Helper Functions & Algorithms

1. **compute_curvature_adaptive_weights()**
   - Mesh curvature computed from edge angle variance
   - High-curvature regions (features) get 50% loss weight
   - Low-curvature regions (smooth) get 100% loss weight
   - Distance-based weighting scaled by local curvature

2. **compute_mesh_curvature()**
   - Per-vertex curvature from local edge angle statistics
   - Used for adaptive loss weighting and visualization

3. **compute_laplacian_regularization()**
   - Graph-based smoothness constraint
   - Prevents over-fitting in smooth surface regions
   - Applied only to final highest-resolution training scale

### LOD System (`MLLODSystem`)

Automatic mesh hierarchy generation:

| Level | Reduction | Purpose |
|-------|-----------|---------|
| Ultra Low | 95% faces | Progressive training seed |
| Low | 87.5% faces | Intermediate training step |
| Medium Base | 75% faces | Primary ML input/training scale |
| High (Original) | 0% (100%) | Ground truth / training target |
| ML Enhanced | 0% (100%) | Network output with learned vertices |

## Data Format

**Input**: OBJ mesh file (any format supported by trimesh)
- Tested with: tennis_ball.obj and other parametric meshes
- Minimum recommended: 300 faces for feasible simplification
- Larger meshes (1000+) yield better training results

**Output**: Multi-mesh visualization scene
- Ultra Low: Red (95% reduction)
- Low: Orange (87.5% reduction)
- Medium Base: Yellow (75% reduction, ML input)
- ML Enhanced: Cyan (network output)
- Original: Blue (ground truth)

Before training: Shows LOD hierarchy
After training: Shows before/after refinement comparison

## Key Design Decisions

1. **Multi-Scale Progressive Refinement**: Cascading training from coarse to fine improves convergence and stability
2. **Curvature-Adaptive Weighting**: High-curvature features get penalized less (50% weight) to preserve geometric details
3. **Learnable Scaling**: Both feature and displacement scales are trainable parameters for automatic adaptation
4. **Laplacian Regularization**: Prevents over-fitting on smooth surfaces while allowing feature detail recovery
5. **Gradient Clipping**: Ensures stable training across all mesh complexities
6. **Research GUI**: Interactive epoch configuration and real-time progress feedback for experimentation

- **Interpolation Strategy**: k-NN with Gaussian weighting (k=8) provides smooth feature transfer
- **Reduction Ratio**: 30% reduction (keeping 70% of faces) balances training speed vs. quality
- **Hidden Dimension**: 512-dim provides sufficient capacity without overfitting
- **Learnable Parameters**: `displacement_scale` and `feature_scale` adapt to mesh characteristics
- **Gradient Clipping**: Norm clipping (max=1.0) prevents training instability
- **Learning Rate Schedule**: Exponential decay (0.95 every 100 steps) improves convergence

## Dependencies

Install via requirements.txt:
```bash
pip install -r requirements.txt
```

Required packages:
- `tensorflow` - Neural network framework
- `numpy` - Numerical computations
- `trimesh` - Mesh loading and visualization
- `scipy` - Spatial data structures (k-NN search)
- `matplotlib` - Plotting (optional for GUI)

Optional for GUI:
- `tkinter` - GUI framework (usually bundled with Python)

## Integration with Unity

The BallReconstructor project includes both Unity development and ML mesh enhancement:
- Unity project files are in the root directory
- ML scripts are in the same directory as `tennis_ball.obj`
- Enhanced meshes can be exported and imported into Unity scenes

## Known Limitations

1. **Fixed Topology**: Model requires same face connectivity for input/output
2. **Single Mesh Training**: Model trains on one mesh at a time (not generalized)
3. **Memory Usage**: Large meshes (>50k vertices) may require GPU
4. **Default Path**: Default mesh path is `C:\Users\Ber\BallReconstructor\tennis_ball.obj`
5. **No Model Saving**: Model is retrained each run (no checkpoint persistence)

## Performance Metrics

Expected results on tennis ball mesh:
- **Reconstruction Error (Mean)**: < 0.01 for good quality
- **Reconstruction Error (Max)**: < 0.05 acceptable
- **Training Loss**: Should converge below 0.001
- **Visual Quality**: Green mesh should closely match Blue mesh

## Troubleshooting

**Issue**: "Invalid mesh file - no vertices found"
- **Solution**: Ensure OBJ file is valid, check file path

**Issue**: Training loss not decreasing
- **Solution**: Increase epochs, adjust learning rate, or try different reduction_ratio

**Issue**: High reconstruction error
- **Solution**: Train longer, increase hidden_dim, use less aggressive reduction

**Issue**: GUI not available
- **Solution**: Install tkinter or use console mode (works automatically)
