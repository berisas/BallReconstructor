# BallReconstructor - Neural Mesh Super-Resolution

A deep learning system that enhances low-poly 3D meshes to high-poly quality using a custom TensorFlow neural network. BallReconstructor uses multi-scale progressive refinement to learn geometric details from simplified meshes and reconstruct high-resolution vertices with impressive accuracy.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Multi-Scale Progressive Training**: Cascading refinement through 3 resolution levels (Ultra-Low → Low → Medium → High)
- **Curvature-Adaptive Loss Weighting**: Intelligent weighting of feature-rich vs smooth regions during training
- **Learnable Scaling Parameters**: Automatic adaptation of feature and displacement scales (trainable parameters)
- **Research GUI Interface**: Interactive training interface with configurable epochs and real-time progress feedback
- **Advanced LOD System**: Automatic multi-level mesh hierarchy generation (5 levels)
- **Laplacian Regularization**: Prevents over-fitting on smooth surfaces while preserving geometric detail
- **Comprehensive Metrics**: Detailed reconstruction error analysis and training statistics
- **Gradient Clipping**: Stable training across mesh complexity variations (L2 norm ≤ 1.0)
- **Flexible Input**: Works with any OBJ format mesh

## Installation

1. Clone the repository:
```bash
git clone https://github.com/berisas/BallReconstructor.git
cd BallReconstructor
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### GUI Mode (Recommended)

```bash
python BallReconstructor.py
```

The GUI provides:
- **Mesh Information Panel**: LOD hierarchy statistics and mesh details
- **Training Configuration**: Configurable epochs (50-1000)
- **Training Progress**: Real-time loss and metric monitoring
- **Visualization Tools**: 
  - "Preview Mesh" - View the LOD hierarchy before training
  - "View ML Enhanced Mesh" - Inspect the network output after training
- **Quality Metrics**: Mean/max reconstruction error and quality improvement

### Console Mode

If tkinter is unavailable, the application runs in console mode automatically with 150 default epochs.

### Programmatic Usage

```python
from BallReconstructor import MLLODSystem

# Load mesh
lod_system = MLLODSystem("path/to/your/mesh.obj")

# Train the model
lod_system.train_ml_model(epochs=300)

# Visualize results
lod_system.show_visual_results()
```

## Architecture

### Neural Network Model: MeshSuperResNet

Advanced encoder-decoder architecture with learnable scaling parameters:

```
Input: Low-resolution vertices (normalized)
       ↓
[Local Encoder] - 3-layer dense network
  • Dense(512) + LayerNorm + Dropout(0.1)
  • Dense(512) + LayerNorm + Dropout(0.1)
  • Dense(512) + LayerNorm
  • Output scaled by learnable feature_scale
       ↓
[Global Context] - Mesh-wide statistics
  • Max pooling over local features
  • Mean pooling over local features
  • Std pooling over local features
       ↓
[Global Encoder] - 2-layer dense network
  • Dense(512) + LayerNorm
  • Dense(256)
       ↓
[Feature Interpolation] - K-NN weighted blending
  • Maps low-res features to query positions
  • Gaussian-weighted k-NN (k=8 or k=16)
       ↓
[Position Decoder] - 5-layer dense network
  • Concatenate: [query_pos, interp_features, global_context]
  • Dense(1024) + LayerNorm + Dropout(0.1)
  • Dense(512) + LayerNorm + Dropout(0.1)
  • Dense(512) + LayerNorm + Dropout(0.1)
  • Dense(256)
  • Dense(3) + Tanh activation
  • Output scaled by learnable displacement_scale
       ↓
Output: High-resolution vertices (predicted displacement + template)
```

**Key Parameters:**
- Hidden dimension: 512
- Dropout rate: 0.1
- Feature scale: Trainable (initialized to 1.0)
- Displacement scale: Trainable (initialized to 0.05)

### Multi-Scale Progressive Training

The system trains through **3 cascading scales** with progressive refinement:

| Scale | Input | Output | K-NN | Weight | Purpose |
|-------|-------|--------|------|--------|---------|
| 1 | Ultra-Low (5% faces) | Low (12.5% faces) | 8 | 1.0 | Coarse structure learning |
| 2 | Low (12.5% faces) | Medium (25% faces) | 8 | 1.0 | Intermediate refinement |
| 3 | Medium (25% faces) | High (100% faces) | 16 | 2.0 | Fine detail refinement (weighted 2x) |

**Scale 3 Special Features:**
- Increased k-NN neighbors (16 instead of 8) for better context
- Feature-aware loss weighting based on curvature:
  - High-curvature regions (features): 50% loss weight
  - Low-curvature regions (smooth): 100% loss weight
- Laplacian smoothness regularization (weight: 0.01)
- Per-vertex outlier clamping at 95th percentile loss

### Helper Functions

1. **compute_mesh_curvature()** - Per-vertex curvature from edge angle variance
2. **compute_curvature_adaptive_weights()** - Distance + curvature-based interpolation weights
3. **compute_laplacian_regularization()** - Graph-based smoothness constraint
4. **compute_feature_aware_loss_weights()** - Per-vertex loss weighting
5. **compute_enhanced_interpolation_weights()** - K-NN Gaussian weighting
6. **create_enhanced_template()** - Template generation with controlled noise

### LOD System

Automatic mesh hierarchy with 5 levels:

| Level | Faces | Reduction | Color | Purpose |
|-------|-------|-----------|-------|---------|
| Ultra Low | 5% of original | 95% | Red | Training seed |
| Low | 12.5% of original | 87.5% | Orange | Intermediate step |
| Medium Base | 25% of original | 75% | Yellow | Primary ML input |
| Original | 100% | 0% | Blue | Ground truth / target |
| ML Enhanced | 100% | 0% | Cyan | Network output |

## Training Details

### Optimizer Configuration
- **Type**: Adam with exponential learning rate decay
- **Initial Learning Rate**: 0.001
- **Decay Rate**: 0.95
- **Decay Steps**: Every 100 epochs

### Loss Functions (Scale 3 only)
- **Position Loss**: MSE between predicted and target vertices
- **Smoothness Loss**: Laplacian regularization (weight: 0.01)
- **Combined**: Feature-weighted position loss + smoothness regularization

### Data Normalization
- All scales normalized to single coordinate space
- Center: Mean of all vertices across all scales
- Scale: Max distance from center

### Template Generation
- Enhanced templates reduce initial noise
- Gaussian noise initialization (2% of vertex std)
- Improves convergence and final accuracy

### Gradient Management
- Gradient clipping: L2 norm ≤ 1.0
- Prevents training instability across mesh complexities
- Per-step normalization

## Visualization & Analysis

### Before Training
Shows the complete LOD hierarchy in a grid:
- Ultra Low, Low meshes (top row)
- Medium Base, Original mesh (bottom row)
- Useful for understanding data preparation

### After Training
Shows before/after refinement comparisons:
- Row 0: Medium Base (untrained) vs Original (reference)
- Row 1: Medium Base vs ML Enhanced (network output)
- Side-by-side comparison reveals learned refinement

### Quality Metrics
After training, displays:
- **Mean Reconstruction Error**: Average vertex displacement error
- **Max Reconstruction Error**: Largest single vertex error
- **Final Training Loss**: Loss value at final epoch
- **Best Training Loss**: Minimum loss achieved during training
- **Face Improvement**: Face count progression (medium → original)
- **Quality Gain**: Percentage improvement

## Project Structure

```
BallReconstructor/
├── BallReconstructor.py       # Main application (LOD system + ML pipeline + GUI)
├── README.md                  # This file
├── CLAUDE.md                  # Technical documentation for AI assistants
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT license
├── tennis_ball.obj            # Sample mesh for testing
├── Assets/                    # Unity project assets
├── ProjectSettings/           # Unity project settings
├── Packages/                  # Unity packages
└── .git/                      # Git repository
```

## Requirements

```
Python 3.8+
TensorFlow 2.13+
NumPy 1.24+
Trimesh 4.0+
SciPy 1.11+
Matplotlib 3.7+ (for visualization)
Tkinter (optional, for GUI mode - usually bundled with Python)
```

Install all dependencies:
```bash
pip install tensorflow numpy trimesh scipy matplotlib
```

## Known Limitations

1. **Fixed Topology**: Same face connectivity required for input and output meshes
2. **Single Mesh Training**: Model trains on one mesh per session (not generalized across objects)
3. **Memory Usage**: Large meshes (>50k vertices) benefit from GPU acceleration
4. **No Model Checkpoint**: Model retrains from scratch each run
5. **Absolute Paths**: Default mesh path is system-specific

## Expected Performance

On the tennis ball mesh with 150 epochs:
- **Mean Reconstruction Error**: < 0.01 (excellent quality)
- **Max Reconstruction Error**: < 0.05 (acceptable)
- **Training Loss**: Converges below 0.001
- **Training Time**: ~2-5 minutes on CPU, <1 minute on GPU

## Future Improvements

- [ ] Model checkpoint saving/loading for checkpoint-based training
- [ ] Multi-mesh training for generalized models across different objects
- [ ] GPU acceleration verification and optimization
- [ ] Texture and normal map enhancement support
- [ ] Batch processing for multiple meshes
- [ ] Real-time preview during training
- [ ] Direct Unity integration via native plugin
- [ ] Web interface for remote training
- [ ] Automated hyperparameter tuning

## Troubleshooting

**Issue**: "Invalid mesh file - no vertices found"
- Ensure OBJ file is valid and properly formatted
- Check that file path is correct

**Issue**: Training loss not decreasing
- Increase number of epochs (try 300-500)
- Verify mesh has sufficient complexity (minimum 300 faces recommended)
- Check that all model variables are trainable

**Issue**: High reconstruction error
- Train for more epochs
- Verify mesh quality and topology consistency
- Consider increasing hidden dimension

**Issue**: GUI not launching
- Tkinter may not be installed
- Application falls back to console mode automatically
- Install tkinter: `pip install tk`

**Issue**: Out of memory
- Use GPU instead of CPU
- Try on smaller mesh (simplify with higher reduction ratio)
- Reduce batch processing

## Contributing

Contributions are welcome! Please feel free to:
- Report issues and bugs
- Suggest improvements
- Submit pull requests with enhancements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use BallReconstructor in your research, please consider citing this project.

## Acknowledgments

- Built with [TensorFlow](https://www.tensorflow.org/) and [Trimesh](https://trimesh.org/)
- Inspired by mesh super-resolution research in computer graphics
- Tennis ball model used for testing and demonstration

---

**Made with machine learning and passion for 3D graphics**
