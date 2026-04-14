# BallReconstructor - Neural Mesh Super-Resolution

A deep learning system that enhances low-poly 3D meshes to high-poly quality using a custom TensorFlow neural network. BallReconstructor uses multi-scale progressive refinement to learn geometric details from simplified meshes and reconstruct high-resolution vertices.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Multi-Scale Progressive Training**: Cascading refinement through 3 resolution levels (Ultra-Low → Low → Medium → High)
- **Curvature-Adaptive Loss Weighting**: Intelligent weighting of feature-rich vs smooth regions during training
- **Learnable Scaling Parameters**: Automatic adaptation of feature and displacement scales (trainable parameters)
- **Research GUI Interface**: Interactive training interface with configurable epochs and real-time progress feedback
- **Advanced LOD System**: Automatic multi-level mesh hierarchy generation (4 levels)
- **Laplacian Regularization**: Prevents over-fitting on smooth surfaces while preserving geometric detail
- **Comprehensive Metrics**: Detailed reconstruction error analysis and training statistics
- **Gradient Clipping**: Stable training across mesh complexity variations (L2 norm ≤ 1.0)
- **Flexible Input**: Works with OBJ, PLY, STL formats and multi-part mesh scenes
- **Experiment Tracking**: Complete experiment logging with config, metrics, and checkpoints
- **Configuration Management**: YAML-based hyperparameter configuration
- **Reproducible Research**: Timestamped experiments with full metadata storage

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

### Console Mode (Headless)

```bash
python BallReconstructor.py path/to/mesh.obj
```

Pass a mesh file path as first argument to run in console mode. Automatically uses 150 default epochs.

### Command-Line Arguments

```bash
python BallReconstructor.py <mesh_file> [options]

Options:
  --experiment NAME    Track experiment with name (creates experiment logs)
  --config PATH        Use custom configuration file (YAML format)
  --no-gui            Force console mode even if GUI is available
```

**Examples:**

```bash
# Basic console mode
python BallReconstructor.py tennis_ball.obj

# With experiment tracking
python BallReconstructor.py my_mesh.obj --experiment exp_v1

# With custom config
python BallReconstructor.py my_mesh.obj --config configs/custom_config.yaml

# With experiment tracking and custom config
python BallReconstructor.py my_mesh.obj --experiment exp_v2 --config configs/exp2.yaml

# Force console mode
python BallReconstructor.py my_mesh.obj --no-gui
```

**Supported Mesh Formats:** `.obj`, `.ply`, `.stl` (via trimesh)

### Programmatic Usage

```python
from training import MLLODSystem

# Load mesh
lod_system = MLLODSystem("path/to/your/mesh.obj")

# Train with default settings
lod_system.train_ml_model(epochs=300)

# Train with experiment tracking
lod_system.train_ml_model(
    epochs=300, 
    experiment_name="my_experiment",
    config_path="configs/my_config.yaml"
)

# Visualize results
lod_system.preview_mesh()
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
- Displacement scale: Trainable (initialized to 0.03) - Conservative refinement

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

Automatic mesh hierarchy with 4 levels (plus ML-enhanced variant):

| Level | Faces | Reduction | Color | Purpose |
|-------|-------|-----------|-------|---------|
| Ultra Low | ~5% of original | ~95% | Red | Training seed |
| Low | ~12.5% of original | ~87.5% | Orange | Intermediate step |
| Medium Base | ~25% of original | ~75% | Yellow | Primary ML input |
| Original (High) | 100% | 0% | Blue | Ground truth / target |
| ML Enhanced | 100% | 0% | Cyan | Network output |

## Training Details

### Experiment Tracking & Reproducibility

When using the `--experiment` flag, BallReconstructor automatically creates a complete experiment record:

```bash
python BallReconstructor.py my_mesh.obj --experiment my_exp_v1
```

This creates a timestamped directory in `logs/my_exp_v1_YYYYMMDD_HHMMSS/` containing:

```
logs/my_exp_v1_20260414_103045/
├── config.yaml              # Full experiment configuration
├── metrics.csv              # Per-epoch metrics (loss, LR, etc.)
├── checkpoints/             # Model checkpoints
├── results/                 # Final results and analysis
└── visualizations/          # Generated visualizations
```

**Benefits:**
- Reproducible research with exact config logging
- Per-epoch metrics tracking in CSV format
- Model checkpoint saving for resuming training
- Organized experiment directory structure

### Configuration Files

Create custom configuration files (YAML format) to control all hyperparameters:

```yaml
# configs/custom_config.yaml
experiment:
  name: 'high_quality_refinement'
  method: 'progressive_multiscale'

dataset:
  mesh_file: 'my_mesh.obj'

training:
  epochs: 500
  batch_size: 1
  learning_rate: 0.001
  lr_decay_rate: 0.95
  lr_decay_steps: 100
  gradient_clip: 1.0

model:
  architecture: 'encoder_decoder'
  hidden_dim: 512
  dropout_rate: 0.1
  feature_scale_init: 1.0
  displacement_scale_init: 0.03
```

Use custom configs:

```bash
python BallReconstructor.py my_mesh.obj --config configs/custom_config.yaml
```

Or with experiment tracking:

```bash
python BallReconstructor.py my_mesh.obj \
  --experiment exp_v2 \
  --config configs/exp2.yaml
```

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
├── BallReconstructor.py       # Main entry point with CLI
├── gui.py                     # Interactive training GUI
├── training.py                # MLLODSystem and training pipeline
├── model.py                   # MeshSuperResNet neural network
├── mesh_utils.py              # Mesh processing utilities
├── README.md                  # This file
├── CLAUDE.md                  # Technical documentation for AI assistants
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT license
├── configs/                   # Configuration files (YAML)
│   └── default_config.yaml
├── logs/                      # Experiment logs (created at runtime)
├── research/                  # Research infrastructure
│   ├── experiment_tracker.py  # Experiment tracking and logging
│   ├── config_manager.py      # Configuration loading
│   ├── benchmark_suite.py     # Benchmarking utilities
│   └── evaluation/            # Evaluation metrics
├── Assets/                    # Unity project assets
├── ProjectSettings/           # Unity project settings
└── Packages/                  # Python packages
```

## Requirements

```
Python 3.8+
TensorFlow 2.13+
NumPy 1.24+
Trimesh 4.0+
SciPy 1.11+
Matplotlib 3.7+
PyYAML 6.0+ (for configuration files)
Tkinter (optional, for GUI - usually bundled with Python)
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

**Input File Formats**: OBJ, PLY, STL (via trimesh), plus multi-part mesh scenes

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

- [ ] Model checkpoint saving/loading for resuming training
- [ ] Multi-mesh training for generalized models across different objects
- [ ] GPU acceleration verification and optimization
- [ ] Texture and normal map enhancement support
- [ ] Batch processing for multiple meshes
- [ ] Real-time preview during training
- [ ] Direct Unity integration via native plugin
- [ ] Web interface for remote training
- [ ] Automated hyperparameter tuning via Bayesian optimization
- [ ] Model export to ONNX format for cross-platform inference

## Completed Improvements

- [x] Experiment tracking with config, metrics, and checkpoints
- [x] Configuration file support (YAML)
- [x] Multiple mesh format support (.obj, .ply, .stl)
- [x] Command-line interface
- [x] Scene/multi-part mesh handling

## Troubleshooting

**Issue**: "Invalid mesh file - no vertices found"
- Ensure OBJ/PLY/STL file is valid and properly formatted
- Check that file path is correct and readable

**Issue**: Command-line arguments not working
- Make sure to use correct syntax: `python BallReconstructor.py <mesh_file> [options]`
- Use `python BallReconstructor.py --help` to see all available options

**Issue**: Experiment tracking not creating logs
- Ensure `research` module is available (check imports in training.py)
- Verify write permissions in the working directory
- Check that `--experiment` flag is provided with experiment name

**Issue**: Config file not being loaded
- Verify YAML file format (check for syntax errors)
- Ensure file path is correct (absolute or relative to working directory)
- Check that required keys are present in config

**Issue**: Training loss not decreasing
- Increase number of epochs (try 300-500)
- Verify mesh has sufficient complexity (minimum 300 faces recommended)
- Check that all model variables are trainable
- Try reducing learning rate in config

**Issue**: High reconstruction error
- Train for more epochs
- Verify mesh quality and topology consistency
- Consider increasing hidden dimension in config
- Check mesh is properly normalized

**Issue**: GUI not launching
- Tkinter may not be installed
- Use `--no-gui` flag to force console mode
- Application falls back to console mode automatically if tkinter unavailable
- Install tkinter: `pip install tk`

**Issue**: Out of memory
- Use GPU instead of CPU (set TensorFlow to use GPU)
- Try on smaller mesh (simplify with higher reduction ratio in code)
- Reduce batch size or mesh complexity

**Issue**: Scene/multi-part mesh errors
- Ensure all parts are valid geometries with vertices and faces
- Check that mesh file is valid and readable
- Try converting to single mesh using external tool

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
