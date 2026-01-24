# GhostObjects - Neural Mesh Super-Resolution

Reconstruction of 3D objects using ML to mimic large poly-mesh objects with a low poly-mesh object.

A deep learning system that enhances low-poly 3D meshes to high-poly quality using a custom TensorFlow neural network. Ghost2Real learns geometric details from simplified meshes and reconstructs high-resolution vertices with impressive accuracy.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Unity](https://img.shields.io/badge/Unity-2021+-purple.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Multi-Scale Progressive Training**: Cascading refinement through 3 resolution levels
- **Curvature-Adaptive Learning**: Intelligent weighting of features vs smooth regions
- **Learnable Scaling Parameters**: Automatic adaptation of feature and displacement scales
- **Research GUI Interface**: Interactive training with configurable epochs and real-time feedback
- **Laplacian Regularization**: Prevents over-fitting on smooth surfaces
- **Advanced LOD System**: Automatic multi-level mesh hierarchy generation
- **Comprehensive Metrics**: Detailed reconstruction error analysis and training statistics
- **Gradient Clipping**: Stable training across mesh complexity variations
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

## Usage

### Quick Start (Console Mode)

```bash
python BallReconstructor.py
```

The application will:
1. Load the tennis ball mesh (or custom mesh if specified)
2. Generate multi-scale LOD variants automatically
3. Train the ML model progressively (3 scales, 150 epochs default)
4. Display side-by-side comparison of refinement results
5. Output comprehensive quality metrics

### GUI Mode (Recommended for Research)

If tkinter is available, the application launches a research-focused interface:

1. **Mesh Information Panel**: View LOD hierarchy and mesh statistics
2. **Training Configuration**: Adjust epochs (50-1000) before training
3. **Training Progress**: Monitor real-time progress and metrics
4. **Results Visualization**: 
   - "View ML Enhanced Mesh" to inspect the network output
   - "Preview Mesh" to see the LOD hierarchy
5. **Quality Metrics**: Mean error, max error, and quality improvement displayed

### Custom Mesh

```python
from BallReconstructor import MLLODSystem

# Load your own mesh
lod_system = MLLODSystem("path/to/your/mesh.obj")

# Train with custom settings
lod_system.train_ml_model(epochs=300)

# Visualize
lod_system.show_visual_results()
```

## How It Works

### 1. Multi-Scale Progressive Refinement
The system trains through **three cascading scales**, progressively refining from coarse to fine:

- **Scale 1**: Ultra-Low (5%) → Low (12.5%)
- **Scale 2**: Low (12.5%) → Medium (25%)  
- **Scale 3**: Medium (25%) → High (100%) - Primary training with 2x weight

### 2. Intelligent Feature Learning
- **Local Encoder**: Extracts geometric features from low-poly vertices
- **Global Encoder**: Captures mesh-wide shape context (max/mean/std statistics)
- **Position Decoder**: Predicts per-vertex displacement refinements using both local and global information

### 3. Advanced Training Techniques
- **Curvature-Adaptive Loss Weighting**: High-curvature features (details) penalized less than smooth regions
- **Laplacian Regularization**: Prevents over-smoothing while preserving geometric detail
- **Feature-Aware Loss**: Differentiates between feature and smooth surface regions
- **Gradient Clipping**: Ensures stable training across varied mesh complexities
- **Progressive Scale Weighting**: Final highest-resolution scale emphasized during training

### 4. Visualization
Displays mesh comparison side-by-side (before/after training):
- **Before Training**: LOD hierarchy showing progressive detail reduction
- **After Training**: Before/after refinement comparisons at each scale

## Model Architecture

**MeshSuperResNet** uses a sophisticated encoder-decoder design:

```
Low-Res Vertices (input)
         ↓
    [Local Encoder]
    Dense(512) → LayerNorm → Dropout
    Dense(512) → LayerNorm → Dropout
    Dense(512) → LayerNorm
         ↓ (scaled by learnable feature_scale)
    [Global Context]
    max/mean/std → [Global Encoder]
    Dense(512) → LayerNorm
    Dense(256)
         ↓
    [Feature Interpolation]
    K-NN weights applied to local features
         ↓
    [Position Decoder]
    Concatenate [position, interpolated_features, global_context]
    Dense(1024) → LayerNorm → Dropout
    Dense(512) → LayerNorm → Dropout
    Dense(512) → LayerNorm → Dropout
    Dense(256)
    Dense(3) + Tanh
         ↓ (scaled by learnable displacement_scale)
    Predicted vertex displacement
         ↓
    High-Res Vertices (output)
```

**Training Hyperparameters:**
- Optimizer: Adam (LR=0.001, exponential decay 0.95 every 100 steps)
- Loss: Feature-weighted MSE + Laplacian smoothness (0.01)
- Gradient clipping: L2 norm ≤ 1.0
- K-NN neighbors: 8 (normal scales), 16 (final scale)
- Batch size: 1 (full mesh)

## Project Structure

```
BallReconstructor/
├── BallReconstructor.py        # Main application (LOD system + ML pipeline)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── CLAUDE.md                   # AI assistant guidance
├── LICENSE                     # MIT license
└── tennis_ball.obj             # Sample mesh for testing
```
├── BallReconstructor.py  # ML mesh super-resolution system
├── tennis_ball.obj        # Test mesh
├── requirements.txt       # Python dependencies
├── CLAUDE.md             # Detailed technical documentation
├── README.md             # This file
├── Assets/               # Unity project assets
├── ProjectSettings/      # Unity project settings
└── Packages/             # Unity packages
```

## Results

Expected performance on tennis ball mesh:
- **Mean Reconstruction Error**: < 0.01
- **Training Loss**: Converges below 0.001
- **Visual Quality**: Green mesh nearly identical to Blue

## Technical Details

### Model Architecture
- Hidden dimension: 512
- Encoder: 3-layer dense network with LayerNormalization
- Decoder: 5-layer network with residual connections
- Learnable scaling parameters for adaptive refinement

### Loss Functions
1. Position Loss: MSE between predicted and target
2. Smoothness Loss: Regularizes displacement magnitude
3. Consistency Loss: Enforces neighbor coherence
4. Scale Loss: Prevents extreme scaling

### Training Configuration
- Optimizer: Adam with exponential learning rate decay
- Initial LR: 0.001, decay: 0.95 every 100 steps
- Gradient clipping: L2 norm max 1.0
- Early stopping: Patience 100 epochs

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- NumPy 1.24+
- Trimesh 4.0+
- SciPy 1.11+
- Matplotlib 3.7+ (for GUI)
- Tkinter (optional, for GUI mode)
- Unity 2021+ (for Unity development)

## Limitations

- Fixed topology: Same face connectivity required for input/output
- Single mesh training: Model doesn't generalize across different objects
- Memory intensive: Large meshes (>50k vertices) benefit from GPU
- No checkpoint saving: Model retrains each session

## Future Improvements

- [ ] Multi-mesh training for generalized models
- [ ] Model checkpoint saving/loading
- [ ] GPU acceleration with CUDA
- [ ] Support for texture and normal map enhancement
- [ ] Batch processing for multiple meshes
- [ ] Real-time preview during training
- [ ] Direct Unity integration via plugin

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with TensorFlow and Trimesh libraries
- Inspired by mesh super-resolution research in computer graphics
- Tennis ball model used for testing and demonstration
- Unity integration for game development workflow

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with machine learning, Unity, and passion for 3D graphics**
