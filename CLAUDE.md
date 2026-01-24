# CLAUDE.md - Technical Documentation

This file provides detailed technical guidance for AI assistants working with BallReconstructor code.

## Project Overview

**BallReconstructor** is a production-grade neural mesh super-resolution system that uses deep learning to enhance low-poly 3D meshes to high-poly quality. The project implements a sophisticated TensorFlow neural network trained with multi-scale progressive refinement to learn geometric details and reconstruct high-resolution mesh vertices.

**Core Component**: [BallReconstructor.py](BallReconstructor.py) - Complete ML pipeline with LOD system and optional GUI interface.

## Running the Application

### GUI Mode (Recommended for interactive research)

```bash
python BallReconstructor.py
```

Automatically launches the research interface if tkinter is available. Features:
- Mesh information panel with LOD statistics
- Configurable epoch slider (50-1000)
- Real-time training progress monitoring
- Integrated visualization tools
- Quality metrics display

### Console Mode

Runs automatically if tkinter is unavailable:

```bash
python BallReconstructor.py
```

Runs with 150 default epochs and displays results in terminal.

### Programmatic Usage

```python
from BallReconstructor import MLLODSystem

# Initialize with custom mesh
lod_system = MLLODSystem("path/to/mesh.obj")

# Train the model with custom settings
lod_system.train_ml_model(epochs=200)

# Visualize results
lod_system.show_visual_results()

# Access trained model
model = lod_system.trained_model
metrics = lod_system.quality_metrics
```

## Architecture Deep Dive

### Neural Network: MeshSuperResNet

The `MeshSuperResNet` class inherits from `tf.keras.Model` and implements a sophisticated encoder-decoder architecture:

**Constructor Parameters:**
- `hidden_dim` (int, default=512): Hidden dimension for all layers

**Trainable Variables:**
- `feature_scale` (tf.Variable): Scales local encoder output (init=1.0, trainable)
- `displacement_scale` (tf.Variable): Scales displacement prediction (init=0.05, trainable)

**Forward Pass Computation:**

```
Call signature: call(low_vertices, query_positions, low_to_query_weights, training=None)

Inputs:
  - low_vertices: [batch=1, num_low, 3] - Low-res vertex coordinates
  - query_positions: [batch=1, num_query, 3] - High-res vertex positions (template)
  - low_to_query_weights: [batch=1, num_query, num_low] - K-NN interpolation weights
  - training: Bool - Whether in training mode (affects Dropout)

Process:
  1. Local Encoder: Extract 512-dim features from low_vertices
  2. Scale features by trainable feature_scale
  3. Global Context: Compute max/mean/std statistics
  4. Global Encoder: Process statistics through 2-layer network → 256-dim
  5. Feature Interpolation: Map 512-dim features to query positions via weight matrix
  6. Position Decoder: Predict 3D displacement vectors
  7. Scale displacement by trainable displacement_scale
  8. Add displacement to query_positions (template) → refined_positions

Output:
  - refined_positions: [batch=1, num_query, 3] - High-res vertex predictions
```

**Layer Composition:**

- **Local Encoder** (512→512→512):
  - Dense(512) + LayerNorm + Dropout(0.1)
  - Dense(512) + LayerNorm + Dropout(0.1)
  - Dense(512) + LayerNorm

- **Global Encoder** (768→512→256):
  - Input: Concatenated [max, mean, std] of local features
  - Dense(512) + LayerNorm
  - Dense(256)

- **Position Decoder** (1024→1024→512→512→256→3):
  - Dense(1024) + LayerNorm + Dropout(0.1)
  - Dense(512) + LayerNorm + Dropout(0.1)
  - Dense(512) + LayerNorm + Dropout(0.1)
  - Dense(256)
  - Dense(3) + Tanh

### Multi-Scale Training Pipeline

**Class: MLLODSystem**

Manages mesh loading, LOD generation, training orchestration, and results visualization.

#### Initialization

```python
lod_system = MLLODSystem(mesh_path)
```

**Process:**
1. Loads mesh from OBJ file
2. Creates 4 mesh variants via quadric mesh decimation:
   - Ultra Low: ~5% of original faces
   - Low: ~12.5% of original faces
   - Medium Base: ~25% of original faces
   - High: 100% (original mesh)

#### Training Method

```python
success = lod_system.train_ml_model(epochs=100)
```

**Key Operations:**

1. **Data Preparation** (`_prepare_training_data()`):
   - Normalizes all meshes to single coordinate space
   - Center: Mean of all vertices
   - Scale: Max distance from center to vertex
   - Creates 3 training scales with curvature-adaptive weights
   - Generates enhanced templates with controlled noise

2. **Model Training** (`_train_model(scales, epochs)`):
   - Creates MeshSuperResNet instance
   - Sets up Adam optimizer with exponential LR decay
   - Iterates through epochs, training on all 3 scales per epoch
   - Applies feature-aware loss weighting (scale 3 only)
   - Uses Laplacian regularization (scale 3 only)
   - Clips gradients to L2 norm ≤ 1.0
   - Returns trained model and loss history

3. **Mesh Generation** (`_generate_ml_enhanced_mesh()`):
   - Uses trained model to refine medium resolution
   - Produces full-resolution mesh with learned refinements

4. **Metrics Calculation** (`_calculate_quality_metrics()`):
   - Computes mean and max reconstruction error
   - Records training loss statistics
   - Calculates quality improvement percentage

### Training Scales Detail

**Scale 1: Ultra-Low → Low**
- Input: Ultra-low mesh
- Output: Low mesh
- K-NN: 8 neighbors
- Loss weight: 1.0
- Purpose: Learn coarse geometric structure

**Scale 2: Low → Medium**
- Input: Low mesh
- Output: Medium base mesh
- K-NN: 8 neighbors
- Loss weight: 1.0
- Purpose: Refine intermediate geometric details

**Scale 3: Medium → High (Primary)**
- Input: Medium base mesh
- Output: High resolution (original) mesh
- K-NN: 16 neighbors (increased for context)
- Loss weight: 2.0 (emphasized for final quality)
- Special features:
  - Feature-aware loss weighting:
    - Curvature threshold: 0.4
    - High-curvature (features): 50% loss weight
    - Low-curvature (smooth): 100% loss weight
  - Laplacian smoothness regularization (weight: 0.01)
  - Per-vertex outlier clamping at 95th percentile

### Helper Functions

#### Curvature Computation
```python
curvatures = compute_mesh_curvature(vertices, faces)
```
- Computes per-vertex curvature from edge angle variance
- Returns normalized curvature values [0, 1]
- Used for adaptive weighting and analysis

#### Interpolation Weights
```python
weights = compute_enhanced_interpolation_weights(low_verts, high_verts, k=8)
```
- K-NN search using scipy.spatial.cKDTree
- Gaussian kernel weighting: exp(-dist²/(2σ²))
- σ = 0.5 × mean nearest-neighbor distance
- Returns sparse weight matrix [batch=1, num_high, num_low]

```python
weights = compute_curvature_adaptive_weights(low_verts, high_verts, 
                                             low_faces, high_faces, k=8)
```
- Combines distance weighting with curvature adaptation
- Curvature factor: 1.0 + high_curv × 2.0 (scale 1.0 to 3.0)
- Increases weight spreading in high-curvature regions

#### Laplacian Regularization
```python
laplacian = compute_laplacian_regularization(vertices, faces)
```
- Graph-based smoothness constraint
- Per-vertex: average_neighbor_positions - vertex_position
- Prevents over-fitting in smooth surface regions
- Applied as L2 regularization term (weight: 0.01)

#### Feature-Aware Loss Weighting
```python
loss_weights, curvatures = compute_feature_aware_loss_weights(vertices, faces, 
                                                              threshold=0.5)
```
- Per-vertex loss weighting based on curvature
- High-curvature (features): 50% weight (more lenient)
- Low-curvature (smooth): 100% weight (stricter)
- Preserves fine details while smoothing flat regions

#### Template Generation
```python
template = create_enhanced_template(low_vertices, high_vertices)
```
- Creates initial template with controlled noise
- Gaussian noise: μ=0, σ = 0.02 × std(vertices)
- Improves convergence by avoiding zero initialization

### LOD System Structure

| Property | Ultra Low | Low | Medium | Original | ML Enhanced |
|----------|-----------|-----|--------|----------|-------------|
| Faces | 5% | 12.5% | 25% | 100% | 100% |
| Role | Training seed | Step 1→2 | Step 2→3 input | Ground truth | Network output |
| Color | Red | Orange | Yellow | Blue | Cyan |

## Optimization Strategy

### Learning Rate Schedule
```python
schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100,
    decay_rate=0.95,
    staircase=True
)
optimizer = Adam(learning_rate=schedule)
```

- Starts at 0.001
- Decays 5% every 100 epochs
- Staircase: True (applies decay at exact steps)

### Gradient Management
- **Clipping**: Per-gradient L2 norm ≤ 1.0
- **Application**: After gradient computation, before optimizer step
- **Effect**: Prevents extreme weight updates on complex meshes

### Loss Composition (Scale 3 only)
```
total_loss = feature_weight * mse_loss + 0.01 * laplacian_term

where:
  mse_loss = mean((pred - target)² × per_vertex_weight)
  laplacian_term = mean((pred - laplacian)²)
  feature_weight = 0.5 (high-curv) or 1.0 (low-curv)
```

## Data Flow

```
OBJ File
  ↓
Load via trimesh
  ↓
Normalize coordinates
  ↓
Generate 4 mesh variants (quadric decimation)
  ↓
Extract vertex coordinates
  ↓
Create 3 training scales:
  ├─ Scale 1: Ultra-Low → Low
  ├─ Scale 2: Low → Medium
  └─ Scale 3: Medium → High (with curvature weights + Laplacian)
  ↓
For each epoch:
  ├─ Train on Scale 1
  ├─ Train on Scale 2
  └─ Train on Scale 3 (2x weight, with regularization)
  ↓
Generate ML Enhanced Mesh
  ├─ Use trained model
  ├─ Input: Medium resolution
  └─ Output: Refined high resolution
  ↓
Calculate Metrics
  ├─ Reconstruction error
  ├─ Training loss
  └─ Quality improvement
  ↓
Visualization
  ├─ Before training: LOD hierarchy
  └─ After training: Before/after comparison
```

## Key Design Decisions

1. **Progressive Multi-Scale Training**: Cascading from coarse to fine improves convergence and stability. Each scale learns incremental refinements.

2. **Curvature-Adaptive Weighting**: Features (high-curvature) are penalized less during training, allowing the network to learn fine details while smoothing flat regions.

3. **Learnable Scaling Parameters**: `feature_scale` and `displacement_scale` adapt to specific mesh characteristics during training.

4. **Laplacian Regularization**: Applied only to final scale to prevent over-fitting while allowing detail recovery in high-curvature regions.

5. **K-NN Interpolation**: Gaussian-weighted k-NN provides smooth feature transfer from low to high resolution. k=16 on final scale for better context.

6. **Gradient Clipping**: Ensures stable training across varied mesh complexities by preventing extreme weight updates.

7. **Template-Based Refinement**: Network predicts displacement from a template rather than absolute positions, simplifying learning task.

8. **Coordinate Space Normalization**: Single coordinate space across all scales prevents scale-related training artifacts.

## Hyperparameter Tuning

**Default Configuration (well-tested):**
- Hidden dimension: 512
- Dropout: 0.1
- Feature scale init: 1.0
- Displacement scale init: 0.05
- Learning rate: 0.001
- Decay rate: 0.95 every 100 epochs
- K-NN (scales 1-2): 8
- K-NN (scale 3): 16
- Scale weights: [1.0, 1.0, 2.0]
- Laplacian weight: 0.01

**Tuning Guide:**
- **Increase epochs** if training loss still decreasing at end
- **Increase hidden_dim** if convergence is too slow
- **Decrease initial_lr** if training is unstable
- **Increase k-NN** for larger/complex meshes
- **Adjust laplacian_weight** to control smoothness vs detail

## Performance Considerations

### Memory Usage
- Batch size: 1 (full mesh per forward pass)
- Dominant factor: Vertex count (quadratic in interpolation matrix size)
- ~500k vertices: ~4GB RAM on CPU, <1GB on GPU

### Training Time
- CPU: 2-5 minutes for 150 epochs
- GPU: <1 minute for 150 epochs
- Per-epoch time: 1-2 seconds on CPU, <0.5 seconds on GPU

### Quality Factors
- **Mesh complexity**: More vertices → better training signal
- **Topology consistency**: Must be preserved between scales
- **Epochs**: More epochs generally better (diminishing returns after 500)
- **Architecture**: 512-dim hidden provides good capacity/speed tradeoff

## Visualization Methods

### Before Training
```python
gui.preview_mesh()  # or lod_system.preview_mesh()
```
Shows 2×2 grid:
- Top-left: Ultra Low (red)
- Top-right: Low (orange)
- Bottom-left: Medium (yellow)
- Bottom-right: Original (blue)

### After Training
Shows 2-row comparison:
- Row 0: Medium (untrained) vs Original (reference)
- Row 1: Medium (trained) vs ML Enhanced (network output)

### Metrics Visualization
Printed to console:
- Reconstruction error (mean and max)
- Training loss progression
- Quality improvement percentage

## Extending the System

### Adding Custom Loss Functions
Modify `_train_model()` to compute additional loss terms and add to `loss`:

```python
custom_loss = custom_loss_function(pred, target)
loss = scale_weight * (combined_loss + 0.1 * custom_loss)
```

### Changing Network Architecture
Modify `MeshSuperResNet.__init__()` to adjust layer sizes or add layers:

```python
self.position_decoder = tf.keras.Sequential([
    # Add custom layers here
    tf.keras.layers.Dense(2048, activation='relu'),
    # ... more layers ...
])
```

### Implementing Model Checkpoints
Modify `_train_model()` to save best model:

```python
if avg_loss < best_loss:
    best_loss = avg_loss
    model.save_weights('best_model.h5')
```

## Known Issues and Workarounds

1. **Out of Memory**: Reduce mesh complexity or use GPU
2. **Slow Convergence**: Increase learning rate or reduce reduction ratio
3. **High Error**: More epochs needed or mesh too small
4. **GUI Errors**: tkinter not installed, falls back to console mode

## Dependencies and Versions

Tested combinations:
- TensorFlow 2.13+ with NumPy 1.24+
- Trimesh 4.0+ with SciPy 1.11+
- Matplotlib 3.7+ with Tkinter (bundled)

## Testing and Validation

**Default Test Mesh**: tennis_ball.obj
- Vertices: ~1500
- Faces: ~3000
- Expected mean error: < 0.01 with 150+ epochs
- Training time: ~2 minutes on CPU

**Custom Mesh Requirements**:
- OBJ format
- Minimum: 300 faces
- Recommended: 1000+ faces
- Consistent topology across LOD levels

---

**Last Updated**: January 2026
**Version**: 1.0
**Status**: Production Ready
