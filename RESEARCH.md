# Research & Development Notes

This document contains research insights, improvement history, and development notes for BallReconstructor.

## Accuracy Improvements

### Issues Identified and Fixed

#### 1. Template Noise During Inference
**Problem**: `create_enhanced_template()` adds Gaussian noise (σ=0.02×std) to improve training convergence, but the same noisy template was used during inference, causing the model to deviate from the clean original mesh.

**Solution**: Use clean template during inference instead of the noisy version used in training.
- **Location**: `_generate_ml_enhanced_mesh()` in `training.py`
- **Result**: Predictions stay closer to reference mesh, improving accuracy

#### 2. Weak Reference Constraint
**Problem**: Model could drift far from the original high-res mesh because there was no explicit penalty for large deviations during training.

**Solution**: Add reference constraint loss that penalizes deviation from the high-res mesh for the final training scale (medium→high).
- **Location**: `_train_scale()` method in `training.py`
- **Weight**: 10% of total loss
- **Result**: Network learns to refine while staying close to reference

#### 3. Excessive Displacement Scale
**Problem**: Initial `displacement_scale_init=0.05` allowed too much freedom in refinement, enabling large jumps from template positions.

**Solution**: Reduce to 0.03 with displacement threshold clamping at 90th percentile to cap outliers.
- **Location**: Default config in `configs/default_config.yaml`
- **Result**: More stable, predictable refinements

#### 4. Overly Lenient Curvature Weighting
**Problem**: High-curvature areas (features) weighted at 0.5 (50% penalty) meant the network learned features poorly to minimize loss.

**Solution**: Increase to 0.7 (70% penalty) - stricter while still respecting feature regions.
- **Logic**: High-curvature=0.7, Low-curvature=1.0
- **Location**: `compute_feature_aware_loss_weights()` in `mesh_utils.py`
- **Result**: Better feature preservation while maintaining smoothness

## Refactoring History

### Original Monolithic Structure
- Single `BallReconstructor.py` file with 1000+ lines
- Mixed concerns: model, training, mesh utilities, GUI all in one file
- Difficult to test, modify, or extend

### Refactored Modular Structure

#### **model.py** (~90 lines)
- `MeshSuperResNet` class - neural network definition
- Clean architecture with helper methods
- Easy to modify or experiment with different layer configurations

#### **mesh_utils.py** (~200 lines)
- 10 utility functions for mesh operations
- Functions grouped logically:
  - Mesh operations: `build_adjacency_list()`, `compute_mesh_curvature()`
  - Weight computation: `compute_interpolation_weights()` (consolidated from 2 functions)
  - Regularization: `compute_laplacian_regularization()`, `compute_feature_aware_loss_weights()`
  - Data processing: `normalize_mesh_vertices()`, `denormalize_vertices()`

#### **training.py** (~400 lines)
- `MLLODSystem` class - complete training orchestration
- Multi-scale progressive learning pipeline
- Mesh loading, LOD generation, training, metrics calculation
- Clean separation of concerns with private methods

#### **gui.py** (~150 lines)
- Research-focused GUI for interactive training
- Configuration parameters (epochs, learning rate, etc.)
- Real-time training progress monitoring
- Mesh visualization

#### **BallReconstructor.py** (~50 lines)
- Main entry point
- Argument parsing for command-line usage
- GUI or console mode selection

## Research Infrastructure

### Experiment Tracking (`research/experiment_tracker.py`)
- Automatic timestamped experiment directories
- Incremental CSV metric logging
- Config and result serialization
- Checkpoint saving

### Configuration System (`research/config_manager.py`)
- YAML-based experiment configuration
- ConfigLoader for loading and accessing configs
- ConfigBuilder for programmatic config creation
- Hyperparameter management and reproducibility

### Evaluation Metrics (`research/evaluation/metrics.py`)
- Chamfer distance (bidirectional point cloud distance)
- Hausdorff distance (max nearest-neighbor distance)
- Mesh Laplacian smoothness
- Per-vertex normal consistency
- Compression quality ratio
- MeshQualityEvaluator class for comprehensive analysis

### Method Abstraction (`research/methods/base_method.py`)
- Abstract `MeshRefinementMethod` base class
- `ProgressiveMultiScaleMethod` - current approach wrapper
- `DirectSingleStageMethod` - placeholder for future baseline
- `MethodFactory` - extensible method registry

### Visualization GUI (`research/visualization_gui.py`)
New comprehensive comparison tool featuring:
- **Metrics Tab**: Before/after training comparison
- **Training Curves Tab**: Loss progression visualization
- **Error Analysis Tab**: Per-vertex error distribution histogram
- **Comparison Tab**: Detailed metrics table
- **3D Comparison Viewer**: Side-by-side mesh comparison
- **Error Heatmap**: Color-coded error visualization (Red=high error, Green=low error)
- **CSV Export**: Save results for analysis

### Benchmarking (`research/benchmark_suite.py`)
- Multi-experiment orchestration
- Config variant generation with parameter grids
- Batch process multiple meshes and configurations
- Result aggregation and comparison reports

## Performance Characteristics

### Training Time
- **CPU**: 2-5 minutes for 150 epochs (~3 seconds per epoch)
- **GPU**: <1 minute for 150 epochs (<0.5 seconds per epoch)
- **Default**: 50 epochs for quick testing (~30 seconds)

### Memory Usage
- **CPU**: ~500MB for typical mesh (966 vertices)
- **GPU**: <1GB for same mesh with CUDA
- Dominant factor: quadratic in vertex count due to k-NN matrix

### Quality Metrics (Tennis Ball Example)
- **Best loss achieved**: 0.001945 (150 epochs)
- **Mean reconstruction error**: <0.005
- **Max reconstruction error**: <0.05
- **Quality improvement**: ~75% face reduction with minimal error

## Future Research Directions

1. **Graph Neural Network Methods**: Add GNN-based alternative for topology-aware refinement
2. **Adaptive Vertex Selection**: Implement curvature-based redundancy removal
3. **Multi-Mesh Transfer Learning**: Train on diverse mesh types for better generalization
4. **Ablation Studies**: Systematic analysis of loss components and architectural choices
5. **Large-Scale Benchmarking**: Compare against other super-resolution methods
