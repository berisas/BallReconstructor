# Visualization & Comparison GUI

A comprehensive GUI for visualizing and comparing mesh refinement experiments with numerical metrics.

## Features

### 📊 Multiple Comparison Views

1. **Metrics Tab** - Side-by-side before/after training statistics
   - Vertex count, face count, polygon information
   - Training loss metrics
   - Reconstruction error statistics
   - Quality improvement percentage

2. **Training Curves Tab** - Learning progression over epochs
   - Multi-scale loss progression
   - Learning rate schedule
   - Convergence analysis

3. **Error Analysis Tab** - Statistical error distribution
   - Histogram of per-vertex reconstruction errors
   - Mean, median, std dev annotations
   - Percentile breakdowns (25th, 75th, 95th)

4. **Comparison Tab** - Detailed metrics table
   - 9 different error metrics
   - Mesh statistics side-by-side
   - Export-ready format

### 🎮 Interactive Controls

- **Select Mesh** - Choose any OBJ/PLY/STL file
- **Train Model** - Run training with progress tracking
- **Show 3D Comparison** - Launch interactive 3D viewer
- **Show Error Heatmap** - Color-coded error visualization on mesh surface
- **Export Results** - Save metrics to CSV

### 📝 Real-time Status

- Live training progress log
- Timestamp-stamped messages
- Experiment loading and tracking

## Usage

### Launch the GUI

```bash
python launch_gui.py
```

Or run from Python:

```python
from research.visualization_gui import run_comparison_gui
run_comparison_gui()
```

### Workflow

1. **Select a Mesh**
   - Click "Select Mesh" button
   - Choose any .obj, .ply, or .stl file
   - Displays filename when loaded

2. **Train the Model**
   - Click "Train Model" button
   - GUI shows training progress in real-time
   - Automatically computes metrics after training

3. **Compare Results**
   - View metrics in "Metrics" tab (before/after)
   - Analyze error distribution in "Error Analysis" tab
   - Review detailed breakdown in "Comparison" tab

4. **Visualize in 3D**
   - Click "Show 3D Comparison" to see meshes side-by-side
   - Use mouse to rotate, zoom, pan

5. **Error Heatmap**
   - Click "Show Error Heatmap" to see per-vertex errors
   - Red/Yellow/Green indicates high/medium/low error

6. **Export**
   - Click "Export Results" to save metrics as CSV

## Displayed Metrics

### Before Training Section

- Mesh type: Medium Resolution
- Vertex count
- Face count
- Total polygon count

### After Training Section

- Training loss (final)
- Best loss achieved
- Reconstruction error (mean)
- Reconstruction error (max)
- Quality improvement %

### Error Statistics

- **Mean Error** - Average deviation from ground truth
- **Median Error** - 50th percentile (less affected by outliers)
- **Std Dev** - Spread of errors
- **Min/Max Error** - Extreme values
- **25th/75th Percentile** - Interquartile range
- **95th Percentile** - Outlier detection

### Mesh Statistics

- Input vertices: Medium resolution base mesh
- Output vertices: ML Enhanced refined mesh
- Reference vertices: Ground truth high resolution

## Understanding the Results

### Good Results (Accurate Refinement)
- Mean Error < 0.005
- Max Error < 0.05 (relative to mesh size)
- Error distribution centered near zero
- ML Enhanced closely matches Original in visualization

### Poor Results (Inaccurate Refinement)
- Mean Error > 0.01
- Very large Max Error values
- Error distribution with high variance
- Visible deviations in 3D comparison

### Quality Indicators
- **Low % improvements** (0-5%) - Model not learning much
- **Medium % improvements** (5-15%) - Good baseline refinement
- **High % improvements** (>15%) - Excellent face count reduction with minimal error

## Tips for Better Results

1. **Use more epochs** - Default 50 is for quick testing; try 150-200 for better accuracy
2. **Larger meshes** - More training signal with higher vertex counts
3. **High-quality ground truth** - Original mesh should be well-formed
4. **Consistent topology** - Ensure LOD meshes maintain face structure

## Technical Details

- **matplotlib** - Plots and visualizations
- **tkinter** - GUI framework
- **trimesh** - 3D mesh handling
- **numpy** - Numerical computations

## Examples

### Comparing Multiple Experiments

```python
from research.visualization_gui import MeshComparisonGUI
import tkinter as tk

root = tk.Tk()
gui = MeshComparisonGUI(root)

# Load different experiments to compare
gui._load_experiment()

root.mainloop()
```

### Batch Processing

```python
from research.visualization_gui import MeshComparisonGUI
from research.benchmark_suite import BenchmarkSuite

# Run benchmark and collect results
suite = BenchmarkSuite()
results = suite.run_benchmark_suite(mesh_files, configs, max_experiments=10)

# Then view in GUI
gui.current_experiment_data = results
```

---

**Created**: April 2026  
**Purpose**: Research visualization and mesh refinement comparison
