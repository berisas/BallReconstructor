"""
BallReconstructor: Neural Mesh Super-Resolution System
Multi-scale progressive refinement for high-poly mesh generation from low-poly inputs.
"""
import os
import sys

try:
    import tkinter as tk
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

from training import MLLODSystem
from gui import run_gui


def run_console_mode(file_path=None, experiment_name=None, config_path=None):
    """
    Run in console mode for headless environments.
    
    Args:
        file_path: Path to mesh file
        experiment_name: Optional name for experiment tracking
        config_path: Optional path to config file
    """
    if file_path is None:
        file_path = "tennis_ball.obj"
    
    if not os.path.exists(file_path):
        print(f"\n❌ Error: Mesh file not found: {file_path}")
        print("\nUsage:")
        print("  python BallReconstructor.py <path_to_mesh_file> [--experiment NAME] [--config PATH]")
        print("\nExample:")
        print("  python BallReconstructor.py C:\\path\\to\\mesh.obj --experiment exp1 --config configs/custom.yaml")
        print("\nSupported formats: .obj, .ply, .stl (via trimesh)")
        return
    
    lod_system = MLLODSystem(file_path)
    
    if not lod_system.mesh_variants:
        print("Failed to load mesh variants")
        return
    
    success = lod_system.train_ml_model(
        epochs=150,
        experiment_name=experiment_name,
        config_path=config_path
    )
    if success:
        lod_system.preview_mesh()
    else:
        print("Training failed")


def main():
    """Main entry point - choose GUI or console mode."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='BallReconstructor: Neural Mesh Super-Resolution')
    parser.add_argument('mesh_file', nargs='?', default=None, help='Path to mesh file')
    parser.add_argument('--experiment', help='Experiment name for tracking')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--no-gui', action='store_true', help='Force console mode')
    
    args = parser.parse_args()
    
    if GUI_AVAILABLE and args.mesh_file is None and not args.no_gui:
        # GUI mode (no file specified, GUI will prompt)
        run_gui()
    else:
        # Console mode (file specified or GUI not available)
        run_console_mode(args.mesh_file, args.experiment, args.config)


if __name__ == "__main__":
    main()

