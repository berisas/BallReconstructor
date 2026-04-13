"""
Research visualization GUI for comparing mesh refinement results.
Displays metrics, training curves, error statistics, and mesh comparisons.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import json
import csv
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import trimesh

from training import MLLODSystem
from research.evaluation.metrics import MeshQualityEvaluator, reconstruction_error_stats


class MeshComparisonGUI:
    """GUI for visualizing and comparing mesh refinement experiments."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("BallReconstructor - Mesh Refinement Comparison Tool")
        self.root.geometry("1400x900")
        
        self.lod_system = None
        self.current_experiment_data = None
        self.experiments = []
        
        self._create_ui()
    
    def _create_ui(self):
        """Build the GUI interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel: Controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Right panel: Metrics and visualization
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self._create_left_panel(left_panel)
        self._create_right_panel(right_panel)
    
    def _create_left_panel(self, parent):
        """Create left control panel."""
        # Title
        title = ttk.Label(parent, text="Experiment Control", font=("Arial", 12, "bold"))
        title.pack(pady=10)
        
        # Load experiment section
        load_frame = ttk.LabelFrame(parent, text="Load Experiment", padding=10)
        load_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(load_frame, text="Load from Logs", 
                  command=self._load_experiment).pack(fill=tk.X, pady=5)
        
        self.experiment_dropdown = ttk.Combobox(load_frame, state="readonly", width=30)
        self.experiment_dropdown.pack(fill=tk.X, pady=5)
        self.experiment_dropdown.bind("<<ComboboxSelected>>", self._on_experiment_selected)
        
        # Run training section
        train_frame = ttk.LabelFrame(parent, text="Run Training", padding=10)
        train_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(train_frame, text="Select Mesh", 
                  command=self._select_mesh).pack(fill=tk.X, pady=5)
        
        self.mesh_label = ttk.Label(train_frame, text="No mesh selected", foreground="gray")
        self.mesh_label.pack(fill=tk.X, pady=5)
        
        self.train_button = ttk.Button(train_frame, text="Train Model", state=tk.DISABLED,
                                       command=self._train_model)
        self.train_button.pack(fill=tk.X, pady=5)
        
        # Comparison section
        comp_frame = ttk.LabelFrame(parent, text="Comparison", padding=10)
        comp_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(comp_frame, text="Show 3D Comparison", 
                  command=self._show_mesh_comparison).pack(fill=tk.X, pady=5)
        
        ttk.Button(comp_frame, text="Show Error Heatmap", 
                  command=self._show_error_heatmap).pack(fill=tk.X, pady=5)
        
        ttk.Button(comp_frame, text="Export Results", 
                  command=self._export_results).pack(fill=tk.X, pady=5)
        
        # Status
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.status_text = tk.Text(status_frame, height=15, width=40, state=tk.DISABLED)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        self.status_text.config(yscrollcommand=scrollbar.set)
    
    def _create_right_panel(self, parent):
        """Create right panel for metrics and visualization."""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Metrics tab
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text="Metrics")
        self._create_metrics_tab(metrics_frame)
        
        # Training curves tab
        curves_frame = ttk.Frame(notebook)
        notebook.add(curves_frame, text="Training Curves")
        self.curves_frame = curves_frame
        
        # Error distribution tab
        error_frame = ttk.Frame(notebook)
        notebook.add(error_frame, text="Error Analysis")
        self.error_frame = error_frame
        
        # Detailed comparison tab
        detail_frame = ttk.Frame(notebook)
        notebook.add(detail_frame, text="Comparison")
        self._create_comparison_tab(detail_frame)
    
    def _create_metrics_tab(self, parent):
        """Create metrics display tab."""
        # Title
        title = ttk.Label(parent, text="Model & Training Metrics", 
                         font=("Arial", 11, "bold"))
        title.pack(pady=10)
        
        # Create columns for before/after
        columns_frame = ttk.Frame(parent)
        columns_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Before column
        before_frame = ttk.LabelFrame(columns_frame, text="Before Training", padding=10)
        before_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.before_text = tk.Text(before_frame, height=25, width=30)
        self.before_text.pack(fill=tk.BOTH, expand=True)
        
        # After column
        after_frame = ttk.LabelFrame(columns_frame, text="After Training", padding=10)
        after_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.after_text = tk.Text(after_frame, height=25, width=30)
        self.after_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_comparison_tab(self, parent):
        """Create detailed comparison tab."""
        title = ttk.Label(parent, text="Quality Metrics Comparison", 
                         font=("Arial", 11, "bold"))
        title.pack(pady=10)
        
        # Create treeview for metrics table
        columns = ("Metric", "Value", "Unit")
        self.metrics_tree = ttk.Treeview(parent, columns=columns, height=20)
        self.metrics_tree.column("#0", width=0, stretch=tk.NO)
        self.metrics_tree.column("Metric", width=250, anchor=tk.W)
        self.metrics_tree.column("Value", width=150, anchor=tk.CENTER)
        self.metrics_tree.column("Unit", width=100, anchor=tk.W)
        
        self.metrics_tree.heading("#0", text="", anchor=tk.W)
        self.metrics_tree.heading("Metric", text="Metric", anchor=tk.W)
        self.metrics_tree.heading("Value", text="Value", anchor=tk.CENTER)
        self.metrics_tree.heading("Unit", text="Unit", anchor=tk.W)
        
        self.metrics_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscroll=scrollbar.set)
    
    def _log_status(self, message):
        """Log message to status text area."""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
        self.root.update()
    
    def _select_mesh(self):
        """Select a mesh file for training."""
        file_path = filedialog.askopenfilename(
            filetypes=[("OBJ files", "*.obj"), ("PLY files", "*.ply"), ("STL files", "*.stl")]
        )
        if file_path:
            self.current_mesh_path = file_path
            mesh_name = Path(file_path).name
            self.mesh_label.config(text=f"✓ {mesh_name}", foreground="green")
            self.train_button.config(state=tk.NORMAL)
            self._log_status(f"Mesh selected: {mesh_name}")
    
    def _train_model(self):
        """Train the model on selected mesh."""
        if not hasattr(self, 'current_mesh_path'):
            messagebox.showwarning("Error", "Please select a mesh first")
            return
        
        try:
            self._log_status("Initializing training...")
            self.lod_system = MLLODSystem(self.current_mesh_path)
            
            self._log_status("Loading and preparing meshes...")
            self._log_status(f"Ultra Low: {len(self.lod_system.mesh_variants['ultra_low'].vertices)} vertices")
            self._log_status(f"Low: {len(self.lod_system.mesh_variants['low'].vertices)} vertices")
            self._log_status(f"Medium: {len(self.lod_system.mesh_variants['medium_base'].vertices)} vertices")
            self._log_status(f"High: {len(self.lod_system.mesh_variants['high'].vertices)} vertices")
            
            self._log_status("Starting training (50 epochs)...")
            self.lod_system.train_ml_model(epochs=50)
            
            self._log_status("Training complete!")
            self._display_metrics()
            self._display_training_curves()
            self._display_error_analysis()
            
        except Exception as e:
            self._log_status(f"Error: {e}")
            messagebox.showerror("Training Error", str(e))
    
    def _display_metrics(self):
        """Display training metrics in before/after format."""
        if not self.lod_system:
            return
        
        # Before training (medium resolution)
        before_info = f"""
BEFORE TRAINING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mesh: Medium Resolution
Vertices: {len(self.lod_system.mesh_variants['medium_base'].vertices):,}
Faces: {len(self.lod_system.mesh_variants['medium_base'].faces):,}
Polygon Count: {len(self.lod_system.mesh_variants['medium_base'].faces) * 3:,}

Status: Untrained template
Optimization: None
Quality: Baseline (0%)
"""
        
        # After training (ML enhanced)
        after_info = f"""
AFTER TRAINING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mesh: ML Enhanced (50 epochs)
Vertices: {len(self.lod_system.ml_enhanced_mesh.vertices):,}
Faces: {len(self.lod_system.ml_enhanced_mesh.faces):,}
Polygon Count: {len(self.lod_system.ml_enhanced_mesh.faces) * 3:,}

Training Loss: {self.lod_system.quality_metrics.get('final_training_loss', 0):.6f}
Best Loss: {self.lod_system.quality_metrics.get('best_training_loss', 0):.6f}
Quality: Trained neural network

RECONSTRUCTION METRICS:
Mean Error: {self.lod_system.quality_metrics.get('reconstruction_error_mean', 0):.6f}
Max Error: {self.lod_system.quality_metrics.get('reconstruction_error_max', 0):.6f}
Quality Improvement: {self.lod_system.quality_metrics.get('quality_improvement', 0):.1f}%
"""
        
        self.before_text.config(state=tk.NORMAL)
        self.before_text.delete(1.0, tk.END)
        self.before_text.insert(1.0, before_info)
        self.before_text.config(state=tk.DISABLED)
        
        self.after_text.config(state=tk.NORMAL)
        self.after_text.delete(1.0, tk.END)
        self.after_text.insert(1.0, after_info)
        self.after_text.config(state=tk.DISABLED)
    
    def _display_training_curves(self):
        """Display training loss curves."""
        if not self.lod_system:
            return
        
        # Clear previous plot
        for widget in self.curves_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        # Placeholder - would need to save loss history
        ax.text(0.5, 0.5, "Training curves would display here\n(Loss history tracking needed)", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Multi-Scale Training Progression")
        
        canvas = FigureCanvasTkAgg(fig, master=self.curves_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _display_error_analysis(self):
        """Display error statistics."""
        if not self.lod_system or not self.lod_system.ml_enhanced_mesh:
            return
        
        # Clear previous plot
        for widget in self.error_frame.winfo_children():
            widget.destroy()
        
        # Compute error statistics
        orig_verts = self.lod_system.mesh_variants['high'].vertices
        enhanced_verts = self.lod_system.ml_enhanced_mesh.vertices
        
        # Align vertex count if needed
        min_verts = min(len(orig_verts), len(enhanced_verts))
        errors = np.linalg.norm(orig_verts[:min_verts] - enhanced_verts[:min_verts], axis=1)
        
        # Create histogram
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(errors):.6f}")
        ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f"Median: {np.median(errors):.6f}")
        
        ax.set_xlabel("Reconstruction Error (distance)")
        ax.set_ylabel("Number of Vertices")
        ax.set_title("Vertex-wise Error Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=self.error_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Update comparison tab with detailed metrics
        self._update_metrics_table(errors)
    
    def _update_metrics_table(self, errors):
        """Update metrics comparison table."""
        # Clear existing items
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        
        metrics_data = [
            ("Reconstruction Error Statistics", "", ""),
            ("Mean Error", f"{np.mean(errors):.6f}", "units"),
            ("Median Error", f"{np.median(errors):.6f}", "units"),
            ("Std Dev", f"{np.std(errors):.6f}", "units"),
            ("Min Error", f"{np.min(errors):.6f}", "units"),
            ("Max Error", f"{np.max(errors):.6f}", "units"),
            ("25th Percentile", f"{np.percentile(errors, 25):.6f}", "units"),
            ("75th Percentile", f"{np.percentile(errors, 75):.6f}", "units"),
            ("95th Percentile", f"{np.percentile(errors, 95):.6f}", "units"),
            ("", "", ""),
            ("Mesh Statistics", "", ""),
            ("Input Vertices", f"{len(self.lod_system.mesh_variants['medium_base'].vertices):,}", "count"),
            ("Output Vertices", f"{len(self.lod_system.ml_enhanced_mesh.vertices):,}", "count"),
            ("Reference Vertices", f"{len(self.lod_system.mesh_variants['high'].vertices):,}", "count"),
        ]
        
        for metric, value, unit in metrics_data:
            self.metrics_tree.insert("", tk.END, values=(metric, value, unit))
    
    def _show_mesh_comparison(self):
        """Display 3D mesh comparison."""
        if not self.lod_system or not self.lod_system.ml_enhanced_mesh:
            messagebox.showwarning("Error", "Please train a model first")
            return
        
        self._log_status("Launching 3D mesh comparison viewer...")
        self.lod_system.preview_mesh()
    
    def _show_error_heatmap(self):
        """Show error visualization (heatmap overlay on original mesh)."""
        if not self.lod_system or not self.lod_system.ml_enhanced_mesh:
            messagebox.showwarning("Error", "Please train a model first")
            return
        
        try:
            self._log_status("Computing error heatmap...")
            
            orig_mesh = self.lod_system.mesh_variants['high']
            enhanced_mesh = self.lod_system.ml_enhanced_mesh
            
            # Compute per-vertex errors (comparing enhanced to original)
            # Match vertex counts
            num_verts = len(enhanced_mesh.vertices)
            orig_verts = orig_mesh.vertices[:num_verts]
            enhanced_verts = enhanced_mesh.vertices[:num_verts]
            
            errors = np.linalg.norm(orig_verts - enhanced_verts, axis=1)
            
            # Create heatmap colors: Red (high error) to Green (low error)
            normalized_errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-6)
            error_colors = plt.cm.RdYlGn_r(normalized_errors)
            error_colors_uint8 = (error_colors[:, :4] * 255).astype(np.uint8)
            
            # Create a new mesh with vertex colors instead of textures
            display_mesh = trimesh.Trimesh(
                vertices=orig_verts,
                faces=orig_mesh.faces[:len(orig_mesh.faces)],  # Use all faces that fit
                process=False
            )
            
            # Set vertex colors directly
            display_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=display_mesh,
                vertex_colors=error_colors_uint8
            )
            
            # Create scene with just the colored mesh
            scene = trimesh.Scene([display_mesh])
            scene.show()
            
            self._log_status(f"Heatmap displayed | Error range: {errors.min():.6f} to {errors.max():.6f}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not create heatmap: {e}")
    
    def _load_experiment(self):
        """Load results from logs directory."""
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            messagebox.showwarning("Error", "No logs directory found")
            return
        
        experiments = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
        self.experiments = experiments
        self.experiment_dropdown['values'] = experiments
        self._log_status(f"Found {len(experiments)} experiments")
    
    def _on_experiment_selected(self, event):
        """Handle experiment selection."""
        selected = self.experiment_dropdown.get()
        if selected:
            self._log_status(f"Loading experiment: {selected}")
            # Load experiment data
            exp_path = os.path.join("logs", selected)
            results_file = os.path.join(exp_path, "results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    self.current_experiment_data = json.load(f)
                self._log_status("Experiment loaded successfully")
    
    def _export_results(self):
        """Export results to CSV."""
        if not self.lod_system:
            messagebox.showwarning("Error", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", 
                                                filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for key, value in self.lod_system.quality_metrics.items():
                    writer.writerow([key, value])
            self._log_status(f"Results exported to {file_path}")
            messagebox.showinfo("Success", "Results exported successfully")


def run_comparison_gui():
    """Launch the comparison GUI."""
    root = tk.Tk()
    gui = MeshComparisonGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_comparison_gui()
