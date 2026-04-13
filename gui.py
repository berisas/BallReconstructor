"""
GUI interface for BallReconstructor research workflow.
"""
import tkinter as tk
from tkinter import ttk
import threading
import os

from training import MLLODSystem


class ResearchGUI:
    """Research-focused GUI for BallReconstructor training."""
    
    def __init__(self, file_path=None):
        self.lod_system = None
        self.root = None
        self.file_path = file_path or "tennis_ball.obj"
        self.epochs_var = None
        self.progress_var = None
        self.status_label = None
        self.results_text = None
        self.ml_mesh_btn = None
    
    def create_gui(self):
        """Create and initialize the GUI."""
        if not os.path.exists(self.file_path):
            print(f"File not found: {self.file_path}")
            return None
        
        self.lod_system = MLLODSystem(self.file_path)
        
        self.root = tk.Tk()
        self.root.title("BallReconstructor - Research Interface")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        self._build_title()
        main_frame = self._build_main_frame()
        self._build_left_panel(main_frame)
        self._build_right_panel(main_frame)
        self._build_bottom_panel(main_frame)
        
        return self.root
    
    def _build_title(self):
        """Build application title section."""
        title = tk.Label(self.root, text="BallReconstructor Neural Mesh Super-Resolution",
                        font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#2c3e50")
        title.pack(pady=15)
        
        subtitle = tk.Label(self.root, text="Research-Focused ML Training Interface",
                           font=("Arial", 11), bg="#f0f0f0", fg="#7f8c8d")
        subtitle.pack(pady=5)
    
    def _build_main_frame(self):
        """Build main container."""
        frame = tk.Frame(self.root, bg="#f0f0f0")
        frame.pack(fill="both", expand=True, padx=20, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        return frame
    
    def _build_left_panel(self, parent):
        """Build mesh information panel."""
        panel = tk.LabelFrame(parent, text="Mesh Information",
                             font=("Arial", 11, "bold"), bg="white", padx=15, pady=15)
        panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        mesh_info = self._get_mesh_info_text()
        text = tk.Text(panel, height=15, width=40, font=("Consolas", 9),
                      wrap="word", bg="#f8f9fa", relief="flat")
        text.insert(1.0, mesh_info)
        text.config(state="disabled")
        text.pack(fill="both", expand=True)
        
        btn = tk.Button(panel, text="Preview Mesh", command=self.preview_mesh,
                       font=("Arial", 10), bg="#3498db", fg="white",
                       activebackground="#2980b9", cursor="hand2", height=2)
        btn.pack(pady=10, fill="x")
    
    def _get_mesh_info_text(self):
        """Generate mesh information text."""
        info = f"Original Mesh: {os.path.basename(self.file_path)}\n\n"
        info += f"Vertices: {len(self.lod_system.original_mesh.vertices):,}\n"
        info += f"Faces: {len(self.lod_system.original_mesh.faces):,}\n\n"
        info += "LOD Variants Generated:\n"
        
        for quality, mesh in self.lod_system.mesh_variants.items():
            reduction = ((len(self.lod_system.original_mesh.faces) - len(mesh.faces)) /
                        len(self.lod_system.original_mesh.faces)) * 100
            info += f"  • {quality:12s}: {len(mesh.faces):4,} faces ({reduction:5.1f}% reduction)\n"
        
        return info
    
    def _build_right_panel(self, parent):
        """Build training configuration panel."""
        panel = tk.LabelFrame(parent, text="Training Configuration",
                             font=("Arial", 11, "bold"), bg="white", padx=15, pady=15)
        panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self._build_epochs_slider(panel)
        self._build_model_info(panel)
        self._build_training_buttons(panel)
    
    def _build_epochs_slider(self, parent):
        """Build epochs configuration slider."""
        frame = tk.Frame(parent, bg="white")
        frame.pack(fill="x", pady=10)
        
        tk.Label(frame, text="Training Epochs:", font=("Arial", 10, "bold"), 
                bg="white").pack(anchor="w")
        
        self.epochs_var = tk.IntVar(value=200)
        slider = tk.Scale(frame, from_=50, to=1000, orient="horizontal",
                         variable=self.epochs_var, length=300, bg="white", font=("Arial", 9))
        slider.pack(fill="x", pady=5)
        
        tk.Label(frame, textvariable=self.epochs_var, font=("Arial", 10),
                bg="white", fg="#2c3e50").pack(anchor="w")
    
    def _build_model_info(self, parent):
        """Build model architecture information."""
        info = """Model Architecture:
  • Hidden Dimension: 512
  • Encoder: 3-layer dense + LayerNorm
  • Decoder: 5-layer residual network
  • Optimizer: Adam (LR=0.001, decay=0.95)
  • K-NN Neighbors: 8 (Gaussian weighting)

Training Strategy:
  • Gradient clipping: L2 norm ≤ 1.0
  • Loss: MSE + Smoothness regularization
  • Multi-scale progressive learning
"""
        text = tk.Text(parent, height=12, width=40, font=("Consolas", 8),
                      wrap="word", bg="#f8f9fa", relief="flat")
        text.insert(1.0, info)
        text.config(state="disabled")
        text.pack(fill="both", expand=True, pady=10)
    
    def _build_training_buttons(self, parent):
        """Build training control buttons."""
        btn = tk.Button(parent, text="Start Training", command=self.train_model,
                       font=("Arial", 12, "bold"), bg="#27ae60", fg="white",
                       activebackground="#229954", cursor="hand2", height=2)
        btn.pack(pady=10, fill="x")
        
        self.ml_mesh_btn = tk.Button(parent, text="View ML Enhanced Mesh",
                                    command=self.show_ml_enhanced_mesh,
                                    font=("Arial", 11), bg="#9b59b6", fg="white",
                                    activebackground="#8e44ad", cursor="hand2",
                                    state="disabled", height=2)
        self.ml_mesh_btn.pack(pady=5, fill="x")
    
    def _build_bottom_panel(self, parent):
        """Build training progress and metrics panel."""
        panel = tk.LabelFrame(parent, text="Training Progress & Metrics",
                             font=("Arial", 11, "bold"), bg="white", padx=15, pady=15)
        panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        self._build_progress_bar(panel)
        self._build_results_text(panel)
    
    def _build_progress_bar(self, parent):
        """Build progress bar and status label."""
        frame = tk.Frame(parent, bg="white")
        frame.pack(fill="x", pady=5)
        
        tk.Label(frame, text="Progress:", font=("Arial", 9), bg="white").pack(side="left", padx=5)
        
        self.progress_var = tk.DoubleVar()
        bar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100, length=400)
        bar.pack(side="left", fill="x", expand=True, padx=5)
        
        self.status_label = tk.Label(parent, text="Ready", font=("Arial", 10),
                                    bg="white", fg="#7f8c8d")
        self.status_label.pack(pady=5)
    
    def _build_results_text(self, parent):
        """Build results display text area."""
        self.results_text = tk.Text(parent, height=10, width=100,
                                   font=("Consolas", 9), wrap="word",
                                   bg="#f8f9fa", relief="flat")
        scrollbar = tk.Scrollbar(parent, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.results_text.pack(fill="both", expand=True, pady=5)
    
    def preview_mesh(self):
        """Show mesh variants preview."""
        self.lod_system.preview_mesh()
    
    def train_model(self):
        """Initiate training in background thread."""
        epochs = self.epochs_var.get()
        self.status_label.config(text=f"Training {epochs} epochs", fg="#e67e22")
        self.progress_var.set(0)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, f"Epochs: {epochs}\n\n")
        self.root.update()
        
        thread = threading.Thread(target=self._train_worker, args=(epochs,), daemon=True)
        thread.start()
    
    def _train_worker(self, epochs):
        """Background worker for model training."""
        success = self.lod_system.train_ml_model(epochs=epochs)
        if success:
            self.root.after(0, self._on_training_success)
        else:
            self.root.after(0, self._on_training_failed)
    
    def _on_training_success(self):
        """Handle successful training completion."""
        self.status_label.config(text="Training complete", fg="#27ae60")
        self.progress_var.set(100)
        self.ml_mesh_btn.config(state="normal")
        
        metrics = self.lod_system.quality_metrics
        results = f"""PERFORMANCE METRICS

Reconstruction:
  Mean Error: {metrics['reconstruction_error_mean']:.8f}
  Max Error: {metrics['reconstruction_error_max']:.8f}
  Final Loss: {metrics['final_training_loss']:.8f}
  Best Loss: {metrics['best_training_loss']:.8f}

Mesh:
  Face Count: {metrics['face_improvement']}
  Quality Gain: {metrics['quality_improvement']:.2f}%

Configuration:
  Epochs: {self.epochs_var.get()}
  Hidden Dim: 512
  LR: 0.001 (exponential decay)
  Gradient Clip: L2 ≤ 1.0
"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
    
    def _on_training_failed(self):
        """Handle training failure."""
        self.status_label.config(text="Training failed", fg="#e74c3c")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, "TRAINING FAILED\n\nCheck console for details.")
    
    def show_ml_enhanced_mesh(self):
        """Display the ML-enhanced mesh result."""
        if self.lod_system.ml_enhanced_mesh is None:
            print("ML enhanced mesh not available!")
            return
        
        import trimesh
        scene = trimesh.Scene()
        mesh = self.lod_system.ml_enhanced_mesh.copy()
        mesh.visual.vertex_colors = [100, 200, 255, 255]
        scene.add_geometry(mesh)
        
        print(f"ML Enhanced Mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
        scene.show()


def run_gui(file_path=None):
    """Run GUI application."""
    gui = ResearchGUI(file_path)
    root = gui.create_gui()
    if root:
        root.mainloop()
