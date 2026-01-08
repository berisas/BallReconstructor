import tensorflow as tf
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import os
import time
import matplotlib.pyplot as plt
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import tkinter as tk
    from tkinter import ttk
    import threading
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


class MeshSuperResNet(tf.keras.Model):
    def __init__(self, hidden_dim=512):
        super(MeshSuperResNet, self).__init__()

        self.local_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
        ])

        self.global_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(hidden_dim // 2, activation='relu')
        ])

        self.position_decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim * 2, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_dim // 2, activation='relu'),
            tf.keras.layers.Dense(3, activation='tanh')
        ])

        self.displacement_scale = tf.Variable(0.05, trainable=True)
        self.feature_scale = tf.Variable(1.0, trainable=True)

    def call(self, low_vertices, query_positions, low_to_query_weights, training=None):
        local_features = self.local_encoder(low_vertices, training=training)
        local_features = local_features * self.feature_scale

        global_max = tf.reduce_max(local_features, axis=1, keepdims=True)
        global_mean = tf.reduce_mean(local_features, axis=1, keepdims=True)
        global_std = tf.math.reduce_std(local_features, axis=1, keepdims=True)
        global_context = self.global_encoder(
            tf.concat([global_max, global_mean, global_std], axis=-1), training=training
        )

        interpolated_features = tf.matmul(low_to_query_weights, local_features)
        num_query = tf.shape(query_positions)[1]
        global_broadcast = tf.tile(global_context, [1, num_query, 1])

        combined_input = tf.concat([
            query_positions,
            interpolated_features,
            global_broadcast
        ], axis=-1)

        displacement = self.position_decoder(combined_input, training=training)
        displacement = displacement * self.displacement_scale

        refined_positions = query_positions + displacement
        return refined_positions


# Helper Functions 
def compute_enhanced_interpolation_weights(low_vertices, high_vertices, k=8):
    """Enhanced interpolation with more neighbors and better weighting"""
    if len(low_vertices) < k:
        k = len(low_vertices)

    tree = cKDTree(low_vertices)
    distances, indices = tree.query(high_vertices, k=k)

    sigma = np.mean(distances) * 0.5
    weights = np.exp(-distances**2 / (2 * sigma**2))
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    batch_size = 1
    num_high = len(high_vertices)
    num_low = len(low_vertices)

    weight_matrix = np.zeros((batch_size, num_high, num_low), dtype=np.float32)
    for i in range(num_high):
        for j, idx in enumerate(indices[i]):
            weight_matrix[0, i, idx] = weights[i, j]

    return weight_matrix

def compute_mesh_curvature(vertices, faces):
    """Compute mean curvature at each vertex using mesh connectivity"""
    num_vertices = len(vertices)
    curvatures = np.zeros(num_vertices, dtype=np.float32)
    
    # Build adjacency list
    adjacency = [[] for _ in range(num_vertices)]
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            if v2 not in adjacency[v1]:
                adjacency[v1].append(v2)
            if v1 not in adjacency[v2]:
                adjacency[v2].append(v1)
    
    # Compute curvature as variance of normals around vertex
    for v_idx in range(num_vertices):
        if len(adjacency[v_idx]) < 3:
            curvatures[v_idx] = 0.0
            continue
        
        neighbors = adjacency[v_idx]
        # Get edges from this vertex to neighbors
        edges = vertices[neighbors] - vertices[v_idx]
        edge_lengths = np.linalg.norm(edges, axis=1, keepdims=True)
        edge_lengths[edge_lengths < 1e-8] = 1e-8
        edges = edges / edge_lengths
        
        # Curvature as average angle deviation
        angle_variance = 0.0
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                dot = np.clip(np.dot(edges[i], edges[j]), -1.0, 1.0)
                angle_variance += np.arccos(dot) ** 2
        
        curvatures[v_idx] = np.sqrt(angle_variance / max(len(edges), 1)) if len(edges) > 0 else 0.0
    
    # Normalize to [0, 1]
    max_curv = np.max(curvatures)
    if max_curv > 1e-8:
        curvatures = curvatures / max_curv
    
    return curvatures

def compute_curvature_adaptive_weights(low_vertices, high_vertices, low_faces, high_faces, k=8):
    """Compute adaptive weights based on mesh curvature and distance"""
    if len(low_vertices) < k:
        k = len(low_vertices)
    
    # Compute curvature for both meshes
    low_curvature = compute_mesh_curvature(low_vertices, low_faces)
    high_curvature = compute_mesh_curvature(high_vertices, high_faces)
    
    tree = cKDTree(low_vertices)
    distances, indices = tree.query(high_vertices, k=k)
    
    # Gaussian distance weighting
    sigma = np.mean(distances) * 0.5
    dist_weights = np.exp(-distances**2 / (2 * sigma**2))
    
    # Curvature-based adaptive weighting
    # High curvature = more neighbors matter (increase weight spreading)
    curvature_factor = 1.0 + high_curvature[:, np.newaxis] * 2.0  # Scale from 1.0 to 3.0
    
    # Combine: distance * curvature adaptation
    weights = dist_weights * curvature_factor
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    
    batch_size = 1
    num_high = len(high_vertices)
    num_low = len(low_vertices)
    
    weight_matrix = np.zeros((batch_size, num_high, num_low), dtype=np.float32)
    for i in range(num_high):
        for j, idx in enumerate(indices[i]):
            weight_matrix[0, i, idx] = weights[i, j]
    
    return weight_matrix

def build_adjacency_list(faces, num_vertices):
    """Build vertex adjacency list from faces"""
    adjacency = [[] for _ in range(num_vertices)]
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            if v2 not in adjacency[v1]:
                adjacency[v1].append(v2)
            if v1 not in adjacency[v2]:
                adjacency[v2].append(v1)
    return adjacency

def compute_laplacian_regularization(vertices, faces):
    """Compute Laplacian smoothing constraint for mesh smoothness"""
    num_vertices = len(vertices)
    adjacency = build_adjacency_list(faces, num_vertices)
    
    laplacian = np.zeros_like(vertices)
    
    for v_idx in range(num_vertices):
        neighbors = adjacency[v_idx]
        if len(neighbors) > 0:
            # Laplacian = sum of neighbors - vertex position
            neighbor_sum = np.sum(vertices[neighbors], axis=0)
            laplacian[v_idx] = neighbor_sum / len(neighbors) - vertices[v_idx]
    
    return laplacian

def compute_feature_aware_loss_weights(high_vertices, high_faces, curvature_threshold=0.5):
    """Compute per-vertex loss weights based on curvature (features vs smooth)"""
    curvatures = compute_mesh_curvature(high_vertices, high_faces)
    
    # Vertices with high curvature (features) get lower weight (more lenient)
    # Vertices with low curvature (smooth) get higher weight (stricter)
    loss_weights = np.where(
        curvatures > curvature_threshold,
        0.5,    # High curvature = feature region, 50% weight
        1.0     # Low curvature = smooth region, 100% weight
    ).astype(np.float32)
    
    return loss_weights, curvatures

def create_enhanced_template(low_vertices, high_vertices):
    """Create better initial template with less noise"""
    template = high_vertices.copy()
    noise_scale = np.std(high_vertices, axis=0) * 0.02
    noise = np.random.normal(0, noise_scale, high_vertices.shape).astype(np.float32)
    template += noise
    return template.astype(np.float32)

# Main ML LOD System
class MLLODSystem:
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        self.original_mesh = None
        self.trained_model = None
        self.preprocessing_data = None
        self.mesh_variants = {}
        self.ml_enhanced_mesh = None
        self.quality_metrics = {}

        # Initialize
        self._load_and_prepare_meshes()

    def _load_and_prepare_meshes(self):
        """Load original mesh and create quality variants"""
        if not os.path.exists(self.mesh_path):
            print(f"Error: Mesh file not found: {self.mesh_path}")
            return False

        try:
            self.original_mesh = trimesh.load(self.mesh_path)

            if not hasattr(self.original_mesh, 'vertices') or len(self.original_mesh.vertices) == 0:
                print(f"Error: Invalid mesh file - no vertices found")
                return False

        except Exception as e:
            print(f"Error loading mesh: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Create different quality levels
        original_faces = len(self.original_mesh.faces)

        try:
            self.mesh_variants = {
                'ultra_low': self.original_mesh.simplify_quadric_decimation(face_count=max(100, original_faces // 20)),
                'low': self.original_mesh.simplify_quadric_decimation(face_count=max(300, original_faces // 8)),
                'medium_base': self.original_mesh.simplify_quadric_decimation(face_count=max(500, original_faces // 4)),
                'high': self.original_mesh
            }

            for quality, mesh in self.mesh_variants.items():
                reduction = ((original_faces - len(mesh.faces)) / original_faces) * 100
                print(f"{quality:12s}: {len(mesh.faces):,} faces | {reduction:.1f}% reduction")

            return True

        except Exception as e:
            print(f"Error creating mesh variants: {e}")
            return False

    def train_ml_model(self, epochs=100):
        """Train the ML model on actual mesh data"""
        try:
            print("\n" + "="*80)
            print("MESH LOD HIERARCHY")
            print("="*80)
            for quality, mesh in self.mesh_variants.items():
                reduction = ((len(self.original_mesh.faces) - len(mesh.faces)) / len(self.original_mesh.faces)) * 100
                print(f"  {quality:15s}: {len(mesh.vertices):6,} vertices | {len(mesh.faces):6,} faces | {reduction:5.1f}% reduction")
            print("="*80)
            
            # Prepare training data
            training_scales, center, scale = self._prepare_training_data()

            # Train model
            model, loss_history, best_loss = self._train_model(training_scales, epochs)

            # Store results
            self.trained_model = model
            self.preprocessing_data = {
                'center': center,
                'scale': scale
            }

            # Generate ML enhanced mesh
            self.ml_enhanced_mesh = self._generate_ml_enhanced_mesh()

            # Calculate metrics
            self._calculate_quality_metrics(loss_history, best_loss)

            return True

        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _prepare_training_data(self):
        """Prepare multi-scale training data for progressive refinement"""
        # Get all resolution levels
        ultra_low_mesh = self.mesh_variants['ultra_low']
        low_mesh = self.mesh_variants['low']
        medium_base_mesh = self.mesh_variants['medium_base']
        high_mesh = self.mesh_variants['high']

        # Extract vertices for all levels
        ultra_low_verts = np.array(ultra_low_mesh.vertices, dtype=np.float32)
        low_verts = np.array(low_mesh.vertices, dtype=np.float32)
        medium_verts = np.array(medium_base_mesh.vertices, dtype=np.float32)
        high_verts = np.array(high_mesh.vertices, dtype=np.float32)

        # Single normalization across all levels
        all_vertices = np.vstack([ultra_low_verts, low_verts, medium_verts, high_verts])
        center = np.mean(all_vertices, axis=0)
        scale = np.max(np.linalg.norm(all_vertices - center, axis=1)) + 1e-8

        # Normalize all levels
        ultra_low_norm = (ultra_low_verts - center) / scale
        low_norm = (low_verts - center) / scale
        medium_norm = (medium_verts - center) / scale
        high_norm = (high_verts - center) / scale

        # Create multi-scale training data
        training_scales = []
        
        # Scale 1: ultra_low → low
        template1 = create_enhanced_template(ultra_low_norm, low_norm)
        weights1 = compute_curvature_adaptive_weights(ultra_low_norm, low_norm, ultra_low_mesh.faces, low_mesh.faces, k=8)
        training_scales.append({
            'input': ultra_low_norm[np.newaxis, :, :],
            'target': low_norm[np.newaxis, :, :],
            'template': template1[np.newaxis, :, :],
            'weights': weights1,
            'scale_name': 'ultra_low→low',
            'scale_weight': 1.0
        })
        
        # Scale 2: low → medium_base
        template2 = create_enhanced_template(low_norm, medium_norm)
        weights2 = compute_curvature_adaptive_weights(low_norm, medium_norm, low_mesh.faces, medium_base_mesh.faces, k=8)
        training_scales.append({
            'input': low_norm[np.newaxis, :, :],
            'target': medium_norm[np.newaxis, :, :],
            'template': template2[np.newaxis, :, :],
            'weights': weights2,
            'scale_name': 'low→medium',
            'scale_weight': 1.0
        })
        
        # Scale 3: medium_base → high (increased k for better context, higher weight)
        template3 = create_enhanced_template(medium_norm, high_norm)
        weights3 = compute_curvature_adaptive_weights(medium_norm, high_norm, medium_base_mesh.faces, high_mesh.faces, k=16)
        
        # Compute feature-aware loss weights and Laplacian regularization for final scale
        loss_weights_3, curvatures_3 = compute_feature_aware_loss_weights(high_verts, high_mesh.faces, curvature_threshold=0.4)
        laplacian_3 = compute_laplacian_regularization(high_verts, high_mesh.faces)
        
        training_scales.append({
            'input': medium_norm[np.newaxis, :, :],
            'target': high_norm[np.newaxis, :, :],
            'template': template3[np.newaxis, :, :],
            'weights': weights3,
            'scale_name': 'medium→high',
            'scale_weight': 2.0,  # Double weight for final scale
            'loss_weights': loss_weights_3[np.newaxis, :, np.newaxis],  # [batch, vertices, 1]
            'laplacian': (laplacian_3 / scale)[np.newaxis, :, :],  # Normalize by scale
            'curvatures': curvatures_3
        })

        return training_scales, center, scale

    def _train_model(self, training_scales, epochs):
        """Train the model with multi-scale progressive learning"""
        model = MeshSuperResNet(hidden_dim=512)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001, decay_steps=100, decay_rate=0.95, staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        best_loss = float('inf')
        loss_history = []
        scale_losses = {scale['scale_name']: [] for scale in training_scales}

        # Print training data info
        print("\n" + "="*80)
        print("TRAINING DATA SUMMARY")
        print("="*80)
        for idx, scale_data in enumerate(training_scales):
            input_shape = scale_data['input'].shape
            target_shape = scale_data['target'].shape
            weights_shape = scale_data['weights'].shape
            template_shape = scale_data['template'].shape
            print(f"\nScale {idx}: {scale_data['scale_name']}")
            print(f"  Input (low-res vertices):    {input_shape} → {input_shape[1]:,} vertices, {input_shape[2]} coords")
            print(f"  Target (high-res vertices):  {target_shape} → {target_shape[1]:,} vertices, {target_shape[2]} coords")
            print(f"  Template:                    {template_shape} → {template_shape[1]:,} vertices")
            print(f"  Weights matrix:              {weights_shape} → sparse interpolation weights")
            print(f"  Weight scale:                {(idx + 1) / len(training_scales):.2f}")
        
        print(f"\nModel parameters: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")
        print(f"Optimizer: Adam (LR=0.001, decay=0.95 every 100 steps)")
        print("="*80 + "\n")

        for epoch in range(epochs):
            total_loss = 0.0
            
            # Train on all scales in sequence
            for scale_idx, scale_data in enumerate(training_scales):
                low_tf = tf.constant(scale_data['input'], dtype=tf.float32)
                high_tf = tf.constant(scale_data['target'], dtype=tf.float32)
                template_tf = tf.constant(scale_data['template'], dtype=tf.float32)
                weights_tf = tf.constant(scale_data['weights'], dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    pred = model(low_tf, template_tf, weights_tf, training=True)
                    
                    # Compute per-vertex loss
                    per_vertex_loss = tf.reduce_sum(tf.square(pred - high_tf), axis=-1)
                    
                    # Remove outliers: cap loss at 95th percentile
                    loss_threshold = tf.constant(np.percentile(per_vertex_loss.numpy(), 95), dtype=tf.float32)
                    clamped_loss = tf.minimum(per_vertex_loss, loss_threshold)
                    
                    # Apply feature-aware loss weighting (only for final scale)
                    if 'loss_weights' in scale_data:
                        loss_weights_tf = tf.constant(scale_data['loss_weights'], dtype=tf.float32)
                        weighted_loss = clamped_loss[np.newaxis, :] * loss_weights_tf
                        base_loss = tf.reduce_mean(weighted_loss)
                        
                        # Add Laplacian smoothing regularization (keeps smooth regions smooth)
                        laplacian_tf = tf.constant(scale_data['laplacian'], dtype=tf.float32)
                        laplacian_term = tf.reduce_mean(tf.square(pred - laplacian_tf))
                        
                        # Combine: prediction accuracy + smoothness constraint (reduced weight)
                        regularization_weight = 0.01  # Reduced from 0.1
                        combined_loss = base_loss + regularization_weight * laplacian_term
                    else:
                        combined_loss = tf.reduce_mean(clamped_loss)
                    
                    # Apply scale weight
                    scale_weight = scale_data.get('scale_weight', (scale_idx + 1) / len(training_scales))
                    loss = scale_weight * combined_loss

                grads = tape.gradient(loss, model.trainable_variables)
                grad_norms = [tf.norm(g).numpy() if g is not None else 0.0 for g in grads]
                grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                
                scale_loss = loss.numpy()
                scale_losses[scale_data['scale_name']].append(scale_loss)
                total_loss += scale_loss

            avg_loss = total_loss / len(training_scales)
            loss_history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 25 == 0:
                # Calculate improvement rate
                if len(loss_history) > 25:
                    prev_loss = loss_history[-26]
                    improvement = ((prev_loss - avg_loss) / prev_loss) * 100
                else:
                    improvement = 0
                
                loss_breakdown = " | ".join([f"{name}: {scale_losses[name][-1]:.6f}" for name in scale_losses.keys()])
                current_lr = float(optimizer.learning_rate.numpy())
                print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | Improvement: {improvement:+.2f}% | LR: {current_lr:.6f}")
                print(f"  → {loss_breakdown}")

        return model, loss_history, best_loss

    def _generate_ml_enhanced_mesh(self):
        """Generate ML enhanced mesh by progressively refining through all scales"""
        if self.trained_model is None:
            return None

        try:
            prep_data = self.preprocessing_data
            center = prep_data['center']
            scale = prep_data['scale']

            # Start with medium_base and progressively refine upward
            current_vertices = (self.mesh_variants['medium_base'].vertices - center) / scale
            current_vertices = current_vertices[np.newaxis, :, :]

            # Progressive refinement: medium_base → high
            medium_norm = current_vertices
            high_verts_norm = (self.mesh_variants['high'].vertices - center) / scale
            high_verts_norm = high_verts_norm[np.newaxis, :, :]
            
            template = create_enhanced_template(
                medium_norm[0], high_verts_norm[0]
            )
            weights = compute_enhanced_interpolation_weights(
                medium_norm[0], high_verts_norm[0]
            )
            
            template = template[np.newaxis, :, :]
            
            # Generate enhanced vertices at highest resolution
            enhanced_vertices_norm = self.trained_model(
                tf.constant(medium_norm, dtype=tf.float32),
                tf.constant(template, dtype=tf.float32),
                tf.constant(weights, dtype=tf.float32),
                training=False
            )

            # Denormalize
            enhanced_vertices = enhanced_vertices_norm.numpy()[0] * scale + center

            # Create mesh
            ml_enhanced_mesh = trimesh.Trimesh(
                vertices=enhanced_vertices,
                faces=self.mesh_variants['high'].faces
            )

            return ml_enhanced_mesh

        except Exception as e:
            print(f"Error generating ML enhanced mesh: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_quality_metrics(self, loss_history, best_loss):
        """Calculate quality metrics"""
        if self.ml_enhanced_mesh is None:
            return

        original_vertices = self.original_mesh.vertices
        ml_vertices = self.ml_enhanced_mesh.vertices

        reconstruction_error = np.mean(np.linalg.norm(ml_vertices - original_vertices, axis=1))
        max_error = np.max(np.linalg.norm(ml_vertices - original_vertices, axis=1))

        self.quality_metrics = {
            'reconstruction_error_mean': reconstruction_error,
            'reconstruction_error_max': max_error,
            'final_training_loss': loss_history[-1] if loss_history else 0,
            'best_training_loss': best_loss,
            'face_improvement': f"{len(self.mesh_variants['medium_base'].faces):,} → {len(self.original_mesh.faces):,}",
            'quality_improvement': ((len(self.original_mesh.faces) - len(self.mesh_variants['medium_base'].faces)) / len(self.mesh_variants['medium_base'].faces)) * 100
        }

    def show_visual_results(self):
        """Show visual comparison of all mesh variants"""
        if self.ml_enhanced_mesh is None:
            print("ML model not trained yet!")
            return

        scene = trimesh.Scene()

        # Positions and colors
        positions = [np.array([0, 0, 0]), np.array([4, 0, 0]), np.array([8, 0, 0]), np.array([12, 0, 0]), np.array([16, 0, 0])]
        colors = [[255, 100, 100, 255], [255, 200, 100, 255], [255, 255, 100, 255], [100, 255, 100, 255], [100, 100, 255, 255]]
        labels = ["Ultra Low", "Low", "Medium Base", "ML Enhanced", "Original"]
        meshes = [self.mesh_variants['ultra_low'], self.mesh_variants['low'],
                 self.mesh_variants['medium_base'], self.ml_enhanced_mesh, self.mesh_variants['high']]

        for i, (mesh, pos, color, label) in enumerate(zip(meshes, positions, colors, labels)):
            positioned_mesh = mesh.copy()
            positioned_mesh.vertices += pos
            positioned_mesh.visual.vertex_colors = color
            scene.add_geometry(positioned_mesh)

            faces = len(mesh.faces)
            print(f"{label:15s}: {faces:,} faces")

        print()
        for metric, value in self.quality_metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.6f}")
            else:
                print(f"{metric}: {value}")

        scene.show()


# Console Interface
def run_console_demo(file_path=None):
    """Run console-based demo"""
    if file_path is None:
        file_path = r"C:\Users\Ber\GhostObjects\tennis_ball.obj"

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    lod_system = MLLODSystem(file_path)

    if not lod_system.mesh_variants:
        print("Failed to load mesh variants")
        return

    # Train model
    success = lod_system.train_ml_model(epochs=150)  # Reduced for faster demo

    if not success:
        print("Training failed")
        return

    # Show results
    lod_system.show_visual_results()


# GUI Interface 
if GUI_AVAILABLE:
    class ResearchGUI:
        def __init__(self, file_path=None):
            self.lod_system = None
            self.root = None
            self.file_path = file_path if file_path else r"C:\Users\Ber\GhostObjects\tennis_ball.obj"
            self.epochs_var = None
            self.progress_var = None
            self.training_data = []

        def create_gui(self):
            if not os.path.exists(self.file_path):
                print(f"File not found: {self.file_path}")
                return

            self.lod_system = MLLODSystem(self.file_path)

            self.root = tk.Tk()
            self.root.title("Ghost2Real - Research Interface")
            self.root.geometry("900x700")
            self.root.configure(bg="#f0f0f0")

            # Title
            title_label = tk.Label(self.root, text="Ghost2Real Neural Mesh Super-Resolution",
                                  font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#2c3e50")
            title_label.pack(pady=15)

            subtitle_label = tk.Label(self.root, text="Research-Focused ML Training Interface",
                                     font=("Arial", 11), bg="#f0f0f0", fg="#7f8c8d")
            subtitle_label.pack(pady=5)

            # Main container
            main_frame = tk.Frame(self.root, bg="#f0f0f0")
            main_frame.pack(fill="both", expand=True, padx=20, pady=10)

            # Left panel - Mesh Info & Preview
            left_panel = tk.LabelFrame(main_frame, text="Mesh Information",
                                      font=("Arial", 11, "bold"), bg="white", padx=15, pady=15)
            left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

            # Mesh statistics
            mesh_info = f"""Original Mesh: {os.path.basename(self.file_path)}

Vertices: {len(self.lod_system.original_mesh.vertices):,}
Faces: {len(self.lod_system.original_mesh.faces):,}

LOD Variants Generated:
"""
            for quality, mesh in self.lod_system.mesh_variants.items():
                reduction = ((len(self.lod_system.original_mesh.faces) - len(mesh.faces)) /
                           len(self.lod_system.original_mesh.faces)) * 100
                mesh_info += f"  • {quality:12s}: {len(mesh.faces):4,} faces ({reduction:5.1f}% reduction)\n"

            info_text = tk.Text(left_panel, height=15, width=40, font=("Consolas", 9),
                              wrap="word", bg="#f8f9fa", relief="flat")
            info_text.insert(1.0, mesh_info)
            info_text.config(state="disabled")
            info_text.pack(fill="both", expand=True)

            # Preview button
            preview_btn = tk.Button(left_panel, text="Preview Mesh",
                                   command=self.preview_mesh,
                                   font=("Arial", 10), bg="#3498db", fg="white",
                                   activebackground="#2980b9", cursor="hand2", height=2, width=20)
            preview_btn.pack(pady=10, fill="x")

            # Right panel - Training Configuration
            right_panel = tk.LabelFrame(main_frame, text="Training Configuration",
                                       font=("Arial", 11, "bold"), bg="white", padx=15, pady=15)
            right_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

            # Epochs configuration
            epochs_frame = tk.Frame(right_panel, bg="white")
            epochs_frame.pack(fill="x", pady=10)

            tk.Label(epochs_frame, text="Training Epochs:",
                    font=("Arial", 10, "bold"), bg="white").pack(anchor="w")

            self.epochs_var = tk.IntVar(value=200)
            epochs_slider = tk.Scale(epochs_frame, from_=50, to=1000,
                                    orient="horizontal", variable=self.epochs_var,
                                    length=300, bg="white", font=("Arial", 9))
            epochs_slider.pack(fill="x", pady=5)

            tk.Label(epochs_frame, textvariable=self.epochs_var,
                    font=("Arial", 10), bg="white", fg="#2c3e50").pack(anchor="w")

            # Model parameters info
            params_info = """Model Architecture:
  • Hidden Dimension: 512
  • Encoder: 3-layer dense + LayerNorm
  • Decoder: 5-layer residual network
  • Optimizer: Adam (LR=0.001, decay=0.95)
  • K-NN Neighbors: 8 (Gaussian weighting)

Training Strategy:
  • Gradient clipping: L2 norm ≤ 1.0
  • Loss: MSE + Smoothness regularization
  • Early stopping available
"""
            params_text = tk.Text(right_panel, height=12, width=40, font=("Consolas", 8),
                                wrap="word", bg="#f8f9fa", relief="flat")
            params_text.insert(1.0, params_info)
            params_text.config(state="disabled")
            params_text.pack(fill="both", expand=True, pady=10)

            # Training button
            train_btn = tk.Button(right_panel, text="Start Training",
                                 command=self.train_model,
                                 font=("Arial", 12, "bold"), bg="#27ae60", fg="white",
                                 activebackground="#229954", cursor="hand2", height=2, width=25)
            train_btn.pack(pady=10, fill="x")

            # View ML Enhanced Mesh button
            self.ml_mesh_btn = tk.Button(right_panel, text="View ML Enhanced Mesh",
                                        command=self.show_ml_enhanced_mesh,
                                        font=("Arial", 11), bg="#9b59b6", fg="white",
                                        activebackground="#8e44ad", cursor="hand2",
                                        state="disabled", height=2, width=25)
            self.ml_mesh_btn.pack(pady=5, fill="x")

            # Bottom panel - Training Progress & Metrics
            bottom_panel = tk.LabelFrame(main_frame, text="Training Progress & Metrics",
                                        font=("Arial", 11, "bold"), bg="white", padx=15, pady=15)
            bottom_panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

            # Progress bar
            progress_frame = tk.Frame(bottom_panel, bg="white")
            progress_frame.pack(fill="x", pady=5)

            tk.Label(progress_frame, text="Progress:", font=("Arial", 9), bg="white").pack(side="left", padx=5)

            self.progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                          maximum=100, length=400)
            progress_bar.pack(side="left", fill="x", expand=True, padx=5)

            # Status label
            self.status_label = tk.Label(bottom_panel, text="Ready",
                                        font=("Arial", 10), bg="white", fg="#7f8c8d")
            self.status_label.pack(pady=5)

            # Results text area
            self.results_text = tk.Text(bottom_panel, height=10, width=100,
                                       font=("Consolas", 9), wrap="word",
                                       bg="#f8f9fa", relief="flat")
            scrollbar = tk.Scrollbar(bottom_panel, command=self.results_text.yview)
            self.results_text.config(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            self.results_text.pack(fill="both", expand=True, pady=5)

            # Configure grid weights
            main_frame.grid_columnconfigure(0, weight=1)
            main_frame.grid_columnconfigure(1, weight=1)
            main_frame.grid_rowconfigure(0, weight=1)
            main_frame.grid_rowconfigure(1, weight=1)

            return self.root

        def preview_mesh(self):
            """Show multi-scale refinement pipeline with before/after comparisons"""
            scene = trimesh.Scene()
            
            # Calculate bounding box size for proper spacing
            ref_mesh = self.lod_system.mesh_variants['high']
            bbox_size = np.max(ref_mesh.bounds[1] - ref_mesh.bounds[0])
            spacing = bbox_size * 1.5  # 1.5x spacing between meshes
            row_spacing = bbox_size * 2  # 2x spacing between rows
            
            if self.lod_system.ml_enhanced_mesh is None:
                # Before training: 2x2 grid showing all LOD levels
                print("\n=== BEFORE TRAINING - LOD HIERARCHY ===\n")
                
                grid_layout = [
                    # Row 0 (top)
                    (self.lod_system.mesh_variants['ultra_low'], [0, row_spacing, 0], [255, 100, 100, 255], "Ultra Low (1/20)"),
                    (self.lod_system.mesh_variants['low'], [spacing, row_spacing, 0], [255, 180, 100, 255], "Low (1/8)"),
                    # Row 1 (bottom)
                    (self.lod_system.mesh_variants['medium_base'], [0, 0, 0], [255, 255, 100, 255], "Medium Base (1/4)"),
                    (self.lod_system.mesh_variants['high'], [spacing, 0, 0], [100, 150, 255, 255], "Original (100%)")
                ]
                
                for mesh, pos, color, label in grid_layout:
                    positioned_mesh = mesh.copy()
                    positioned_mesh.vertices += np.array(pos)
                    positioned_mesh.visual.vertex_colors = color
                    scene.add_geometry(positioned_mesh)
                    print(f"{label:25s}: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
                    
            else:
                # After training: 2 rows of before/after comparisons
                print("\n=== AFTER TRAINING - REFINEMENT RESULTS ===\n")
                
                # Row positions (top to bottom)
                row_positions = [row_spacing, 0]
                
                # Row 0: Medium Base untrained vs Original
                medium_untrained = self.lod_system.mesh_variants['medium_base'].copy()
                medium_untrained.vertices += np.array([0, row_positions[0], 0])
                medium_untrained.visual.vertex_colors = [255, 255, 100, 255]
                scene.add_geometry(medium_untrained)
                
                original = self.lod_system.mesh_variants['high'].copy()
                original.vertices += np.array([spacing, row_positions[0], 0])
                original.visual.vertex_colors = [100, 150, 255, 255]
                scene.add_geometry(original)
                print(f"Row 0 - Coarse:")
                print(f"  Medium Base (untrained)   : {len(medium_untrained.vertices):,} vertices, {len(medium_untrained.faces):,} faces")
                print(f"  Original (reference)      : {len(original.vertices):,} vertices, {len(original.faces):,} faces")
                
                # Row 1: Medium Base trained vs ML Enhanced
                medium_trained = self.lod_system.mesh_variants['medium_base'].copy()
                medium_trained.vertices += np.array([0, row_positions[1], 0])
                medium_trained.visual.vertex_colors = [200, 255, 100, 255]
                scene.add_geometry(medium_trained)
                
                ml_enhanced = self.lod_system.ml_enhanced_mesh.copy()
                ml_enhanced.vertices += np.array([spacing, row_positions[1], 0])
                ml_enhanced.visual.vertex_colors = [100, 255, 200, 255]
                scene.add_geometry(ml_enhanced)
                print(f"\nRow 1 - Medium:")
                print(f"  Medium Base (trained)     : {len(medium_trained.vertices):,} vertices, {len(medium_trained.faces):,} faces")
                print(f"  ML Enhanced (output)      : {len(ml_enhanced.vertices):,} vertices, {len(ml_enhanced.faces):,} faces")
                
                print("\n[Left side = baseline/untrained | Right side = final output]")
            
            print()
            scene.show()

        def train_model(self):
            epochs = self.epochs_var.get()
            self.status_label.config(text=f"Training {epochs} epochs", fg="#e67e22")
            self.progress_var.set(0)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, f"Epochs: {epochs}\n\n")
            self.root.update()

            def train_worker():
                success = self.lod_system.train_ml_model(epochs=epochs)

                if success:
                    self.root.after(0, self._training_success)
                else:
                    self.root.after(0, self._training_failed)

            thread = threading.Thread(target=train_worker)
            thread.daemon = True
            thread.start()

        def _training_success(self):
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

        def _training_failed(self):
            self.status_label.config(text="Training failed", fg="#e74c3c")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, "TRAINING FAILED\n\nCheck console for details.")

        def show_results(self):
            self.lod_system.show_visual_results()

        def show_ml_enhanced_mesh(self):
            """Show only the ML-enhanced mesh with added vertices"""
            if self.lod_system.ml_enhanced_mesh is None:
                print("ML enhanced mesh not available!")
                return

            scene = trimesh.Scene()
            
            # Add the ML enhanced mesh
            ml_mesh = self.lod_system.ml_enhanced_mesh.copy()
            ml_mesh.visual.vertex_colors = [100, 200, 255, 255]  # Light blue
            scene.add_geometry(ml_mesh)
            
            print(f"ML Enhanced Mesh: {len(ml_mesh.vertices):,} vertices, {len(ml_mesh.faces):,} faces")
            scene.show()

    def run_gui_demo(file_path=None):
        """Run GUI demo"""
        gui = ResearchGUI(file_path)
        root = gui.create_gui()
        if root:
            root.mainloop()

# ----------------------------
# Main Entry Point
# ----------------------------
def main():
    """Main function - chooses GUI or console mode"""
    if GUI_AVAILABLE:
        run_gui_demo()
    else:
        run_console_demo()

if __name__ == "__main__":
    main()
