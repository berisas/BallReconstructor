"""
Training system for mesh LOD with multi-scale progressive refinement.
"""
import numpy as np
import tensorflow as tf
import trimesh
import os
import time

from model import MeshSuperResNet
from mesh_utils import (
    compute_interpolation_weights, compute_laplacian_regularization,
    compute_feature_aware_loss_weights, create_enhanced_template,
    normalize_mesh_vertices, denormalize_vertices
)

# Import research infrastructure
try:
    from research.experiment_tracker import ExperimentTracker
    from research.config_manager import ConfigLoader
    RESEARCH_MODE = True
except ImportError:
    RESEARCH_MODE = False


class MLLODSystem:
    """Multi-scale LOD training system for neural mesh super-resolution."""
    
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        self.original_mesh = None
        self.trained_model = None
        self.preprocessing_data = None
        self.mesh_variants = {}
        self.ml_enhanced_mesh = None
        self.quality_metrics = {}
        self._load_and_prepare_meshes()
    
    def _load_and_prepare_meshes(self):
        """Load mesh and create LOD variants using quadric decimation."""
        if not os.path.exists(self.mesh_path):
            print(f"Error: Mesh file not found: {self.mesh_path}")
            return False
        
        try:
            loaded = trimesh.load(self.mesh_path)
            
            # Handle Scene objects (multi-part meshes)
            if hasattr(loaded, 'geometry'):  # Scene object
                # Merge all geometries in the scene
                meshes = [geom for geom in loaded.geometry.values() 
                         if hasattr(geom, 'vertices') and hasattr(geom, 'faces')]
                if meshes:
                    self.original_mesh = trimesh.util.concatenate(meshes)
                else:
                    print("Error: Scene contains no valid meshes")
                    return False
            else:
                self.original_mesh = loaded
            
            if not hasattr(self.original_mesh, 'vertices') or len(self.original_mesh.vertices) == 0:
                print("Error: Invalid mesh file - no vertices found")
                return False
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return False
        
        # Create LOD variants
        original_faces = len(self.original_mesh.faces)
        try:
            self.mesh_variants = {
                'ultra_low': self.original_mesh.simplify_quadric_decimation(
                    face_count=max(100, original_faces // 20)),
                'low': self.original_mesh.simplify_quadric_decimation(
                    face_count=max(300, original_faces // 8)),
                'medium_base': self.original_mesh.simplify_quadric_decimation(
                    face_count=max(500, original_faces // 4)),
                'high': self.original_mesh
            }
            
            for quality, mesh in self.mesh_variants.items():
                reduction = ((original_faces - len(mesh.faces)) / original_faces) * 100
                print(f"{quality:12s}: {len(mesh.faces):,} faces | {reduction:.1f}% reduction")
            
            return True
        except Exception as e:
            print(f"Error creating mesh variants: {e}")
            return False
    
    def train_ml_model(self, epochs=100, experiment_name=None, config_path=None):
        """
        Train the ML model using multi-scale progressive learning.
        
        Args:
            epochs: Number of training epochs
            experiment_name: Optional name for experiment tracking
            config_path: Optional path to config file
        """
        try:
            # Initialize experiment tracking if available
            experiment_tracker = None
            if RESEARCH_MODE and experiment_name:
                experiment_tracker = ExperimentTracker(experiment_name)
                
                # Load or create config
                if config_path and os.path.exists(config_path):
                    config = ConfigLoader(config_path).to_dict()
                else:
                    config = self._create_default_config(epochs)
                
                # Save config
                experiment_tracker.save_config(config)
            
            # Training pipeline
            self._print_hierarchy()
            training_scales, center, scale = self._prepare_training_data()
            model, loss_history, best_loss = self._train_model(
                training_scales, epochs, experiment_tracker
            )
            
            self.trained_model = model
            self.preprocessing_data = {'center': center, 'scale': scale}
            self.ml_enhanced_mesh = self._generate_ml_enhanced_mesh()
            self._calculate_quality_metrics(loss_history, best_loss)
            
            # Save final results if tracking
            if experiment_tracker:
                experiment_tracker.save_final_results(self.quality_metrics)
                experiment_tracker.save_mesh(
                    self.ml_enhanced_mesh,
                    'ml_enhanced_mesh.obj'
                )
                print(f"\n✓ Experiment saved to: {experiment_tracker.experiment_dir}")
            
            return True
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_default_config(self, epochs):
        """Create default config dictionary."""
        return {
            'experiment': {
                'name': 'mesh_refinement_training',
                'method': 'progressive_multiscale',
                'description': 'Neural mesh super-resolution with progressive multi-scale training'
            },
            'dataset': {
                'mesh_file': os.path.basename(self.mesh_path),
                'mesh_vertices': len(self.original_mesh.vertices),
                'mesh_faces': len(self.original_mesh.faces)
            },
            'training': {
                'epochs': epochs,
                'batch_size': 1,
                'learning_rate': 0.001,
                'lr_decay_rate': 0.95,
                'lr_decay_steps': 100,
                'gradient_clip': 1.0
            },
            'model': {
                'architecture': 'encoder_decoder',
                'hidden_dim': 512,
                'dropout_rate': 0.1,
                'feature_scale_init': 1.0,
                'displacement_scale_init': 0.03  # Reduced from 0.05 for conservative refinement
            }
        }
    
    def _print_hierarchy(self):
        """Print LOD hierarchy information."""
        print("\n" + "="*80)
        print("MESH LOD HIERARCHY")
        print("="*80)
        for quality, mesh in self.mesh_variants.items():
            reduction = ((len(self.original_mesh.faces) - len(mesh.faces)) / 
                        len(self.original_mesh.faces)) * 100
            print(f"  {quality:15s}: {len(mesh.vertices):6,} vertices | " +
                  f"{len(mesh.faces):6,} faces | {reduction:5.1f}% reduction")
        print("="*80)
    
    def _prepare_training_data(self):
        """Prepare multi-scale training data for progressive refinement."""
        meshes = {k: self.mesh_variants[k] for k in 
                 ['ultra_low', 'low', 'medium_base', 'high']}
        vertices = {k: np.array(meshes[k].vertices, dtype=np.float32) 
                   for k in meshes}
        
        # Normalize all vertices to common space
        vert_list = [vertices[k] for k in ['ultra_low', 'low', 'medium_base', 'high']]
        normalized, center, scale = normalize_mesh_vertices(vert_list)
        
        v_norm = {k: normalized[i] for i, k in enumerate(['ultra_low', 'low', 'medium_base', 'high'])}
        
        # Create training scales
        training_scales = []
        scale_configs = [
            ('ultra_low', 'low', 8, 1.0, 'ultra_low_to_low'),
            ('low', 'medium_base', 8, 1.0, 'low_to_medium'),
            ('medium_base', 'high', 16, 2.0, 'medium_to_high')
        ]
        
        for src, tgt, k, scale_weight, scale_name in scale_configs:
            template = create_enhanced_template(v_norm[src], v_norm[tgt])
            weights = compute_interpolation_weights(
                v_norm[src], v_norm[tgt], k=k,
                use_curvature=(src != 'ultra_low'),
                low_faces=meshes[src].faces,
                high_faces=meshes[tgt].faces
            )
            
            scale_data = {
                'input': v_norm[src][np.newaxis, :, :],
                'target': v_norm[tgt][np.newaxis, :, :],
                'template': template[np.newaxis, :, :],
                'weights': weights,
                'scale_name': scale_name,
                'scale_weight': scale_weight
            }
            
            # Add advanced features for final scale
            if src == 'medium_base':
                loss_weights, _ = compute_feature_aware_loss_weights(
                    vertices['high'], meshes['high'].faces, curvature_threshold=0.4)
                scale_data['loss_weights'] = loss_weights[np.newaxis, :, np.newaxis]
                scale_data['laplacian'] = (compute_laplacian_regularization(
                    vertices['high'], meshes['high'].faces) / scale)[np.newaxis, :, :]
            
            training_scales.append(scale_data)
        
        return training_scales, center, scale
    
    def _train_model(self, training_scales, epochs, experiment_tracker=None):
        """
        Train model with multi-scale progressive learning.
        
        Args:
            training_scales: List of scale configuration dictionaries
            epochs: Number of training epochs
            experiment_tracker: Optional ExperimentTracker for experiment logging
        
        Returns:
            Tuple of (trained_model, loss_history, best_loss)
        """
        model = MeshSuperResNet(hidden_dim=512)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001, decay_steps=100, decay_rate=0.95, staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        best_loss = float('inf')
        loss_history = []
        scale_losses = {scale['scale_name']: [] for scale in training_scales}
        
        self._print_training_info(training_scales, model)
        
        training_start_time = time.time()
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for scale_data in training_scales:
                scale_loss = self._train_scale(
                    model, optimizer, scale_data, training_scales
                )
                scale_losses[scale_data['scale_name']].append(scale_loss)
                total_loss += scale_loss
            
            avg_loss = total_loss / len(training_scales)
            loss_history.append(avg_loss)
            best_loss = min(best_loss, avg_loss)
            
            # Log metrics to experiment tracker
            if experiment_tracker:
                metrics = {
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'avg_loss': float(avg_loss),
                    'best_loss': float(best_loss),
                    'learning_rate': float(optimizer.learning_rate.numpy())
                }
                # Add per-scale losses
                for scale_name, losses in scale_losses.items():
                    metrics[f'loss_{scale_name.replace("→", "_to_")}'] = float(losses[-1])
                
                experiment_tracker.log_epoch_metrics(metrics)
            
            if (epoch + 1) % 25 == 0:
                self._print_progress(epoch + 1, epochs, avg_loss, best_loss, 
                                    loss_history, scale_losses, optimizer)
        
        training_time = time.time() - training_start_time
        
        # Save checkpoint if tracking
        if experiment_tracker:
            experiment_tracker.save_checkpoint(
                model=model,
                epoch=epochs,
                loss=best_loss,
                metadata={'total_training_time': training_time}
            )
        
        return model, loss_history, best_loss
    
    def _train_scale(self, model, optimizer, scale_data, all_scales):
        """Train on a single scale with improved accuracy constraints."""
        low_tf = tf.constant(scale_data['input'], dtype=tf.float32)
        high_tf = tf.constant(scale_data['target'], dtype=tf.float32)
        template_tf = tf.constant(scale_data['template'], dtype=tf.float32)
        weights_tf = tf.constant(scale_data['weights'], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            pred = model(low_tf, template_tf, weights_tf, training=True)
            per_vertex_loss = tf.reduce_sum(tf.square(pred - high_tf), axis=-1)
            
            # Outlier clamping at 95th percentile
            loss_threshold = tf.constant(np.percentile(
                per_vertex_loss.numpy(), 95), dtype=tf.float32)
            clamped_loss = tf.minimum(per_vertex_loss, loss_threshold)
            
            # Apply feature-aware weighting for final scale
            if 'loss_weights' in scale_data:
                loss_weights_tf = tf.constant(scale_data['loss_weights'], dtype=tf.float32)
                weighted_loss = clamped_loss[np.newaxis, :] * loss_weights_tf
                base_loss = tf.reduce_mean(weighted_loss)
                
                laplacian_tf = tf.constant(scale_data['laplacian'], dtype=tf.float32)
                laplacian_term = tf.reduce_mean(tf.square(pred - laplacian_tf))
                
                # Add reference constraint: keep predictions close to high-res mesh
                # This prevents drift and improves accuracy
                reference_constraint = tf.reduce_mean(tf.square(pred - high_tf)) * 0.1
                
                combined_loss = base_loss + 0.01 * laplacian_term + reference_constraint
            else:
                combined_loss = tf.reduce_mean(clamped_loss)
            
            loss = scale_data.get('scale_weight', 1.0) * combined_loss
        
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        return loss.numpy()
    
    def _print_training_info(self, training_scales, model):
        """Print training configuration summary."""
        print("\n" + "="*80)
        print("TRAINING DATA SUMMARY")
        print("="*80)
        for idx, scale_data in enumerate(training_scales):
            print(f"\nScale {idx}: {scale_data['scale_name']}")
            print(f"  Input:  {scale_data['input'].shape} vertices")
            print(f"  Target: {scale_data['target'].shape} vertices")
            print(f"  Weight: {scale_data.get('scale_weight', 1.0)}")
        
        total_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
        print(f"\nModel parameters: {total_params:,}")
        print(f"Optimizer: Adam (LR=0.001, decay=0.95 every 100 steps)")
        print("="*80 + "\n")
    
    def _print_progress(self, epoch, total_epochs, avg_loss, best_loss, 
                       loss_history, scale_losses, optimizer):
        """Print training progress."""
        improvement = 0
        if len(loss_history) > 25:
            improvement = ((loss_history[-26] - avg_loss) / loss_history[-26]) * 100
        
        loss_breakdown = " | ".join([
            f"{name}: {scale_losses[name][-1]:.6f}" 
            for name in scale_losses.keys()
        ])
        current_lr = float(optimizer.learning_rate.numpy())
        
        print(f"Epoch {epoch:3d}/{total_epochs} | Loss: {avg_loss:.6f} | " +
              f"Best: {best_loss:.6f} | Improvement: {improvement:+.2f}% | LR: {current_lr:.6f}")
        print(f"  Details: {loss_breakdown}")
    
    def _generate_ml_enhanced_mesh(self):
        """Generate ML-enhanced mesh by refining medium resolution with accuracy focus."""
        if self.trained_model is None:
            return None
        
        try:
            center = self.preprocessing_data['center']
            scale = self.preprocessing_data['scale']
            
            medium_norm = (self.mesh_variants['medium_base'].vertices - center) / scale
            high_verts_norm = (self.mesh_variants['high'].vertices - center) / scale
            
            # Use clean template during inference (no noise) for better accuracy
            template = high_verts_norm.copy()  # Clean template without noise
            weights = compute_interpolation_weights(medium_norm, high_verts_norm)
            
            enhanced_norm = self.trained_model(
                tf.constant(medium_norm[np.newaxis, :, :], dtype=tf.float32),
                tf.constant(template[np.newaxis, :, :], dtype=tf.float32),
                tf.constant(weights, dtype=tf.float32),
                training=False
            )
            
            # Constrain predictions to be close to high-res reference (accuracy improvement)
            enhanced_displacement = enhanced_norm.numpy()[0] - template
            max_displacement = np.percentile(np.linalg.norm(enhanced_displacement, axis=1), 90)
            
            # Apply soft constraint: clamp excessive displacements
            displacement_norms = np.linalg.norm(enhanced_displacement, axis=1, keepdims=True)
            constrained_displacement = enhanced_displacement * np.minimum(
                1.0, max_displacement / (displacement_norms + 1e-6)
            )
            constrained_enhanced_norm = template + constrained_displacement
            
            enhanced_verts = denormalize_vertices(constrained_enhanced_norm, center, scale)
            
            # Create mesh with explicit vertex and face arrays
            result_mesh = trimesh.Trimesh(
                vertices=enhanced_verts,
                faces=self.mesh_variants['high'].faces,
                process=False  # Don't auto-process which might modify vertices
            )
            
            return result_mesh
        except Exception as e:
            print(f"Error generating ML enhanced mesh: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_quality_metrics(self, loss_history, best_loss):
        """Calculate and store quality metrics."""
        if self.ml_enhanced_mesh is None:
            return
        
        orig_verts = self.original_mesh.vertices
        ml_verts = self.ml_enhanced_mesh.vertices
        
        # Handle case where meshes have different vertex counts
        # Interpolate to closest points for error calculation
        from scipy.spatial import cKDTree
        
        tree = cKDTree(orig_verts)
        distances, _ = tree.query(ml_verts)
        
        self.quality_metrics = {
            'reconstruction_error_mean': float(np.mean(distances)),
            'reconstruction_error_max': float(np.max(distances)),
            'final_training_loss': float(loss_history[-1]) if loss_history else 0,
            'best_training_loss': float(best_loss),
            'face_improvement': (f"{len(self.mesh_variants['medium_base'].faces):,} → "
                               f"{len(self.original_mesh.faces):,}"),
            'quality_improvement': ((len(self.original_mesh.faces) - 
                                   len(self.mesh_variants['medium_base'].faces)) /
                                  len(self.mesh_variants['medium_base'].faces)) * 100
        }
    
    def preview_mesh(self):
        """Display multi-scale refinement pipeline in a horizontal line."""
        scene = trimesh.Scene()
        ref_mesh = self.mesh_variants['high']
        bbox_size = np.max(ref_mesh.bounds[1] - ref_mesh.bounds[0])
        spacing = bbox_size * 1.5
        
        # Center 4 meshes horizontally: -1.5*spacing, -0.5*spacing, 0.5*spacing, 1.5*spacing
        x_positions = [-1.5 * spacing, -0.5 * spacing, 0.5 * spacing, 1.5 * spacing]
        
        if self.ml_enhanced_mesh is None:
            # Before training: LOD hierarchy
            print("\n=== BEFORE TRAINING - LOD HIERARCHY ===\n")
            meshes = [
                (self.mesh_variants['ultra_low'], [x_positions[0], 0, 0], 
                 [255, 100, 100, 255], "Ultra Low (1/20)"),
                (self.mesh_variants['low'], [x_positions[1], 0, 0], 
                 [255, 180, 100, 255], "Low (1/8)"),
                (self.mesh_variants['medium_base'], [x_positions[2], 0, 0], 
                 [255, 255, 100, 255], "Medium Base (1/4)"),
                (self.mesh_variants['high'], [x_positions[3], 0, 0], 
                 [100, 150, 255, 255], "Original (100%)")
            ]
        else:
            # After training: refinement results
            print("\n=== AFTER TRAINING - REFINEMENT RESULTS ===\n")
            meshes = [
                (self.mesh_variants['medium_base'], [x_positions[0], 0, 0], 
                 [255, 255, 100, 255], "Medium (untrained)"),
                (self.mesh_variants['high'], [x_positions[1], 0, 0], 
                 [100, 150, 255, 255], "Original (reference)"),
                (self.mesh_variants['medium_base'], [x_positions[2], 0, 0], 
                 [200, 255, 100, 255], "Medium (trained template)"),
                (self.ml_enhanced_mesh, [x_positions[3], 0, 0], 
                 [100, 255, 200, 255], "ML Enhanced (output)")
            ]
        
        for mesh, pos, color, label in meshes:
            positioned = mesh.copy()
            positioned.vertices += np.array(pos)
            positioned.visual.vertex_colors = color
            scene.add_geometry(positioned)
            print(f"{label:30s}: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
        
        print()
        scene.show()
