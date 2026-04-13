"""
Evaluation metrics for mesh quality assessment and benchmarking.
Implements standard metrics for comparing meshes and refinement quality.
"""
import numpy as np
import trimesh
from typing import Tuple, Dict, Any
from scipy.spatial import cKDTree


def chamfer_distance(pred_vertices: np.ndarray, target_vertices: np.ndarray) -> float:
    """
    Compute bidirectional Chamfer distance between two point clouds.
    
    Measures surface quality by computing average distance from each point
    in one mesh to nearest point in other mesh.
    
    Args:
        pred_vertices: [N, 3] predicted vertex coordinates
        target_vertices: [M, 3] target/reference vertex coordinates
    
    Returns:
        Average bidirectional Chamfer distance
    """
    # Build KD-trees for efficient nearest neighbor search
    pred_tree = cKDTree(pred_vertices)
    target_tree = cKDTree(target_vertices)
    
    # Distance from predicted to target
    pred_to_target, _ = target_tree.query(pred_vertices)
    
    # Distance from target to predicted
    target_to_pred, _ = pred_tree.query(target_vertices)
    
    # Bidirectional average
    chamfer = (np.mean(pred_to_target) + np.mean(target_to_pred)) / 2.0
    
    return float(chamfer)


def hausdorff_distance(pred_vertices: np.ndarray, target_vertices: np.ndarray) -> float:
    """
    Compute Hausdorff distance (max of maximum nearest-neighbor distances).
    
    Captures worst-case surface deviation between two meshes.
    
    Args:
        pred_vertices: [N, 3] predicted vertices
        target_vertices: [M, 3] target vertices
    
    Returns:
        Hausdorff distance (max bidirectional nearest neighbor distance)
    """
    pred_tree = cKDTree(pred_vertices)
    target_tree = cKDTree(target_vertices)
    
    # Max distance from predicted to target
    pred_to_target, _ = target_tree.query(pred_vertices)
    
    # Max distance from target to predicted
    target_to_pred, _ = pred_tree.query(target_vertices)
    
    hausdorff = max(np.max(pred_to_target), np.max(target_to_pred))
    
    return float(hausdorff)


def mesh_laplacian_smoothness(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Compute Laplacian-based surface smoothness metric.
    
    Measures surface regularity by computing average deviation of each vertex
    from the average of its neighbors. Lower values = smoother surface.
    
    Args:
        vertices: [N, 3] vertex coordinates
        faces: [F, 3] face indices
    
    Returns:
        Average Laplacian residual (smoothness metric)
    """
    # Build adjacency information
    adjacency = {}
    for face in faces:
        for i in range(3):
            v = face[i]
            neighbor = face[(i + 1) % 3]
            if v not in adjacency:
                adjacency[v] = set()
            adjacency[v].add(neighbor)
    
    # Compute Laplacian residuals
    residuals = []
    for v_idx, neighbors in adjacency.items():
        if len(neighbors) > 0:
            neighbor_mean = np.mean(vertices[list(neighbors)], axis=0)
            residual = np.linalg.norm(vertices[v_idx] - neighbor_mean)
            residuals.append(residual)
    
    return float(np.mean(residuals)) if residuals else 0.0


def vertex_normal_consistency(pred_vertices: np.ndarray, pred_faces: np.ndarray,
                             target_vertices: np.ndarray, target_faces: np.ndarray) -> float:
    """
    Measure normal consistency between predicted and target surfaces.
    
    Computes average angle between surface normals at corresponding locations.
    Lower values = better normal alignment.
    
    Args:
        pred_vertices, pred_faces: Predicted mesh
        target_vertices, target_faces: Target mesh
    
    Returns:
        Average normal deviation angle in degrees
    """
    # Build meshes
    pred_mesh = trimesh.Trimesh(vertices=pred_vertices, faces=pred_faces)
    target_mesh = trimesh.Trimesh(vertices=target_vertices, faces=target_faces)
    
    # Sample points on predicted surface and get normals
    sample_pts, face_indices = trimesh.sample.sample_surface(pred_mesh, count=1000)
    pred_normals = pred_mesh.face_normals[face_indices]
    
    # Find closest points on target mesh
    target_tree = cKDTree(target_vertices)
    _, closest_faces = target_tree.query(sample_pts)
    target_normals = target_mesh.face_normals[closest_faces]
    
    # Compute angle between normals
    dots = np.sum(pred_normals * target_normals, axis=1)
    dots = np.clip(dots, -1.0, 1.0)  # Numerical stability
    angles = np.arccos(dots) * 180.0 / np.pi
    
    return float(np.mean(angles))


def compression_quality_ratio(low_poly_count: int, high_poly_count: int) -> float:
    """
    Compute compression ratio quality metric.
    
    Measures how much detail is preserved relative to mesh complexity increase.
    
    Args:
        low_poly_count: Number of vertices in low-poly mesh
        high_poly_count: Number of vertices in high-poly mesh
    
    Returns:
        Compression ratio (high_poly / low_poly)
    """
    if low_poly_count == 0:
        return 0.0
    
    return float(high_poly_count / low_poly_count)


def execution_efficiency(training_time: float, inference_time: float, 
                        refinement_ratio: float) -> Dict[str, float]:
    """
    Compute execution efficiency metrics comparing training vs inference cost.
    
    Args:
        training_time: Total training time in seconds
        inference_time: Inference time per mesh in seconds
        refinement_ratio: Refinement complexity ratio (vertices increased)
    
    Returns:
        Dict with efficiency metrics
    """
    return {
        'training_time_seconds': training_time,
        'inference_time_seconds': inference_time,
        'training_to_inference_ratio': training_time / inference_time if inference_time > 0 else 0.0,
        'efficiency_score': (1.0 / inference_time) * refinement_ratio if inference_time > 0 else 0.0
    }


def reconstruction_error_stats(pred_vertices: np.ndarray, target_vertices: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive reconstruction error statistics.
    
    Args:
        pred_vertices: [N, 3] predicted vertices
        target_vertices: [M, 3] target vertices
    
    Returns:
        Dictionary with error statistics
    """
    target_tree = cKDTree(target_vertices)
    distances, _ = target_tree.query(pred_vertices)
    
    return {
        'mean_error': float(np.mean(distances)),
        'median_error': float(np.median(distances)),
        'std_error': float(np.std(distances)),
        'min_error': float(np.min(distances)),
        'max_error': float(np.max(distances)),
        'p25_error': float(np.percentile(distances, 25)),
        'p75_error': float(np.percentile(distances, 75)),
        'p95_error': float(np.percentile(distances, 95)),
    }


class MeshQualityEvaluator:
    """Comprehensive mesh quality evaluation framework."""
    
    def __init__(self, pred_mesh: np.ndarray, target_mesh: np.ndarray,
                 pred_faces: np.ndarray = None, target_faces: np.ndarray = None):
        """
        Initialize evaluator with predicted and target meshes.
        
        Args:
            pred_mesh: Predicted mesh vertices [N, 3]
            target_mesh: Target mesh vertices [M, 3]
            pred_faces: Predicted mesh faces [F, 3]
            target_faces: Target mesh faces [F, 3]
        """
        self.pred_vertices = pred_mesh
        self.target_vertices = target_mesh
        self.pred_faces = pred_faces
        self.target_faces = target_faces
        self.metrics = {}
    
    def evaluate_all(self) -> Dict[str, Any]:
        """Compute all available metrics."""
        self.metrics['chamfer_distance'] = chamfer_distance(
            self.pred_vertices, self.target_vertices
        )
        self.metrics['hausdorff_distance'] = hausdorff_distance(
            self.pred_vertices, self.target_vertices
        )
        
        self.metrics['reconstruction_errors'] = reconstruction_error_stats(
            self.pred_vertices, self.target_vertices
        )
        
        if self.pred_faces is not None:
            self.metrics['pred_smoothness'] = mesh_laplacian_smoothness(
                self.pred_vertices, self.pred_faces
            )
        
        if self.target_faces is not None:
            self.metrics['target_smoothness'] = mesh_laplacian_smoothness(
                self.target_vertices, self.target_faces
            )
        
        if self.pred_faces is not None and self.target_faces is not None:
            self.metrics['normal_deviation'] = vertex_normal_consistency(
                self.pred_vertices, self.pred_faces,
                self.target_vertices, self.target_faces
            )
        
        return self.metrics
    
    def get_report(self) -> str:
        """Generate human-readable quality report."""
        report = "\n" + "="*80 + "\n"
        report += "MESH QUALITY EVALUATION REPORT\n"
        report += "="*80 + "\n\n"
        
        if not self.metrics:
            self.evaluate_all()
        
        # Summary metrics
        report += "DISTANCE METRICS:\n"
        report += f"  Chamfer Distance: {self.metrics['chamfer_distance']:.6f}\n"
        report += f"  Hausdorff Distance: {self.metrics['hausdorff_distance']:.6f}\n\n"
        
        # Error statistics
        if 'reconstruction_errors' in self.metrics:
            errors = self.metrics['reconstruction_errors']
            report += "RECONSTRUCTION ERROR STATISTICS:\n"
            report += f"  Mean: {errors['mean_error']:.6f}\n"
            report += f"  Median: {errors['median_error']:.6f}\n"
            report += f"  Std Dev: {errors['std_error']:.6f}\n"
            report += f"  Min: {errors['min_error']:.6f}\n"
            report += f"  Max: {errors['max_error']:.6f}\n"
            report += f"  95th Percentile: {errors['p95_error']:.6f}\n\n"
        
        # Smoothness metrics
        if 'pred_smoothness' in self.metrics:
            report += "SURFACE SMOOTHNESS:\n"
            report += f"  Predicted: {self.metrics['pred_smoothness']:.6f}\n"
        if 'target_smoothness' in self.metrics:
            report += f"  Target: {self.metrics['target_smoothness']:.6f}\n\n"
        
        # Normal consistency
        if 'normal_deviation' in self.metrics:
            report += f"NORMAL CONSISTENCY: {self.metrics['normal_deviation']:.2f}°\n\n"
        
        report += "="*80 + "\n"
        return report
