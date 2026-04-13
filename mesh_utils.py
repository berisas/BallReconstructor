"""
Mesh utility functions for operations, curvature computation, and weight matrices.
"""
import numpy as np
from scipy.spatial import cKDTree


def build_adjacency_list(faces, num_vertices):
    """Build vertex adjacency list from faces."""
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


def compute_mesh_curvature(vertices, faces):
    """Compute mean curvature at each vertex using mesh connectivity."""
    num_vertices = len(vertices)
    curvatures = np.zeros(num_vertices, dtype=np.float32)
    adjacency = build_adjacency_list(faces, num_vertices)
    
    for v_idx in range(num_vertices):
        if len(adjacency[v_idx]) < 3:
            curvatures[v_idx] = 0.0
            continue
        
        neighbors = adjacency[v_idx]
        edges = vertices[neighbors] - vertices[v_idx]
        edge_lengths = np.linalg.norm(edges, axis=1, keepdims=True)
        edge_lengths[edge_lengths < 1e-8] = 1e-8
        edges = edges / edge_lengths
        
        angle_variance = 0.0
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                dot = np.clip(np.dot(edges[i], edges[j]), -1.0, 1.0)
                angle_variance += np.arccos(dot) ** 2
        
        curvatures[v_idx] = np.sqrt(angle_variance / max(len(edges), 1)) if len(edges) > 0 else 0.0
    
    max_curv = np.max(curvatures)
    if max_curv > 1e-8:
        curvatures = curvatures / max_curv
    
    return curvatures


def _build_weight_matrix(indices, weights, num_high, num_low):
    """Helper: Convert k-NN indices and weights to sparse weight matrix."""
    batch_size = 1
    weight_matrix = np.zeros((batch_size, num_high, num_low), dtype=np.float32)
    for i in range(num_high):
        for j, idx in enumerate(indices[i]):
            weight_matrix[0, i, idx] = weights[i, j]
    return weight_matrix


def compute_interpolation_weights(low_vertices, high_vertices, k=8, use_curvature=False, 
                                   low_faces=None, high_faces=None):
    """
    Compute interpolation weights using k-NN with optional curvature adaptation.
    
    Args:
        low_vertices: Low-resolution vertex coordinates
        high_vertices: High-resolution vertex coordinates
        k: Number of nearest neighbors
        use_curvature: Whether to apply curvature-adaptive weighting
        low_faces: Low-res faces (required if use_curvature=True)
        high_faces: High-res faces (required if use_curvature=True)
    
    Returns:
        Weight matrix [batch=1, num_high, num_low]
    """
    if len(low_vertices) < k:
        k = len(low_vertices)
    
    tree = cKDTree(low_vertices)
    distances, indices = tree.query(high_vertices, k=k)
    
    sigma = np.mean(distances) * 0.5
    weights = np.exp(-distances**2 / (2 * sigma**2))
    
    if use_curvature and low_faces is not None and high_faces is not None:
        high_curvature = compute_mesh_curvature(high_vertices, high_faces)
        curvature_factor = 1.0 + high_curvature[:, np.newaxis] * 2.0
        weights = weights * curvature_factor
    
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return _build_weight_matrix(indices, weights, len(high_vertices), len(low_vertices))


def compute_laplacian_regularization(vertices, faces):
    """Compute Laplacian smoothing constraint for mesh smoothness."""
    num_vertices = len(vertices)
    adjacency = build_adjacency_list(faces, num_vertices)
    laplacian = np.zeros_like(vertices)
    
    for v_idx in range(num_vertices):
        neighbors = adjacency[v_idx]
        if len(neighbors) > 0:
            neighbor_sum = np.sum(vertices[neighbors], axis=0)
            laplacian[v_idx] = neighbor_sum / len(neighbors) - vertices[v_idx]
    
    return laplacian


def compute_feature_aware_loss_weights(vertices, faces, curvature_threshold=0.5):
    """
    Compute per-vertex loss weights based on curvature.
    Improved version: more balanced weighting for better accuracy.
    High-curvature (features): 70% weight. Low-curvature (smooth): 100% weight.
    """
    curvatures = compute_mesh_curvature(vertices, faces)
    loss_weights = np.where(
        curvatures > curvature_threshold,
        0.7,    # Features: 70% weight (stricter than before) for better accuracy
        1.0     # Smooth: 100% weight (full penalty)
    ).astype(np.float32)
    return loss_weights, curvatures


def create_enhanced_template(low_vertices, high_vertices):
    """Create initial template with controlled Gaussian noise."""
    template = high_vertices.copy()
    noise_scale = np.std(high_vertices, axis=0) * 0.02
    noise = np.random.normal(0, noise_scale, high_vertices.shape).astype(np.float32)
    return (template + noise).astype(np.float32)


def normalize_mesh_vertices(vertices_list, center=None, scale=None):
    """
    Normalize vertices to common coordinate space.
    
    Args:
        vertices_list: List of vertex arrays
        center: Center point (computed if None)
        scale: Scale factor (computed if None)
    
    Returns:
        Tuple of (normalized_vertices_list, center, scale)
    """
    all_vertices = np.vstack(vertices_list)
    if center is None:
        center = np.mean(all_vertices, axis=0)
    if scale is None:
        scale = np.max(np.linalg.norm(all_vertices - center, axis=1)) + 1e-8
    
    normalized = [(v - center) / scale for v in vertices_list]
    return normalized, center, scale


def denormalize_vertices(vertices, center, scale):
    """Convert from normalized to original coordinate space."""
    return vertices * scale + center
