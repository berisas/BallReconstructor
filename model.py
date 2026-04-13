"""
Neural network model for mesh super-resolution.
"""
import tensorflow as tf


class MeshSuperResNet(tf.keras.Model):
    """Multi-scale encoder-decoder network for mesh super-resolution."""
    
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._build_layers()
        self.feature_scale = tf.Variable(1.0, trainable=True)
        self.displacement_scale = tf.Variable(0.05, trainable=True)
    
    def _build_layers(self):
        """Build encoder and decoder layers."""
        # Local encoder: extract per-vertex features
        self.local_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
        ])
        
        # Global encoder: global context from statistics
        self.global_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.hidden_dim // 2, activation='relu')
        ])
        
        # Position decoder: predict displacement vectors
        self.position_decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim * 2, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.hidden_dim // 2, activation='relu'),
            tf.keras.layers.Dense(3, activation='tanh')
        ])
    
    def call(self, low_vertices, query_positions, low_to_query_weights, training=None):
        """
        Forward pass for mesh super-resolution.
        
        Args:
            low_vertices: [batch, num_low, 3] - Low-res vertices
            query_positions: [batch, num_query, 3] - Query positions (template)
            low_to_query_weights: [batch, num_query, num_low] - Interpolation weights
            training: Whether in training mode
        
        Returns:
            refined_positions: [batch, num_query, 3] - Refined vertex positions
        """
        # Extract and scale local features
        local_features = self.local_encoder(low_vertices, training=training)
        local_features = local_features * self.feature_scale
        
        # Compute global context
        global_context = self._compute_global_context(local_features, training)
        
        # Interpolate features to query positions
        interpolated_features = tf.matmul(low_to_query_weights, local_features)
        
        # Broadcast global context to all query positions
        num_query = tf.shape(query_positions)[1]
        global_broadcast = tf.tile(global_context, [1, num_query, 1])
        
        # Combine all features and predict displacement
        combined_input = tf.concat([
            query_positions,
            interpolated_features,
            global_broadcast
        ], axis=-1)
        
        displacement = self.position_decoder(combined_input, training=training)
        displacement = displacement * self.displacement_scale
        
        return query_positions + displacement
    
    def _compute_global_context(self, local_features, training):
        """Compute global context from local features using statistics."""
        global_max = tf.reduce_max(local_features, axis=1, keepdims=True)
        global_mean = tf.reduce_mean(local_features, axis=1, keepdims=True)
        global_std = tf.math.reduce_std(local_features, axis=1, keepdims=True)
        context = tf.concat([global_max, global_mean, global_std], axis=-1)
        return self.global_encoder(context, training=training)
