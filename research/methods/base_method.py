"""
Base class for mesh refinement methods.
Provides interface for different approaches to compare.
"""
from abc import ABC, abstractmethod
import time
from typing import Dict, Any, Tuple
import numpy as np


class MeshRefinementMethod(ABC):
    """Abstract base class for mesh refinement methods."""
    
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.model = None
        self.training_history = []
        self.training_time_seconds = 0
        self.inference_time_seconds = 0
    
    @abstractmethod
    def train(self, train_data: Dict[str, np.ndarray], config: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the refinement model.
        
        Args:
            train_data: Training data with 'low_vertices', 'high_vertices', etc.
            config: Configuration parameters
        
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def refine(self, low_resolution_mesh) -> np.ndarray:
        """
        Refine low-resolution vertices to high-resolution.
        
        Args:
            low_resolution_mesh: Input mesh (vertices, faces)
        
        Returns:
            High-resolution vertex positions
        """
        pass
    
    def get_training_time(self) -> float:
        """Get total training time in seconds."""
        return self.training_time_seconds
    
    def get_inference_time(self) -> float:
        """Get total inference time in seconds."""
        return self.inference_time_seconds
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get method information for reporting."""
        return {
            "method_name": self.method_name,
            "training_time_seconds": self.training_time_seconds,
            "inference_time_seconds": self.inference_time_seconds,
            "training_history_length": len(self.training_history)
        }


class ProgressiveMultiScaleMethod(MeshRefinementMethod):
    """Progressive multi-scale refinement (current implementation)."""
    
    def __init__(self):
        super().__init__("progressive_multiscale")
        self.lod_system = None
    
    def train(self, lod_system, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Train using progressive multi-scale approach.
        
        Args:
            lod_system: MLLODSystem instance
            config: Configuration (epochs, etc.)
        
        Returns:
            Training metrics
        """
        from training import MLLODSystem
        
        self.lod_system = lod_system
        start_time = time.time()
        
        epochs = config.get("training", {}).get("epochs", 150)
        success = self.lod_system.train_ml_model(epochs=epochs)
        
        self.training_time_seconds = time.time() - start_time
        
        if success:
            self.training_history = self.lod_system.training_loss_history if hasattr(
                self.lod_system, 'training_loss_history'
            ) else []
        
        return {
            "success": success,
            "training_time_seconds": self.training_time_seconds,
            "method": self.method_name
        }
    
    def refine(self, low_resolution_mesh):
        """Refine using trained model."""
        if self.lod_system is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        start_time = time.time()
        refined = self.lod_system.ml_enhanced_mesh
        self.inference_time_seconds = time.time() - start_time
        
        return refined


class DirectSingleStageMethod(MeshRefinementMethod):
    """Direct single-stage refinement (for comparison)."""
    
    def __init__(self):
        super().__init__("direct_single_stage")
        self.model = None
    
    def train(self, train_data: Dict[str, np.ndarray], config: Dict[str, Any]) -> Dict[str, float]:
        """Train single-stage model directly."""
        # Placeholder for future implementation
        print("Direct single-stage method - placeholder")
        return {"success": False, "method": self.method_name}
    
    def refine(self, low_resolution_mesh):
        """Direct refinement."""
        raise NotImplementedError("Direct method not yet implemented")


class MethodFactory:
    """Factory for creating refinement methods."""
    
    _methods = {
        "progressive_multiscale": ProgressiveMultiScaleMethod,
        "direct_single_stage": DirectSingleStageMethod,
    }
    
    @classmethod
    def create(cls, method_name: str) -> MeshRefinementMethod:
        """Create a refinement method by name."""
        if method_name not in cls._methods:
            raise ValueError(f"Unknown method: {method_name}. Available: {list(cls._methods.keys())}")
        return cls._methods[method_name]()
    
    @classmethod
    def register(cls, name: str, method_class):
        """Register a new method."""
        cls._methods[name] = method_class
    
    @classmethod
    def list_methods(cls) -> list:
        """List available methods."""
        return list(cls._methods.keys())
