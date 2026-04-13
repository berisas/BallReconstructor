"""
Experiment tracking and logging for research benchmarking.
Handles: configs, results, metrics, visualizations.
"""
import json
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import csv


class ExperimentTracker:
    """Track experiments with configs, metrics, and results."""
    
    def __init__(self, experiment_name: str, base_log_dir: str = "logs"):
        self.base_log_dir = base_log_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            base_log_dir, 
            f"{experiment_name}_{self.timestamp}"
        )
        
        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "visualizations"), exist_ok=True)
        
        self.metrics_log = []
        self.config = {}
        self.results = {}
    
    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration."""
        self.config = config
        config_path = os.path.join(self.experiment_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Config saved to {config_path}")
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log per-epoch metrics."""
        entry = {"epoch": epoch, **metrics}
        self.metrics_log.append(entry)
        
        # Save incrementally
        metrics_path = os.path.join(self.experiment_dir, "metrics.csv")
        if epoch == 0:  # First epoch - create with headers
            with open(metrics_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=entry.keys())
                writer.writeheader()
                writer.writerow(entry)
        else:  # Append
            with open(metrics_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=entry.keys())
                writer.writerow(entry)
    
    def save_final_results(self, results: Dict[str, Any]):
        """Save final experiment results."""
        self.results = results
        results_path = os.path.join(self.experiment_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
    
    def save_checkpoint(self, model, filename: str = "model.h5"):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.experiment_dir, "checkpoints", filename)
        model.save_weights(checkpoint_path)
        return checkpoint_path
    
    def save_visualization(self, fig, filename: str):
        """Save matplotlib figure."""
        viz_path = os.path.join(self.experiment_dir, "visualizations", filename)
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        return viz_path
    
    def save_mesh(self, mesh, filename: str):
        """Save mesh file."""
        results_dir = os.path.join(self.experiment_dir, "results")
        mesh_path = os.path.join(results_dir, filename)
        mesh.export(mesh_path)
        return mesh_path
    
    def get_experiment_path(self) -> str:
        """Get root experiment directory."""
        return self.experiment_dir
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        return {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "experiment_dir": self.experiment_dir,
            "config": self.config,
            "results": self.results,
            "metrics_entries": len(self.metrics_log)
        }
    
    def print_summary(self):
        """Print experiment summary."""
        summary = self.get_summary()
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(f"Name: {summary['experiment_name']}")
        print(f"Time: {summary['timestamp']}")
        print(f"Path: {summary['experiment_dir']}")
        print(f"Config keys: {list(summary['config'].keys())}")
        print(f"Results keys: {list(summary['results'].keys())}")
        print(f"Metrics logged: {summary['metrics_entries']}")
        print("="*80 + "\n")


class BenchmarkComparison:
    """Compare multiple experiments."""
    
    def __init__(self, experiments: Dict[str, str]):
        """
        Args:
            experiments: Dict mapping names to experiment directories
        """
        self.experiments = experiments
        self.data = {}
        self._load_experiments()
    
    def _load_experiments(self):
        """Load all experiment data."""
        for name, exp_dir in self.experiments.items():
            self.data[name] = {
                "config": self._load_yaml(os.path.join(exp_dir, "config.yaml")),
                "results": self._load_json(os.path.join(exp_dir, "results.json")),
                "metrics": self._load_csv(os.path.join(exp_dir, "metrics.csv"))
            }
    
    @staticmethod
    def _load_yaml(path):
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def _load_json(path):
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)
    
    @staticmethod
    def _load_csv(path):
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return list(csv.DictReader(f))
    
    def compare_results(self) -> Dict[str, Dict]:
        """Compare key results across experiments."""
        comparison = {}
        for name, data in self.data.items():
            comparison[name] = data["results"]
        return comparison
    
    def compare_metrics(self, metric_key: str) -> Dict[str, List[float]]:
        """Compare specific metric across experiments."""
        comparison = {}
        for name, data in self.data.items():
            values = []
            for entry in data["metrics"]:
                if metric_key in entry:
                    try:
                        values.append(float(entry[metric_key]))
                    except (ValueError, TypeError):
                        continue
            comparison[name] = values
        return comparison
    
    def summary_table(self) -> Dict[str, Dict]:
        """Generate summary table for all experiments."""
        table = {}
        for name, data in self.data.items():
            table[name] = {
                "final_loss": data["results"].get("final_loss", "N/A"),
                "epochs": data["config"].get("training", {}).get("epochs", "N/A"),
                "model": data["config"].get("model", {}).get("architecture", "N/A"),
                "training_time": data["results"].get("training_time_seconds", "N/A"),
            }
        return table
