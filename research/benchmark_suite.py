"""
Benchmarking framework for comparing mesh refinement methods and configurations.
Runs multiple experiments and generates comparison reports.
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import yaml


class BenchmarkSuite:
    """Orchestrate and compare multiple refinement experiments."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.benchmarks_dir = os.path.join(project_root, "research", "benchmarks")
        self.datasets_dir = os.path.join(project_root, "research", "datasets")
        self.configs_dir = os.path.join(project_root, "configs")
        self.results = []
        
        # Create directories if they don't exist
        os.makedirs(self.benchmarks_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
    
    def create_config_variants(self, base_name: str = "default",
                              param_grids: Dict[str, List[Any]] = None) -> List[str]:
        """
        Create multiple config files with parameter variations.
        
        Args:
            base_name: Base configuration filename (without .yaml)
            param_grids: Dict mapping parameter paths to list of values
                        Example: {'model.hidden_dim': [256, 512, 1024]}
        
        Returns:
            List of created config file paths
        """
        if param_grids is None:
            param_grids = {
                'model.hidden_dim': [256, 512, 1024],
                'training.epochs': [100, 150, 200],
            }
        
        # Load base config
        base_config_path = os.path.join(self.configs_dir, f"{base_name}.yaml")
        if not os.path.exists(base_config_path):
            print(f"Warning: Base config not found at {base_config_path}")
            return []
        
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        config_files = []
        
        # Generate combinations
        import itertools
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        for combo in itertools.product(*param_values):
            config = dict(base_config)
            variant_name_parts = []
            
            # Apply parameter variations
            for param_path, value in zip(param_names, combo):
                keys = param_path.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
                variant_name_parts.append(f"{keys[-1]}{value}")
            
            # Save variant config
            variant_name = f"{base_name}_" + "_".join(variant_name_parts)
            variant_path = os.path.join(self.configs_dir, f"{variant_name}.yaml")
            
            with open(variant_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            config_files.append(variant_path)
            print(f"  ✓ Created: {variant_name}.yaml")
        
        return config_files
    
    def run_experiment(self, mesh_file: str, experiment_name: str, 
                      config_path: str = None) -> bool:
        """
        Run a single refinement experiment.
        
        Args:
            mesh_file: Path to mesh file
            experiment_name: Name for this experiment run
            config_path: Optional path to config file
        
        Returns:
            True if successful, False otherwise
        """
        print(f"\n▶ Running: {experiment_name}")
        print(f"  Mesh: {mesh_file}")
        if config_path:
            print(f"  Config: {config_path}")
        
        try:
            cmd = [
                sys.executable, "BallReconstructor.py",
                mesh_file,
                "--experiment", experiment_name,
                "--no-gui"
            ]
            if config_path:
                cmd.extend(["--config", config_path])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"  ✓ Completed successfully")
                return True
            else:
                print(f"  ✗ Failed with return code {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")
                return False
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout (>10 minutes)")
            return False
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            return False
    
    def run_benchmark_suite(self, mesh_files: List[str], configs: List[str],
                           max_experiments: int = None) -> Dict[str, Any]:
        """
        Run full benchmark suite across multiple meshes and configurations.
        
        Args:
            mesh_files: List of mesh file paths
            configs: List of config file paths
            max_experiments: Maximum experiments to run (for quick testing)
        
        Returns:
            Dictionary with benchmark results summary
        """
        print("\n" + "="*80)
        print("BENCHMARK SUITE")
        print("="*80)
        print(f"Meshes: {len(mesh_files)}")
        print(f"Configs: {len(configs)}")
        total_experiments = len(mesh_files) * len(configs)
        print(f"Total experiments: {total_experiments}")
        if max_experiments:
            print(f"Will run: {min(total_experiments, max_experiments)}")
        print("="*80)
        
        results = {
            'total_experiments': total_experiments,
            'results': [],
            'successful': 0,
            'failed': 0
        }
        
        experiment_count = 0
        for mesh_file in mesh_files:
            mesh_name = Path(mesh_file).stem
            
            for config in configs:
                if max_experiments and experiment_count >= max_experiments:
                    print(f"\nReached max experiments limit ({max_experiments})")
                    break
                
                config_name = Path(config).stem
                experiment_name = f"{mesh_name}_{config_name}"
                
                success = self.run_experiment(mesh_file, experiment_name, config)
                
                results['results'].append({
                    'experiment': experiment_name,
                    'mesh': mesh_file,
                    'config': config,
                    'success': success
                })
                
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                
                experiment_count += 1
        
        return results
    
    def collect_results(self, experiments_dir: str = None) -> Dict[str, Any]:
        """
        Collect and aggregate results from experiment runs.
        
        Args:
            experiments_dir: Root directory containing experiment subdirectories
        
        Returns:
            Aggregated results dictionary
        """
        if experiments_dir is None:
            experiments_dir = "logs"
        
        if not os.path.exists(experiments_dir):
            print(f"Warning: Experiments directory not found: {experiments_dir}")
            return {}
        
        aggregated = {
            'total_experiments': 0,
            'experiments': [],
            'summary_metrics': {}
        }
        
        # Scan experiment directories
        for exp_dir in os.listdir(experiments_dir):
            exp_path = os.path.join(experiments_dir, exp_dir)
            if not os.path.isdir(exp_path):
                continue
            
            # Look for metrics.csv
            metrics_file = os.path.join(exp_path, "metrics.csv")
            results_file = os.path.join(exp_path, "results.json")
            config_file = os.path.join(exp_path, "config.yaml")
            
            exp_data = {'experiment': exp_dir}
            
            # Load config
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    exp_data['config'] = yaml.safe_load(f)
            
            # Load results
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    exp_data['results'] = json.load(f)
            
            # Parse metrics
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Has header
                        import csv
                        reader = csv.DictReader(lines)
                        exp_data['metrics'] = list(reader)
            
            aggregated['experiments'].append(exp_data)
            aggregated['total_experiments'] += 1
        
        return aggregated
    
    def generate_comparison_report(self, results: Dict[str, Any], 
                                  output_file: str = None) -> str:
        """
        Generate human-readable comparison report.
        
        Args:
            results: Aggregated results dictionary
            output_file: Optional file to save report
        
        Returns:
            Report string
        """
        report = "\n" + "="*80 + "\n"
        report += "BENCHMARK COMPARISON REPORT\n"
        report += "="*80 + "\n\n"
        
        report += f"Total Experiments: {results.get('total_experiments', 0)}\n\n"
        
        # Experiment details
        report += "EXPERIMENTS:\n"
        for exp in results.get('experiments', []):
            report += f"\n  • {exp['experiment']}\n"
            
            if 'results' in exp:
                res = exp['results']
                report += f"    Final Loss: {res.get('best_training_loss', 'N/A')}\n"
                report += f"    Mean Error: {res.get('reconstruction_error_mean', 'N/A')}\n"
                report += f"    Max Error: {res.get('reconstruction_error_max', 'N/A')}\n"
            
            if 'config' in exp and 'training' in exp['config']:
                cfg = exp['config']['training']
                report += f"    Config: {cfg.get('epochs', 'N/A')} epochs, "
                report += f"LR={cfg.get('learning_rate', 'N/A')}\n"
        
        report += "\n" + "="*80 + "\n"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report


def example_benchmark():
    """Example: Run a quick benchmark with parameter variations."""
    suite = BenchmarkSuite(project_root=".")
    
    # Create config variants
    print("\nCreating config variants...")
    param_grids = {
        'model.hidden_dim': [256, 512],  # Quick test
        'training.epochs': [150, 200],
    }
    configs = suite.create_config_variants('default', param_grids)
    
    # Test meshes
    mesh_files = [
        "tennis_ball.obj",
        "tennis_ball_new.obj"
    ]
    
    # Filter to only existing meshes
    existing_meshes = [m for m in mesh_files if os.path.exists(m)]
    
    if not existing_meshes:
        print("Warning: No test meshes found")
        print(f"Please ensure mesh files exist: {mesh_files}")
        return
    
    # Run benchmark (max 4 experiments for quick test)
    results = suite.run_benchmark_suite(existing_meshes, configs, max_experiments=4)
    
    # Generate report
    final_results = suite.collect_results()
    report = suite.generate_comparison_report(
        final_results,
        output_file="research/benchmarks/comparison_report.txt"
    )
    print(report)


if __name__ == "__main__":
    example_benchmark()
