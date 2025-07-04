#!/usr/bin/env python
"""Run SMC training on multiple datasets."""

import subprocess
import time
from pathlib import Path

# Dataset paths to run (excluding PUMS and PUMD which are already done)
dataset_paths = [
    "data/CTGAN/covertype", 
    "data/CTGAN/kddcup", 
    "data/CTGAN/sydt", 
    "data/lpm/CES",
    "data/lpm/PUMD",
    "data/lpm/PUMS",
]

# Configuration for each run
config = {
    "particles": 1,
    "clusters": 100,
    "batch_size": 1000,
    "rejuvenation": 10,
    "alpha_pi": 1.0,
    "alpha_theta": 1.0,
    "seed": 42,
    "output_dir": "results",
    "save_every": 100,
    "n_time_steps": 500,
}

def run_dataset(dataset_path):
    """Run SMC training for a single dataset."""
    dataset_name = dataset_path.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Running SMC on {dataset_name}")
    print(f"Dataset path: {dataset_path}")
    print(f"{'='*60}\n")
    
    # Build command
    cmd = [
        "uv", "run", "python", "experiments/train_smc_full_pums.py",
        "--dataset", dataset_path,
        "--particles", str(config["particles"]),
        "--clusters", str(config["clusters"]),
        "--batch-size", str(config["batch_size"]),
        "--rejuvenation", str(config["rejuvenation"]),
        "--alpha-pi", str(config["alpha_pi"]),
        "--alpha-theta", str(config["alpha_theta"]),
        "--seed", str(config["seed"]),
        "--output-dir", config["output_dir"],
        "--save-every", str(config["save_every"])
    ]
    
    start_time = time.time()
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(result.stdout)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ {dataset_name} completed in {elapsed_time/60:.2f} minutes")
        
        return True, elapsed_time
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {dataset_name} failed after {elapsed_time/60:.2f} minutes")
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        
        return False, elapsed_time

def main():
    """Run SMC training on all datasets."""
    print("Starting SMC training on multiple datasets")
    print(f"Configuration: {config}")
    
    # Create results directory
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Track results
    results = []
    total_start = time.time()
    
    # Run each dataset
    for dataset_path in dataset_paths:
        success, elapsed_time = run_dataset(dataset_path)
        results.append({
            "dataset": dataset_path,
            "success": success,
            "time_minutes": elapsed_time / 60
        })
    
    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print("\nResults:")
    
    for res in results:
        status = "✓" if res["success"] else "✗"
        print(f"  {status} {res['dataset']}: {res['time_minutes']:.2f} minutes")
    
    # Count successes
    successes = sum(1 for r in results if r["success"])
    print(f"\nCompleted: {successes}/{len(dataset_paths)} datasets")

if __name__ == "__main__":
    main()