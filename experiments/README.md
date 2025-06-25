# Experiments

This directory contains scripts for generating three key comparison plots for the TabSMC paper.

## 1. SGD Long vs SMC Comparison

**Plot generated:** `figures/sgd_long_vs_smc_rejuv10_comparison.png`

**Files:**
- `run_sgd_baseline_long.py` - Runs SGD baseline for 300 steps
- `run_smc_rejuv_10.py` - Runs SMC with 10 rejuvenation steps
- `plot_sgd_long_vs_smc_rejuv10.py` - Creates the comparison plot
- `run_sgd_long_vs_smc_comparison.sh` - Shell script to run the full experiment

**To run:**
```bash
bash experiments/run_sgd_long_vs_smc_comparison.sh
```

## 2. Multi-particle SMC Gibbs Steps Comparison

**Plot generated:** `figures/multi_particle_smc_gibbs_steps_10runs_no_t0.png`

**Files:**
- `generate_shared_test_set.py` - Generates shared test set for fair comparison
- `run_smc_no_rejuvenation_gibbs.py` - SMC without rejuvenation
- `run_smc_rejuv_1_gibbs.py` - SMC with 1 rejuvenation step
- `run_smc_rejuv_10_gibbs.py` - SMC with 10 rejuvenation steps
- `run_smc_then_rejuv_gibbs.py` - SMC then rejuvenation-only
- `plot_gibbs_steps_results.py` - Creates the comparison plot
- `run_all_gibbs_steps_experiments.sh` - Shell script to run all experiments

**To run:**
```bash
bash experiments/run_all_gibbs_steps_experiments.sh
```

## 3. PUMS Multi-run Benchmark

**Plot generated:** `figures/pums_multirun_benchmark.png`

**Files:**
- `run_pums_final_comparison.py` - Runs multiple trials of SignSGD vs SMC on PUMS data

**To run:**
```bash
uv run python experiments/run_pums_final_comparison.py
```

This script performs 5 runs of each method, discards the first run (JIT warmup), and plots the remaining 4 runs showing wall-clock time vs test log-likelihood.

## 4. Full PUMS Dataset Training

**Script generated:** Various result and checkpoint files

**Files:**
- `train_smc_full_pums.py` - Train SMC on full PUMS dataset with memory-aware settings
- `plot_smc_pums_results.py` - Plot and analyze full PUMS training results

**To run:**
```bash
# Train with default settings (1000 particles, 50 time steps)
uv run python experiments/train_smc_full_pums.py

# Train with custom settings
uv run python experiments/train_smc_full_pums.py \
    --particles 1500 \
    --clusters 15 \
    --time-steps 100 \
    --batch-size 200 \
    --rejuvenation 10 \
    --output-dir results/pums_large

# Plot results
uv run python experiments/plot_smc_pums_results.py single \
    results/smc_pums_final_P1000_C10_T50_B200.pkl \
    --output figures/smc_pums_training.png \
    --summary
```

**Memory considerations:**
- Maximum recommended particles: ~2000 (based on GPU memory tests)
- Each particle uses ~4-5 MB GPU memory
- The script automatically saves checkpoints and warns about high particle counts

## Data Dependencies

All experiments depend on:
- Synthetic data generation: `generate_synthetic_data.py`
- Optimal likelihood computation: `compute_optimal_likelihood.py`
- PUMS data files (for PUMS benchmark): `data/pums10000.npy`, `data/pums_test1000.npy`, `data/pums_mask.npy`

## GPU Memory Testing

**Additional utilities:**
- `test_gpu_memory_limits.py` - Test GPU memory limits with JAX preallocation enabled
- `test_actual_gpu_memory.py` - Test actual memory usage with preallocation disabled  
- `diagnose_gpu_memory.py` - Diagnose GPU memory allocation patterns