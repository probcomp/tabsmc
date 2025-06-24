#!/bin/bash
# Run SGD (300 steps) and SMC with 10 rejuvenation steps for direct comparison

echo "Running SGD (300 steps) vs SMC (10 rejuv steps) comparison..."
echo "============================================================="

# Check if results already exist
if [ -f "data/sgd_baseline_long_results.pkl" ] && [ -f "data/smc_rejuv_10_results.pkl" ]; then
    echo "Result files already exist. Using existing results."
    echo "Delete the .pkl files if you want to re-run experiments."
else
    # Run SGD baseline with 300 steps
    if [ ! -f "data/sgd_baseline_long_results.pkl" ]; then
        echo -e "\n1. Running SGD baseline (300 steps)..."
        uv run python experiments/run_sgd_baseline_long.py
    else
        echo -e "\n1. SGD long results already exist, skipping..."
    fi

    # Run SMC with 10 rejuvenation steps
    if [ ! -f "data/smc_rejuv_10_results.pkl" ]; then
        echo -e "\n2. Running SMC with 10 rejuvenation steps..."
        uv run python experiments/run_smc_rejuv_10.py
    else
        echo -e "\n2. SMC rejuv-10 results already exist, skipping..."
    fi
fi

# Generate comparison plot
echo -e "\n3. Creating comparison plot..."
uv run python experiments/plot_sgd_long_vs_smc_rejuv10.py

echo -e "\nDone! Check figures/sgd_long_vs_smc_rejuv10_comparison.png"