#!/bin/bash
# Run all SMC Gibbs steps experiments separately to avoid JAX compilation cross-contamination

echo "Running all SMC Gibbs steps experiments..."
echo "=========================================="

# First, generate the shared test set
echo "0. Generating shared test set..."
uv run python experiments/generate_shared_test_set.py

# Run each method in a separate process
echo -e "\n1. Running SMC without rejuvenation..."
uv run python experiments/run_smc_no_rejuvenation_gibbs.py

echo -e "\n2. Running SMC with 1 rejuvenation step..."
uv run python experiments/run_smc_rejuv_1_gibbs.py

echo -e "\n3. Running SMC with 10 rejuvenation steps..."
uv run python experiments/run_smc_rejuv_10_gibbs.py

echo -e "\n4. Running SMC then rejuvenation-only..."
uv run python experiments/run_smc_then_rejuv_gibbs.py

echo -e "\n5. Creating Gibbs steps plot..."
uv run python experiments/plot_gibbs_steps_results.py

echo -e "\nAll Gibbs steps experiments completed!"