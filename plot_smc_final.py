import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Load the step 770 checkpoint since step 780 appears corrupted
results_dir = Path("results")
checkpoint_file = results_dir / "smc_pums_checkpoint_step_770.pkl"

if checkpoint_file.exists():
    print(f"Loading checkpoint: {checkpoint_file.name}")
    
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract full history
    log_likelihoods_raw = checkpoint['log_likelihoods']
    
    # Each element is a dict with keys: ['step', 'log_gammas', 'log_weights', 'batch_log_likelihoods']
    steps = []
    log_likelihoods = []
    
    for item in log_likelihoods_raw:
        steps.append(item['step'])
        # batch_log_likelihoods is likely a JAX array
        batch_ll = item['batch_log_likelihoods']
        
        # Convert to numpy if needed and compute mean
        if hasattr(batch_ll, 'shape'):
            # It's an array
            ll_array = np.array(batch_ll)
            ll_value = float(np.mean(ll_array))
        elif isinstance(batch_ll, (list, tuple)):
            ll_value = sum(batch_ll) / len(batch_ll) if len(batch_ll) > 0 else 0
        else:
            ll_value = float(batch_ll)
            
        log_likelihoods.append(ll_value)
    
    # Create plot
    plt.figure(figsize=(12, 7))
    plt.plot(steps, log_likelihoods, 'b-', linewidth=1.5)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Mean Batch Log-Likelihood', fontsize=12)
    plt.title('SMC Training: Log-Likelihood Evolution (770 Steps)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    if len(log_likelihoods) > 0:
        improvement = log_likelihoods[-1] - log_likelihoods[0]
        plt.text(0.02, 0.98, f'Total Steps: {len(log_likelihoods)}\nImprovement: {improvement:.4f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('figures/smc_complete_log_likelihood_770.png', dpi=150)
    
    print(f"Plot saved to figures/smc_complete_log_likelihood_770.png")
    print(f"Total steps: {len(log_likelihoods)}")
    if len(log_likelihoods) > 0:
        print(f"Initial log-likelihood: {log_likelihoods[0]:.4f}")
        print(f"Final log-likelihood: {log_likelihoods[-1]:.4f}")
        print(f"Total improvement: {log_likelihoods[-1] - log_likelihoods[0]:.4f}")
        
    # Also save raw data for further analysis
    data = {'steps': steps, 'log_likelihoods': log_likelihoods}
    with open('figures/smc_log_likelihood_data_770.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("Raw data saved to figures/smc_log_likelihood_data_770.pkl")
else:
    print(f"Checkpoint file not found: {checkpoint_file}")