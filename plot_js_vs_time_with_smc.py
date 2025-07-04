#!/usr/bin/env python3
"""
Plot JS distance vs time, colored by model and faceted by dataset.
Includes both baseline models (ARF, Diffusion, TVAE) and SMC results.
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_and_combine_results():
    """Load and combine baseline and SMC results."""
    
    # Load baseline results
    baseline_file = "js_distances_all_results.parquet"
    if not Path(baseline_file).exists():
        raise FileNotFoundError(f"Baseline results not found: {baseline_file}")
    
    df_baseline = pl.read_parquet(baseline_file)
    print(f"Loaded {len(df_baseline)} baseline results")
    
    # Load SMC clean results
    smc_file = "js_distances_smc_clean_results.parquet"
    if not Path(smc_file).exists():
        raise FileNotFoundError(f"SMC clean results not found: {smc_file}")
    
    df_smc = pl.read_parquet(smc_file)
    print(f"Loaded {len(df_smc)} clean SMC results")
    
    # Standardize dataset names for SMC to match baseline format
    # Clean SMC dataset names are like "SMC_Clean/ces", baseline are like "CTGAN/covertype"
    dataset_mapping = {
        'SMC_Clean/ces': 'lpm/CES',
        'SMC_Clean/pumd': 'lpm/PUMD', 
        'SMC_Clean/pums': 'lpm/PUMS',
        'SMC_Clean/covertype': 'CTGAN/covertype',
        'SMC_Clean/kddcup': 'CTGAN/kddcup',
        'SMC_Clean/sydt': 'CTGAN/sydt'
    }
    
    # Map SMC dataset names to match baseline format
    df_smc = df_smc.with_columns([
        pl.col('dataset').replace(dataset_mapping).alias('dataset'),
        pl.lit('SMC').alias('model')  # Change model name to just 'SMC'
    ])
    
    # Select common columns to ensure compatibility (excluding parameter column)
    common_columns = ['dataset', 'method', 'dataset_name', 'model', 'time', 'replicate', 'n_original_samples', 'n_features', 'js_distance']
    df_baseline = df_baseline.select(common_columns)
    df_smc = df_smc.select(common_columns)
    
    # Ensure consistent data types
    df_smc = df_smc.with_columns([
        pl.col('time').cast(pl.Float64),
        pl.col('replicate').cast(pl.Int64),
        pl.col('n_original_samples').cast(pl.Int64),
        pl.col('n_features').cast(pl.Int64),
        pl.col('js_distance').cast(pl.Float64)
    ])
    
    # Combine the datasets
    df_combined = pl.concat([df_baseline, df_smc], how="vertical")
    
    print(f"Combined dataset: {len(df_combined)} total results")
    print(f"Datasets: {sorted(df_combined['dataset'].unique().to_list())}")
    print(f"Models: {sorted(df_combined['model'].unique().to_list())}")
    
    return df_combined

def create_js_vs_time_plot_with_smc():
    """Create JS vs time plot with model colors and dataset facets, including SMC."""
    
    # Load and combine results
    df = load_and_combine_results()
    
    # Convert to pandas for easier plotting
    df_pandas = df.to_pandas()
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("Set1")
    
    # Create figure with subplots (2 rows, 3 columns for 6 datasets)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Get unique datasets and models
    datasets = sorted(df['dataset'].unique().to_list())
    models = sorted(df['model'].unique().to_list())
    
    # Map internal dataset names to display names
    dataset_display_names = {
        'CTGAN/covertype': 'Covertype',
        'CTGAN/kddcup': 'KDDCup', 
        'CTGAN/sydt': 'SYDT',
        'lpm/CES': 'Cooperative Election Study',
        'lpm/PUMS': 'US Census',
        'lpm/PUMD': 'Consumer Expenditure Survey'
    }
    
    # Define colors for models (including SMC)
    model_colors = {
        'ARF': '#1f77b4',        # Blue
        'Diffusion': '#ff7f0e',  # Orange  
        'TVAE': '#2ca02c',       # Green
        'SMC': '#d62728'         # Red
    }
    
    print(f"Plotting {len(datasets)} datasets: {datasets}")
    print(f"Models: {models}")
    
    # Plot each dataset
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        # Filter data for this dataset
        dataset_data = df_pandas[df_pandas['dataset'] == dataset]
        
        print(f"\nDataset {dataset}: {len(dataset_data)} points")
        
        # Plot each model
        for model in models:
            model_data = dataset_data[dataset_data['model'] == model]
            
            if len(model_data) > 0:
                print(f"  {model}: {len(model_data)} points")
                
                # Sort by time for proper line plotting
                model_data = model_data.sort_values('time')
                
                # Plot individual points
                ax.scatter(model_data['time'], model_data['js_distance'], 
                          color=model_colors[model], alpha=0.7, s=70, label=model)
                
                # Add trend line for models with multiple points
                if len(model_data) > 1:
                    # Group by time and take mean (in case of multiple replicates)
                    time_groups = model_data.groupby('time')['js_distance'].agg(['mean', 'std']).reset_index()
                    
                    ax.plot(time_groups['time'], time_groups['mean'], 
                           color=model_colors[model], linewidth=5, alpha=0.8)
                    
                    # Add error bars if we have std and multiple replicates
                    if not time_groups['std'].isna().all() and len(time_groups) > 1:
                        ax.fill_between(time_groups['time'], 
                                      time_groups['mean'] - time_groups['std'],
                                      time_groups['mean'] + time_groups['std'],
                                      color=model_colors[model], alpha=0.2)
        
        # Formatting for each subplot
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('JS Distance', fontsize=12)
        display_name = dataset_display_names.get(dataset, dataset)
        ax.set_title(f'{display_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max * 1.05)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)
    
    # Overall title
    fig.suptitle('Synthetic data quality vs time by model and dataset', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save plot
    output_file = 'js_distance_vs_time_by_model_and_dataset_with_clean_smc.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    
    return fig

def create_summary_statistics():
    """Print summary statistics including SMC results."""
    df = load_and_combine_results()
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (INCLUDING SMC)")
    print("="*60)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"Total combinations: {len(df)}")
    print(f"JS distance range: {df['js_distance'].min():.4f} - {df['js_distance'].max():.4f}")
    print(f"Mean JS distance: {df['js_distance'].mean():.4f}")
    print(f"Median JS distance: {df['js_distance'].median():.4f}")
    
    # Statistics by model
    print(f"\nStatistics by Model:")
    model_summary = df.group_by('model').agg([
        pl.col('js_distance').count().alias('n_points'),
        pl.col('js_distance').mean().alias('mean_js'),
        pl.col('js_distance').median().alias('median_js'),
        pl.col('js_distance').min().alias('min_js'),
        pl.col('js_distance').max().alias('max_js')
    ]).sort('mean_js')
    
    for row in model_summary.iter_rows(named=True):
        print(f"  {row['model']}: {row['n_points']} points, "
              f"mean = {row['mean_js']:.4f}, median = {row['median_js']:.4f}")
    
    # Statistics by dataset
    print(f"\nStatistics by Dataset:")
    dataset_summary = df.group_by('dataset').agg([
        pl.col('js_distance').count().alias('n_points'),
        pl.col('js_distance').mean().alias('mean_js'),
        pl.col('js_distance').median().alias('median_js')
    ]).sort('dataset')
    
    for row in dataset_summary.iter_rows(named=True):
        print(f"  {row['dataset']}: {row['n_points']} points, "
              f"mean = {row['mean_js']:.4f}, median = {row['median_js']:.4f}")
    
    # SMC specific analysis
    print(f"\nSMC Specific Analysis:")
    smc_data = df.filter(pl.col('model') == 'SMC')
    if len(smc_data) > 0:
        print(f"SMC results: {len(smc_data)} points")
        print(f"SMC mean JS distance: {smc_data['js_distance'].mean():.4f}")
        print(f"SMC median JS distance: {smc_data['js_distance'].median():.4f}")
        
        # Compare SMC performance across timesteps
        smc_timestep_summary = smc_data.group_by('time').agg([
            pl.col('js_distance').count().alias('n_datasets'),
            pl.col('js_distance').mean().alias('mean_js'),
            pl.col('js_distance').median().alias('median_js')
        ]).sort('time')
        
        print(f"SMC performance by timestep:")
        for row in smc_timestep_summary.iter_rows(named=True):
            print(f"  Timestep {row['time']}: {row['n_datasets']} datasets, "
                  f"mean = {row['mean_js']:.4f}, median = {row['median_js']:.4f}")

def main():
    """Main function."""
    try:
        print("Creating JS vs Time plot with SMC results...")
        
        # Create the plot
        fig = create_js_vs_time_plot_with_smc()
        
        # Print summary statistics
        create_summary_statistics()
        
        print(f"\nPlot completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()