#!/usr/bin/env python3
"""
Plot JS distance vs time, colored by model and faceted by dataset.
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_js_vs_time_plot():
    """Create JS vs time plot with model colors and dataset facets."""
    
    # Load results
    df = pl.read_parquet("js_distances_all_results.parquet")
    
    print(f"Loaded {len(df)} results")
    print(f"Datasets: {sorted(df['dataset'].unique().to_list())}")
    print(f"Models: {sorted(df['model'].unique().to_list())}")
    
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
    
    # Define colors for models
    model_colors = {
        'ARF': '#1f77b4',        # Blue
        'Diffusion': '#ff7f0e',  # Orange  
        'TVAE': '#2ca02c'        # Green
    }
    
    # Plot each dataset
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        # Filter data for this dataset
        dataset_data = df_pandas[df_pandas['dataset'] == dataset]
        
        # Plot each model
        for model in models:
            model_data = dataset_data[dataset_data['model'] == model]
            
            if len(model_data) > 0:
                # Sort by time for proper line plotting
                model_data = model_data.sort_values('time')
                
                # Plot individual points
                ax.scatter(model_data['time'], model_data['js_distance'], 
                          color=model_colors[model], alpha=0.6, s=30, label=model)
                
                # Add trend line (moving average or fitted line)
                if len(model_data) > 3:
                    # Group by time and take mean (in case of multiple replicates)
                    time_groups = model_data.groupby('time')['js_distance'].agg(['mean', 'std']).reset_index()
                    
                    ax.plot(time_groups['time'], time_groups['mean'], 
                           color=model_colors[model], linewidth=2, alpha=0.8)
                    
                    # Add error bars if we have std
                    if not time_groups['std'].isna().all():
                        ax.fill_between(time_groups['time'], 
                                      time_groups['mean'] - time_groups['std'],
                                      time_groups['mean'] + time_groups['std'],
                                      color=model_colors[model], alpha=0.2)
        
        # Formatting for each subplot
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('JS Distance', fontsize=12)
        ax.set_title(f'{dataset}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max * 1.05)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)
    
    # Overall title
    fig.suptitle('Jensen-Shannon Distance vs Time by Model and Dataset', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save plot
    output_file = 'js_distance_vs_time_by_model_and_dataset.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    
    # Show plot
    plt.show()
    
    return fig

def create_summary_statistics():
    """Print summary statistics for the plot."""
    df = pl.read_parquet("js_distances_all_results.parquet")
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"Total combinations: {len(df)}")
    print(f"JS distance range: {df['js_distance'].min():.4f} - {df['js_distance'].max():.4f}")
    print(f"Mean JS distance: {df['js_distance'].mean():.4f}")
    print(f"Median JS distance: {df['js_distance'].median():.4f}")
    
    # Statistics by dataset and model
    print(f"\nDetailed Statistics by Dataset and Model:")
    summary = df.group_by(['dataset', 'model']).agg([
        pl.col('js_distance').count().alias('n_points'),
        pl.col('js_distance').mean().alias('mean_js'),
        pl.col('js_distance').median().alias('median_js'),
        pl.col('js_distance').min().alias('min_js'),
        pl.col('js_distance').max().alias('max_js'),
        pl.col('time').min().alias('min_time'),
        pl.col('time').max().alias('max_time')
    ]).sort(['dataset', 'model'])
    
    for row in summary.iter_rows(named=True):
        print(f"\n{row['dataset']} - {row['model']}:")
        print(f"  Points: {row['n_points']}")
        print(f"  Time range: {row['min_time']:.1f} - {row['max_time']:.1f}")
        print(f"  JS range: {row['min_js']:.4f} - {row['max_js']:.4f}")
        print(f"  JS mean/median: {row['mean_js']:.4f} / {row['median_js']:.4f}")
    
    # Time evolution analysis
    print(f"\nTime Evolution Analysis:")
    print("Looking at correlation between time and JS distance by dataset/model...")
    
    correlations = df.group_by(['dataset', 'model']).agg([
        pl.corr('time', 'js_distance').alias('time_js_correlation')
    ]).sort(['dataset', 'model'])
    
    for row in correlations.iter_rows(named=True):
        corr = row['time_js_correlation']
        direction = "improving" if corr < -0.1 else "worsening" if corr > 0.1 else "stable"
        print(f"  {row['dataset']} - {row['model']}: r={corr:.3f} ({direction})")

def main():
    """Main function to create plot and show statistics."""
    print("Creating JS distance vs time plot...")
    
    # Create the plot
    fig = create_js_vs_time_plot()
    
    # Show summary statistics
    create_summary_statistics()
    
    print(f"\nPlot creation completed!")

if __name__ == "__main__":
    main()