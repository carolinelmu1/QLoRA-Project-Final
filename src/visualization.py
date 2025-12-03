"""
Visualization utilities for QLoRA diagnostic analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_memory_comparison(results_df, save_path=None):
    """
    Create bar chart comparing memory usage across configurations
    
    Args:
        results_df: DataFrame with columns [experiment_name, peak_memory_mb]
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by quantization type
    results_df['quantization_type'] = results_df['experiment_name'].apply(
        lambda x: 'LoRA (16-bit)' if '16bit' in x else 'QLoRA (4-bit)'
    )
    results_df['rank'] = results_df['experiment_name'].apply(
        lambda x: int(x.split('_r')[1])
    )
    
    # Sort by rank
    results_df = results_df.sort_values('rank')
    
    # Plot grouped bar chart
    x = np.arange(len(results_df['rank'].unique()))
    width = 0.35
    
    lora_data = results_df[results_df['quantization_type'] == 'LoRA (16-bit)']
    qlora_data = results_df[results_df['quantization_type'] == 'QLoRA (4-bit)']
    
    ax.bar(x - width/2, lora_data['peak_memory_mb'], width, 
           label='LoRA (16-bit)', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, qlora_data['peak_memory_mb'], width,
           label='QLoRA (4-bit)', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('LoRA Rank (r)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak GPU Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Memory Usage Comparison: LoRA vs QLoRA', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'r={r}' for r in sorted(results_df['rank'].unique())])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_rank_threshold_analysis(results_df, metric='mean_cosine_similarity', save_path=None):
    """
    Plot performance vs rank for LoRA and QLoRA
    
    Args:
        results_df: DataFrame with performance metrics
        metric: Metric to plot (e.g., 'mean_cosine_similarity', 'mean_token_match')
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Parse data
    results_df['quantization_type'] = results_df['experiment_name'].apply(
        lambda x: 'LoRA (16-bit)' if '16bit' in x else 'QLoRA (4-bit)'
    )
    results_df['rank'] = results_df['experiment_name'].apply(
        lambda x: int(x.split('_r')[1])
    )
    
    # Plot lines
    for quant_type in results_df['quantization_type'].unique():
        data = results_df[results_df['quantization_type'] == quant_type].sort_values('rank')
        
        color = '#3498db' if 'LoRA' in quant_type and '16-bit' in quant_type else '#e74c3c'
        marker = 'o' if 'LoRA' in quant_type and '16-bit' in quant_type else 's'
        
        ax.plot(data['rank'], data[metric], marker=marker, markersize=8,
                linewidth=2.5, label=quant_type, color=color, alpha=0.8)
    
    # Add threshold line if cosine similarity
    if 'cosine' in metric.lower():
        ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1.5, 
                   label='Threshold (0.95)', alpha=0.6)
    
    ax.set_xlabel('LoRA Rank (r)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'Rank Threshold Analysis: {metric.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_layer_sensitivity_heatmap(sensitivity_df, save_path=None):
    """
    Create heatmap showing layer sensitivity to quantization
    
    Args:
        sensitivity_df: DataFrame with columns [layer, configuration, metric]
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Pivot for heatmap
    pivot_data = sensitivity_df.pivot(
        index='layer',
        columns='configuration',
        values='performance_drop_percent'
    )
    
    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Performance Drop (%)'},
        ax=ax,
        vmin=0,
        vmax=pivot_data.max().max()
    )
    
    ax.set_title('Layer Sensitivity Analysis: Performance Drop by Layer', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_weight_similarity_matrix(similarity_dict, save_path=None):
    """
    Create heatmap of weight similarity between LoRA and QLoRA
    
    Args:
        similarity_dict: Dictionary of {layer_name: cosine_similarity}
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert to matrix format (simplified for visualization)
    layers = list(similarity_dict.keys())
    similarities = list(similarity_dict.values())
    
    # Create a single-row heatmap
    data = np.array(similarities).reshape(1, -1)
    
    sns.heatmap(
        data,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn',
        xticklabels=layers,
        yticklabels=['LoRA vs QLoRA'],
        vmin=0.9,
        vmax=1.0,
        cbar_kws={'label': 'Cosine Similarity'},
        ax=ax
    )
    
    ax.set_title('Weight Similarity Matrix: LoRA vs QLoRA Adapters',
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_training_curves(train_logs_dict, save_path=None):
    """
    Plot training loss curves for multiple experiments
    
    Args:
        train_logs_dict: Dict of {experiment_name: training_logs}
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for i, (name, logs) in enumerate(train_logs_dict.items()):
        steps = logs.get('steps', [])
        losses = logs.get('losses', [])
        
        ax.plot(steps, losses, label=name, color=colors[i % len(colors)],
                linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


def create_results_table(results_list, save_path=None):
    """
    Create formatted results table
    
    Args:
        results_list: List of result dictionaries
        save_path: Path to save CSV
    
    Returns:
        DataFrame
    """
    df = pd.DataFrame(results_list)
    
    # Select and order columns
    columns_order = [
        'experiment_name',
        'quantization',
        'rank',
        'peak_memory_mb',
        'training_time_seconds',
        'time_per_step',
        'training_loss',
        'mean_token_match',
        'mean_cosine_similarity',
    ]
    
    # Keep only available columns
    available_cols = [col for col in columns_order if col in df.columns]
    df_ordered = df[available_cols]
    
    # Round numeric columns
    numeric_cols = df_ordered.select_dtypes(include=[np.number]).columns
    df_ordered[numeric_cols] = df_ordered[numeric_cols].round(4)
    
    if save_path:
        df_ordered.to_csv(save_path, index=False)
        print(f"✓ Saved table: {save_path}")
    
    return df_ordered


def print_diagnostic_summary(results_dict):
    """
    Print formatted diagnostic summary
    
    Args:
        results_dict: Dictionary of diagnostic results
    """
    print("\n" + "="*70)
    print(" "*20 + "DIAGNOSTIC SUMMARY")
    print("="*70)
    
    print("\n📊 MEMORY EFFICIENCY")
    print("-" * 70)
    lora_mem = results_dict.get('lora_memory_mb', 0)
    qlora_mem = results_dict.get('qlora_memory_mb', 0)
    reduction = ((lora_mem - qlora_mem) / lora_mem * 100) if lora_mem > 0 else 0
    
    print(f"  LoRA (16-bit):  {lora_mem:>8.2f} MB")
    print(f"  QLoRA (4-bit):  {qlora_mem:>8.2f} MB")
    print(f"  Reduction:      {reduction:>8.2f}%")
    
    print("\n🎯 PERFORMANCE PRESERVATION")
    print("-" * 70)
    cos_sim = results_dict.get('mean_cosine_similarity', 0)
    token_match = results_dict.get('mean_token_match', 0)
    
    print(f"  Cosine Similarity:  {cos_sim:>6.4f}")
    print(f"  Token Match Rate:   {token_match:>6.4f}")
    
    threshold_met = "✓ YES" if cos_sim >= 0.95 else "✗ NO"
    print(f"  Threshold (≥0.95):  {threshold_met}")
    
    print("\n⚡ TRAINING EFFICIENCY")
    print("-" * 70)
    lora_time = results_dict.get('lora_time_per_step', 0)
    qlora_time = results_dict.get('qlora_time_per_step', 0)
    speedup = (lora_time / qlora_time) if qlora_time > 0 else 1.0
    
    print(f"  LoRA time/step:   {lora_time:>7.3f}s")
    print(f"  QLoRA time/step:  {qlora_time:>7.3f}s")
    print(f"  Speedup:          {speedup:>7.2f}x")
    
    print("\n" + "="*70 + "\n")