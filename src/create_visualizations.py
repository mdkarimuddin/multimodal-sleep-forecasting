"""
Create comprehensive visualizations for EDA and model performance
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch jobs
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import torch

# Try to import seaborn, make optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

def create_eda_visualizations(df, save_dir=OUTPUT_DIR):
    """
    Create Exploratory Data Analysis visualizations
    """
    print("Creating EDA visualizations...")
    
    # Set style
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Sleep metrics distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sleep_metrics = ['sleep_efficiency', 'total_sleep_time', 'deep_sleep_pct']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, metric in enumerate(sleep_metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if metric in df.columns:
            data = df[metric].dropna()
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color=colors[idx])
            ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Distribution of {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
            ax.legend()
            ax.grid(alpha=0.3)
    
    # 4th subplot: Correlation heatmap
    ax = axes[1, 1]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_data = df[numeric_cols].corr()
        if HAS_SEABORN:
            sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                       ax=ax, square=True, linewidths=0.5)
        else:
            im = ax.imshow(corr_data, cmap='coolwarm', aspect='auto')
            ax.set_xticks(range(len(corr_data.columns)))
            ax.set_yticks(range(len(corr_data.columns)))
            ax.set_xticklabels(corr_data.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_data.columns)
            plt.colorbar(im, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'eda_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: eda_distributions.png")
    
    # 2. Activity vs Sleep relationships
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    relationships = [
        ('active_minutes', 'sleep_efficiency', 'Activity vs Sleep Efficiency'),
        ('steps', 'total_sleep_time', 'Steps vs Total Sleep Time'),
        ('daytime_hrv', 'deep_sleep_pct', 'HRV vs Deep Sleep %'),
        ('daytime_hr_avg', 'sleep_efficiency', 'Heart Rate vs Sleep Efficiency')
    ]
    
    for idx, (x_col, y_col, title) in enumerate(relationships):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if x_col in df.columns and y_col in df.columns:
            x_data = df[x_col].dropna()
            y_data = df[y_col].dropna()
            common_idx = x_data.index.intersection(y_data.index)
            
            if len(common_idx) > 0:
                ax.scatter(df.loc[common_idx, x_col], df.loc[common_idx, y_col], 
                          alpha=0.5, s=20)
                
                # Add trend line
                z = np.polyfit(df.loc[common_idx, x_col], df.loc[common_idx, y_col], 1)
                p = np.poly1d(z)
                ax.plot(df.loc[common_idx, x_col], p(df.loc[common_idx, x_col]), 
                       "r--", alpha=0.8, linewidth=2, label='Trend')
                
                # Calculate correlation
                corr = df.loc[common_idx, x_col].corr(df.loc[common_idx, y_col])
                ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=11)
                ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=11)
                ax.set_title(f'{title}\nCorrelation: {corr:.3f}', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'eda_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: eda_relationships.png")
    
    # 3. Time series plots (if date column exists)
    if 'date' in df.columns:
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        time_series_cols = ['sleep_efficiency', 'total_sleep_time', 'active_minutes']
        
        for idx, col in enumerate(time_series_cols):
            if col in df.columns:
                ax = axes[idx]
                df_sorted = df.sort_values('date')
                ax.plot(df_sorted['date'], df_sorted[col], marker='o', markersize=3, 
                       alpha=0.7, linewidth=1.5)
                ax.set_xlabel('Date', fontsize=11)
                ax.set_ylabel(col.replace('_', ' ').title(), fontsize=11)
                ax.set_title(f'{col.replace("_", " ").title()} Over Time', fontsize=12, fontweight='bold')
                ax.grid(alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'eda_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: eda_timeseries.png")
    
    print("✅ EDA visualizations complete!")

def create_performance_visualizations(results, predictions, targets, training_history=None, save_dir=OUTPUT_DIR):
    """
    Create comprehensive performance metric visualizations
    """
    print("Creating performance visualizations...")
    
    # 1. Prediction scatter plots (already in evaluation.py, but enhanced)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['sleep_efficiency', 'total_sleep_time', 'deep_sleep_pct']
    metric_labels = ['Sleep Efficiency', 'Total Sleep Time (hours)', 'Deep Sleep %']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if metric in predictions and len(predictions[metric]) > 0:
            pred = np.array(predictions[metric])
            tgt = np.array(targets[metric])
            
            axes[idx].scatter(tgt, pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(tgt.min(), pred.min())
            max_val = max(tgt.max(), pred.max())
            axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Metrics
            if metric in results:
                r2 = results[metric].get('r2', 0)
                mae = results[metric].get('mae', 0)
                axes[idx].set_title(f'{label}\nR² = {r2:.3f}, MAE = {mae:.3f}', fontsize=12, fontweight='bold')
            
            axes[idx].set_xlabel('Actual', fontsize=11)
            axes[idx].set_ylabel('Predicted', fontsize=11)
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'performance_predictions_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: performance_predictions_scatter.png")
    
    # 2. Metrics comparison bar chart
    if results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics_list = list(results.keys())
        metric_names = [m.replace('_', ' ').title() for m in metrics_list]
        
        # R² scores
        ax = axes[0, 0]
        r2_scores = [results[m].get('r2', 0) for m in metrics_list]
        bars = ax.bar(metric_names, r2_scores, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('R² Score by Metric', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(alpha=0.3, axis='y')
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # MAE
        ax = axes[0, 1]
        mae_scores = [results[m].get('mae', 0) for m in metrics_list]
        bars = ax.bar(metric_names, mae_scores, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title('Mean Absolute Error by Metric', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        for bar, score in zip(bars, mae_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # RMSE
        ax = axes[1, 0]
        rmse_scores = [results[m].get('rmse', 0) for m in metrics_list]
        bars = ax.bar(metric_names, rmse_scores, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('Root Mean Squared Error by Metric', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        for bar, score in zip(bars, rmse_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # MAPE
        ax = axes[1, 1]
        mape_scores = [results[m].get('mape', 0) for m in metrics_list]
        bars = ax.bar(metric_names, mape_scores, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
        ax.set_ylabel('MAPE (%)', fontsize=12)
        ax.set_title('Mean Absolute Percentage Error by Metric', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        for bar, score in zip(bars, mape_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                   f'{score:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: performance_metrics_comparison.png")
    
    # 3. Training history (if available)
    if training_history:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        train_losses = training_history.get('train_losses', [])
        val_losses = training_history.get('val_losses', [])
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'o-', label='Train Loss', alpha=0.7, linewidth=2, markersize=4)
        
        # Only plot validation if not all inf
        if val_losses and not all(np.isinf(v) or v == float('inf') for v in val_losses):
            val_losses_clean = [v if not (np.isinf(v) or v == float('inf')) else None for v in val_losses]
            ax.plot(epochs, val_losses_clean, 's-', label='Validation Loss', alpha=0.7, linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training History', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: performance_training_history.png")
    
    # 4. Error distribution
    if predictions and targets:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, metric in enumerate(metrics):
            if metric in predictions and len(predictions[metric]) > 0:
                pred = np.array(predictions[metric])
                tgt = np.array(targets[metric])
                errors = tgt - pred
                
                axes[idx].hist(errors, bins=30, edgecolor='black', alpha=0.7, 
                              color=['#2E86AB', '#A23B72', '#F18F01'][idx])
                axes[idx].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
                axes[idx].axvline(errors.mean(), color='green', linestyle='--', linewidth=2, 
                                 label=f'Mean: {errors.mean():.3f}')
                axes[idx].set_xlabel('Prediction Error', fontsize=11)
                axes[idx].set_ylabel('Frequency', fontsize=11)
                axes[idx].set_title(f'{metric.replace("_", " ").title()}\nError Distribution', 
                                   fontsize=12, fontweight='bold')
                axes[idx].legend()
                axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: performance_error_distribution.png")
    
    print("✅ Performance visualizations complete!")

def main():
    """Main function to create all visualizations"""
    print("=" * 60)
    print("CREATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 60)
    print()
    
    # 1. EDA Visualizations
    data_file = Path(__file__).parent.parent / 'data' / 'processed'
    processed_files = list(data_file.glob('*.csv'))
    
    if processed_files:
        df = pd.read_csv(processed_files[0])
        create_eda_visualizations(df)
        print()
    
    # 2. Performance Visualizations
    results_file = OUTPUT_DIR / 'evaluation_results.json'
    history_file = OUTPUT_DIR / 'training_history.json'
    
    results = None
    training_history = None
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    
    if history_file.exists():
        with open(history_file, 'r') as f:
            training_history = json.load(f)
    
    # Load predictions if available (from evaluation.py output)
    # For now, create placeholder - in real usage, these come from evaluation
    predictions = {}
    targets = {}
    
    if results:
        # Create performance visualizations
        create_performance_visualizations(results, predictions, targets, training_history)
    
    print()
    print("=" * 60)
    print("✅ All visualizations created!")
    print(f"📁 Saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()







