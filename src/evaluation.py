"""
Evaluation script for Multimodal Sleep Forecasting
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Try to import sklearn metrics, provide fallbacks if not available
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    # Fallback metrics
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

# Try to import seaborn, make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from data_loading import load_processed_data, create_data_loaders
from models.lstm_forecaster import MultimodalLSTMForecaster

# Directories
MODEL_DIR = Path(__file__).parent.parent / 'models'
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set
    Returns metrics and predictions
    """
    model.eval()
    all_predictions = {'sleep_efficiency': [], 'total_sleep_time': [], 'deep_sleep_pct': []}
    all_targets = {'sleep_efficiency': [], 'total_sleep_time': [], 'deep_sleep_pct': []}
    
    with torch.no_grad():
        for modalities, targets in test_loader:
            modalities = {k: v.to(device) for k, v in modalities.items()}
            predictions = model(modalities)
            
            # Handle targets - can be list of dicts
            for metric in ['sleep_efficiency', 'total_sleep_time', 'deep_sleep_pct']:
                # Flatten predictions from (batch_size, 1) to (batch_size,)
                all_predictions[metric].extend(predictions[metric].cpu().numpy().flatten())
                
                # Extract targets - handle both dict and other formats
                if isinstance(targets, (list, tuple)) and len(targets) > 0:
                    if isinstance(targets[0], dict):
                        all_targets[metric].extend([t[metric] for t in targets])
                    else:
                        # Fallback: assume targets are in order
                        metric_idx = ['sleep_efficiency', 'total_sleep_time', 'deep_sleep_pct'].index(metric)
                        all_targets[metric].extend([t[metric_idx] if isinstance(t, (list, tuple)) and len(t) > metric_idx else 0.0 for t in targets])
    
    # Calculate metrics
    results = {}
    for metric in ['sleep_efficiency', 'total_sleep_time', 'deep_sleep_pct']:
        pred = np.array(all_predictions[metric])
        tgt = np.array(all_targets[metric])
        
        # Check if we have valid data
        if len(pred) == 0 or len(tgt) == 0:
            print(f"  ⚠️  No predictions/targets for {metric}")
            results[metric] = {
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'mape': np.nan
            }
            continue
        
        # Remove any NaN or Inf values
        valid_mask = np.isfinite(pred) & np.isfinite(tgt)
        if valid_mask.sum() == 0:
            print(f"  ⚠️  No valid predictions/targets for {metric}")
            results[metric] = {
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'mape': np.nan
            }
            continue
        
        pred_clean = pred[valid_mask]
        tgt_clean = tgt[valid_mask]
        
        if len(pred_clean) == 0:
            results[metric] = {
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'mape': np.nan
            }
            continue
        
        # Calculate metrics
        try:
            mae = mean_absolute_error(tgt_clean, pred_clean)
            rmse = np.sqrt(mean_squared_error(tgt_clean, pred_clean))
            r2 = r2_score(tgt_clean, pred_clean)
            mape = np.mean(np.abs((tgt_clean - pred_clean) / (tgt_clean + 1e-8))) * 100
            
            results[metric] = {
                'mae': float(mae) if np.isfinite(mae) else np.nan,
                'rmse': float(rmse) if np.isfinite(rmse) else np.nan,
                'r2': float(r2) if np.isfinite(r2) else np.nan,
                'mape': float(mape) if np.isfinite(mape) else np.nan
            }
        except Exception as e:
            print(f"  ⚠️  Error calculating metrics for {metric}: {e}")
            results[metric] = {
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'mape': np.nan
            }
    
    return results, all_predictions, all_targets

def plot_predictions(results, predictions, targets, save_path=None):
    """Create prediction scatter plots"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['sleep_efficiency', 'total_sleep_time', 'deep_sleep_pct']
    metric_labels = ['Sleep Efficiency', 'Total Sleep Time (hours)', 'Deep Sleep %']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        pred = np.array(predictions[metric])
        tgt = np.array(targets[metric])
        
        axes[idx].scatter(tgt, pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(tgt.min(), pred.min())
        max_val = max(tgt.max(), pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        # Metrics
        r2 = results[metric]['r2']
        mae = results[metric]['mae']
        
        axes[idx].set_xlabel('Actual', fontsize=12)
        axes[idx].set_ylabel('Predicted', fontsize=12)
        axes[idx].set_title(f'{label}\nR² = {r2:.3f}, MAE = {mae:.3f}', fontsize=12)
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved predictions plot: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_training_history(history_path, save_path=None):
    """Plot training history"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training History', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot: {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    """Main evaluation function"""
    print("=" * 60)
    print("MULTIMODAL SLEEP FORECASTING - EVALUATION")
    print("=" * 60)
    print()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Load model
    model_path = MODEL_DIR / 'best_lstm_forecaster.pt'
    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        print("   Run training first: python src/training.py")
        return
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = MultimodalLSTMForecaster(
        hr_dim=4,
        hrv_dim=2,
        activity_dim=4,
        temp_dim=1,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f})")
    print()
    
    # Load data
    print("Loading test data...")
    df = load_processed_data()
    print()
    
    # Create test loader
    _, _, test_loader = create_data_loaders(
        df,
        sequence_length=14,
        forecast_horizon=1,
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.15
    )
    print()
    
    # Evaluate
    print("Evaluating on test set...")
    results, predictions, targets = evaluate_model(model, test_loader, device)
    print()
    
    # Print results
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print()
    
    for metric, metrics_dict in results.items():
        print(f"{metric.replace('_', ' ').title()}:")
        print(f"  MAE:  {metrics_dict['mae']:.4f}")
        print(f"  RMSE: {metrics_dict['rmse']:.4f}")
        print(f"  R²:   {metrics_dict['r2']:.4f}")
        print(f"  MAPE: {metrics_dict['mape']:.2f}%")
        print()
    
    # Save results
    results_path = OUTPUT_DIR / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Predictions scatter plot
    plot_predictions(
        results, predictions, targets,
        save_path=OUTPUT_DIR / 'predictions_scatter.png'
    )
    
    # Training history
    history_path = OUTPUT_DIR / 'training_history.json'
    if history_path.exists():
        plot_training_history(
            history_path,
            save_path=OUTPUT_DIR / 'training_history.png'
        )
    
    # Create comprehensive visualizations (EDA + Performance)
    try:
        from create_visualizations import create_eda_visualizations, create_performance_visualizations
        
        # EDA visualizations
        df = load_processed_data()
        create_eda_visualizations(df)
        
        # Enhanced performance visualizations
        create_performance_visualizations(results, predictions, targets, 
                                        training_history=json.load(open(history_path)) if history_path.exists() else None)
    except Exception as e:
        print(f"  ⚠️  Could not create additional visualizations: {e}")
    
    print("\n✅ Evaluation completed!")
    print(f"\nVisualizations saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()

