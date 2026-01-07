"""
Training script for Multimodal Sleep Forecasting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime

from data_loading import load_processed_data, create_data_loaders
from models.lstm_forecaster import MultimodalLSTMForecaster

# Directories
MODEL_DIR = Path(__file__).parent.parent / 'models'
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for predicting multiple sleep metrics
    """
    def __init__(self, weights={'sleep_efficiency': 1.0, 'total_sleep_time': 1.0, 'deep_sleep_pct': 1.0}):
        super().__init__()
        self.weights = weights
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # Initialize total_loss as a tensor (not float)
        # Get device from first prediction
        first_pred = predictions['sleep_efficiency']
        device = first_pred.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Handle targets - can be list of dicts or list of tuples
        if isinstance(targets, (list, tuple)) and len(targets) > 0:
            # Check if first element is dict
            if isinstance(targets[0], dict):
                # Targets are dicts
                for metric in ['sleep_efficiency', 'total_sleep_time', 'deep_sleep_pct']:
                    pred = predictions[metric]  # Shape: (batch_size, 1)
                    tgt = torch.tensor([t[metric] for t in targets], dtype=torch.float32, device=device).unsqueeze(1)  # Shape: (batch_size, 1)
                    loss = self.mse_loss(pred, tgt)
                    total_loss = total_loss + self.weights[metric] * loss
            else:
                # Targets might be in different format, try to extract
                # Assume targets is list of dicts from DataLoader
                for metric in ['sleep_efficiency', 'total_sleep_time', 'deep_sleep_pct']:
                    pred = predictions[metric]
                    tgt_values = []
                    for t in targets:
                        if isinstance(t, dict):
                            tgt_values.append(t[metric])
                        else:
                            # Fallback: assume order is sleep_efficiency, total_sleep_time, deep_sleep_pct
                            metric_idx = ['sleep_efficiency', 'total_sleep_time', 'deep_sleep_pct'].index(metric)
                            tgt_values.append(t[metric_idx] if isinstance(t, (list, tuple)) else 0.0)
                    tgt = torch.tensor(tgt_values, dtype=torch.float32, device=device).unsqueeze(1)  # Shape: (batch_size, 1)
                    loss = self.mse_loss(pred, tgt)
                    total_loss = total_loss + self.weights[metric] * loss
        else:
            # No targets - return zero loss tensor
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    train_loss = 0.0
    num_batches = 0
    
    for modalities, targets in tqdm(train_loader, desc='Training'):
        # Move to device
        modalities = {k: v.to(device) for k, v in modalities.items()}
        
        optimizer.zero_grad()
        predictions = model(modalities)
        loss = criterion(predictions, targets)
        
        # Check if loss is valid
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0.0:
            # Debug: print predictions and targets
            if num_batches == 0:  # Only print first batch
                print(f"  Debug - Loss: {loss.item()}")
                for metric in ['sleep_efficiency', 'total_sleep_time', 'deep_sleep_pct']:
                    if metric in predictions:
                        pred_vals = predictions[metric].detach().cpu().numpy()
                        print(f"    {metric} - pred: mean={pred_vals.mean():.4f}, std={pred_vals.std():.4f}, range=[{pred_vals.min():.4f}, {pred_vals.max():.4f}]")
                        if isinstance(targets, (list, tuple)) and len(targets) > 0 and isinstance(targets[0], dict):
                            tgt_vals = [t[metric] for t in targets]
                            print(f"    {metric} - tgt: mean={np.mean(tgt_vals):.4f}, std={np.std(tgt_vals):.4f}, range=[{min(tgt_vals):.4f}, {max(tgt_vals):.4f}]")
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
        num_batches += 1
    
    return train_loss / max(num_batches, 1)

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    val_loss = 0.0
    
    # Handle empty validation loader
    if len(val_loader) == 0:
        return float('inf')  # Return high loss if no validation data
    
    with torch.no_grad():
        for modalities, targets in tqdm(val_loader, desc='Validating'):
            modalities = {k: v.to(device) for k, v in modalities.items()}
            predictions = model(modalities)
            loss = criterion(predictions, targets)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda'):
    """
    Main training loop
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = MultiTaskLoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nTraining on device: {device}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate (skip if empty)
        if len(val_loader) > 0:
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = MODEL_DIR / 'best_lstm_forecaster.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, model_path)
                print(f"  → New best model saved! (Val Loss: {val_loss:.4f})")
        else:
            val_loss = float('inf')
            val_losses.append(val_loss)
            scheduler.step(train_loss)  # Use train_loss for scheduler
            print(f"Train Loss: {train_loss:.4f}, Val Loss: N/A (no validation data)")
            
            # Save best model based on train loss
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                model_path = MODEL_DIR / 'best_lstm_forecaster.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': train_loss,  # Use train_loss as proxy
                    'train_loss': train_loss,
                }, model_path)
                print(f"  → New best model saved! (Train Loss: {train_loss:.4f})")
        
        print()
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'num_epochs': num_epochs
    }
    
    history_path = OUTPUT_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("=" * 60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_DIR / 'best_lstm_forecaster.pt'}")
    print(f"History saved to: {history_path}")
    
    return train_losses, val_losses

def main():
    """Main training function"""
    print("=" * 60)
    print("MULTIMODAL SLEEP FORECASTING - TRAINING")
    print("=" * 60)
    print()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Load data
    print("Loading data...")
    df = load_processed_data()
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        df,
        sequence_length=14,
        forecast_horizon=1,
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.15
    )
    print()
    
    # Create model
    print("Initializing model...")
    model = MultimodalLSTMForecaster(
        hr_dim=4,
        hrv_dim=2,
        activity_dim=4,
        temp_dim=1,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    )
    print()
    
    # Train
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=50,
        lr=0.001,
        device=device
    )
    
    print("\n✅ Training completed successfully!")
    print("\nNext steps:")
    print("1. Evaluate model: python src/evaluation.py")
    print("2. Generate predictions: python src/forecasting.py")

if __name__ == '__main__':
    main()

