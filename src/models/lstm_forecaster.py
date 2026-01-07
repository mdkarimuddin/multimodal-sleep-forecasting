"""
Multimodal LSTM Forecaster for Sleep Quality Prediction
PyTorch implementation with multimodal fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalLSTMForecaster(nn.Module):
    """
    LSTM-based forecaster with multimodal fusion
    
    Architecture:
    - Separate LSTM encoders for each modality (HR, HRV, Activity, Temperature)
    - Late fusion: Concatenate final hidden states
    - Multi-task head: Predict sleep_efficiency, total_sleep_time, deep_sleep_pct
    """
    def __init__(self, 
                 hr_dim=4, hrv_dim=2, activity_dim=4, temp_dim=1,
                 hidden_dim=128, num_layers=2, dropout=0.2):
        """
        Args:
            hr_dim: Heart rate feature dimension
            hrv_dim: HRV feature dimension
            activity_dim: Activity feature dimension
            temp_dim: Temperature feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hr_dim = hr_dim
        self.hrv_dim = hrv_dim
        self.activity_dim = activity_dim
        self.temp_dim = temp_dim
        self.hidden_dim = hidden_dim
        
        # Separate encoders for each modality
        self.hr_encoder = nn.LSTM(
            hr_dim, 
            hidden_dim // 4, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.hrv_encoder = nn.LSTM(
            hrv_dim, 
            hidden_dim // 4, 
            num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.activity_encoder = nn.LSTM(
            activity_dim, 
            hidden_dim // 4, 
            num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.temp_encoder = nn.LSTM(
            temp_dim, 
            hidden_dim // 4, 
            num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Forecasting head (multi-task)
        # Add activation/sigmoid for bounded outputs
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # sleep_efficiency, total_sleep_time, deep_sleep_pct
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, modalities):
        """
        Forward pass
        
        Args:
            modalities: dict with keys 'hr', 'hrv', 'activity', 'temp'
                Each value: (batch_size, sequence_length, feature_dim)
        
        Returns:
            dict with predictions for each sleep metric
        """
        # Encode each modality
        hr_out, (hr_h, _) = self.hr_encoder(modalities['hr'])
        hrv_out, (hrv_h, _) = self.hrv_encoder(modalities['hrv'])
        activity_out, (act_h, _) = self.activity_encoder(modalities['activity'])
        temp_out, (temp_h, _) = self.temp_encoder(modalities['temp'])
        
        # Use final hidden states (last layer, last timestep)
        hr_final = hr_h[-1]  # (batch_size, hidden_dim//4)
        hrv_final = hrv_h[-1]
        activity_final = act_h[-1]
        temp_final = temp_h[-1]
        
        # Concatenate modalities (late fusion)
        fused = torch.cat([hr_final, hrv_final, activity_final, temp_final], dim=1)
        
        # Fusion layer
        fused = self.fusion(fused)
        fused = F.relu(fused)
        fused = self.dropout(fused)
        
        # Forecast
        predictions = self.forecast_head(fused)
        
        # Apply appropriate activations for bounded outputs
        # Sleep efficiency: 0-1 (sigmoid)
        # Total sleep time: 4-11 hours (sigmoid scaled)
        # Deep sleep %: 0-1 (sigmoid)
        sleep_efficiency = torch.sigmoid(predictions[:, 0]) * 0.4 + 0.6  # Scale to [0.6, 1.0]
        total_sleep_time = torch.sigmoid(predictions[:, 1]) * 7 + 4  # Scale to [4, 11] hours
        deep_sleep_pct = torch.sigmoid(predictions[:, 2]) * 0.25 + 0.05  # Scale to [0.05, 0.30]
        
        return {
            'sleep_efficiency': sleep_efficiency,
            'total_sleep_time': total_sleep_time,
            'deep_sleep_pct': deep_sleep_pct
        }
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == '__main__':
    # Test model
    print("Testing MultimodalLSTMForecaster...")
    
    # Create model
    model = MultimodalLSTMForecaster(
        hr_dim=4, hrv_dim=2, activity_dim=4, temp_dim=1,
        hidden_dim=128, num_layers=2, dropout=0.2
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create dummy input
    batch_size = 8
    sequence_length = 14
    
    modalities = {
        'hr': torch.randn(batch_size, sequence_length, 4),
        'hrv': torch.randn(batch_size, sequence_length, 2),
        'activity': torch.randn(batch_size, sequence_length, 4),
        'temp': torch.randn(batch_size, sequence_length, 1)
    }
    
    # Forward pass
    predictions = model(modalities)
    
    print("\nPredictions:")
    for key, value in predictions.items():
        print(f"  {key}: shape {value.shape}, range [{value.min():.3f}, {value.max():.3f}]")
    
    print("\n✅ Model test successful!")

