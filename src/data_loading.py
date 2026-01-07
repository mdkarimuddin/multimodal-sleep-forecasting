"""
Data loading and PyTorch Dataset for multimodal wearable data
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Try to import sklearn, provide fallback if not available
try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    # Fallback train_test_split
    def train_test_split(*arrays, test_size=0.2, random_state=None, **kwargs):
        """Simple fallback train_test_split"""
        if random_state is not None:
            np.random.seed(random_state)
        n = len(arrays[0])
        n_test = int(n * test_size)
        indices = np.random.permutation(n)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        result = []
        for array in arrays:
            if isinstance(array, pd.DataFrame) or isinstance(array, pd.Series):
                result.append(array.iloc[train_indices])
                result.append(array.iloc[test_indices])
            else:
                result.append(array[train_indices])
                result.append(array[test_indices])
        return tuple(result)

DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'

class MultimodalWearableDataset(Dataset):
    """
    PyTorch Dataset for multimodal wearable data
    Creates sequences for time-series forecasting
    """
    def __init__(self, df, sequence_length=14, forecast_horizon=1):
        """
        Args:
            df: Preprocessed DataFrame
            sequence_length: Number of days to use as input (default: 14)
            forecast_horizon: Days ahead to predict (default: 1)
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Sort by user and date
        if 'user_id' in df.columns and 'date' in df.columns:
            df = df.sort_values(['user_id', 'date']).reset_index(drop=True)
        
        # Create sequences
        self.sequences, self.targets = self._create_sequences(df)
        
        print(f"Created {len(self.sequences)} sequences")
        print(f"  Sequence length: {sequence_length} days")
        print(f"  Forecast horizon: {forecast_horizon} day(s)")
    
    def _create_sequences(self, df):
        """Create input sequences and targets"""
        sequences = []
        targets = []
        
        # Group by user
        if 'user_id' in df.columns:
            user_groups = df.groupby('user_id')
        else:
            # If no user_id, treat as single user
            user_groups = [(0, df)]
        
        for user_id, user_data in user_groups:
            user_data = user_data.reset_index(drop=True)
            
            # Create sequences for this user
            for i in range(len(user_data) - self.sequence_length - self.forecast_horizon + 1):
                # Input sequence (past N days)
                seq = user_data.iloc[i:i+self.sequence_length]
                
                # Target (future sleep quality)
                target_idx = i + self.sequence_length + self.forecast_horizon - 1
                target = user_data.iloc[target_idx]
                
                # Extract multimodal features
                multimodal_seq = self._extract_modalities(seq)
                
                # Extract targets
                target_sleep = {
                    'sleep_efficiency': target.get('sleep_efficiency', 0.8),
                    'total_sleep_time': target.get('total_sleep_time', 7.5),
                    'deep_sleep_pct': target.get('deep_sleep_pct', 0.18)
                }
                
                sequences.append(multimodal_seq)
                targets.append(target_sleep)
        
        return sequences, targets
    
    def _extract_modalities(self, seq):
        """
        Extract features for each modality
        Returns dict with keys: 'hr', 'hrv', 'activity', 'temp'
        """
        modalities = {}
        
        # Heart Rate modality
        hr_features = []
        if 'daytime_hr_avg' in seq.columns:
            hr_features.append(seq['daytime_hr_avg'].values)
        if 'daytime_hr_max' in seq.columns:
            hr_features.append(seq['daytime_hr_max'].values)
        elif 'daytime_hr_avg' in seq.columns:
            hr_features.append(seq['daytime_hr_avg'].values * 1.2)  # Estimate max
        if 'daytime_hr_min' in seq.columns:
            hr_features.append(seq['daytime_hr_min'].values)
        elif 'daytime_hr_avg' in seq.columns:
            hr_features.append(seq['daytime_hr_avg'].values * 0.8)  # Estimate min
        if 'sleep_hr_avg' in seq.columns:
            hr_features.append(seq['sleep_hr_avg'].values)
        elif 'daytime_hr_avg' in seq.columns:
            hr_features.append(seq['daytime_hr_avg'].values * 0.85)  # Estimate sleep HR
        
        if hr_features:
            modalities['hr'] = torch.FloatTensor(np.array(hr_features).T)
        else:
            # Fallback: create dummy HR data
            modalities['hr'] = torch.FloatTensor(np.random.uniform(60, 80, (len(seq), 4)))
        
        # HRV modality
        hrv_features = []
        if 'daytime_hrv' in seq.columns:
            hrv_features.append(seq['daytime_hrv'].values)
        if 'sleep_hrv' in seq.columns:
            hrv_features.append(seq['sleep_hrv'].values)
        elif 'daytime_hrv' in seq.columns:
            hrv_features.append(seq['daytime_hrv'].values * 1.1)  # Estimate sleep HRV
        elif 'rmssd' in seq.columns:
            hrv_features.append(seq['rmssd'].values)
        
        if hrv_features:
            modalities['hrv'] = torch.FloatTensor(np.array(hrv_features).T)
        else:
            # Fallback: create dummy HRV data
            modalities['hrv'] = torch.FloatTensor(np.random.uniform(40, 80, (len(seq), 2)))
        
        # Activity modality
        activity_features = []
        if 'steps' in seq.columns:
            activity_features.append(seq['steps'].values)
        if 'active_minutes' in seq.columns:
            activity_features.append(seq['active_minutes'].values)
        if 'calories_burned' in seq.columns:
            activity_features.append(seq['calories_burned'].values)
        elif 'steps' in seq.columns:
            # Estimate calories from steps
            activity_features.append(seq['steps'].values * 0.04)
        if 'training_strain' in seq.columns:
            activity_features.append(seq['training_strain'].values)
        elif 'active_minutes' in seq.columns:
            # Estimate training strain
            activity_features.append(seq['active_minutes'].values / 60)
        
        if activity_features:
            modalities['activity'] = torch.FloatTensor(np.array(activity_features).T)
        else:
            # Fallback: create dummy activity data
            modalities['activity'] = torch.FloatTensor(np.random.uniform(0, 10000, (len(seq), 4)))
        
        # Temperature (synthetic if not available)
        if 'temperature' in seq.columns:
            modalities['temp'] = torch.FloatTensor(seq['temperature'].values).unsqueeze(1)
        elif 'daytime_hr_avg' in seq.columns:
            # Estimate temperature from HR
            temp = 36.5 + (seq['daytime_hr_avg'].values - 65) * 0.02
            modalities['temp'] = torch.FloatTensor(temp).unsqueeze(1)
        else:
            # Fallback: create dummy temperature
            modalities['temp'] = torch.FloatTensor(np.random.uniform(36.3, 36.8, (len(seq), 1)))
        
        return modalities
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def load_processed_data(data_file=None):
    """
    Load preprocessed data
    """
    if data_file is None:
        # Find first processed CSV file
        processed_files = list(DATA_DIR.glob('*.csv'))
        if not processed_files:
            raise FileNotFoundError(f"No processed data found in {DATA_DIR}")
        data_file = processed_files[0]
        print(f"Using data file: {data_file.name}")
    
    df = pd.read_csv(data_file)
    
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].notna()].copy()
    
    print(f"Loaded {len(df)} records")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df

def create_data_loaders(df, sequence_length=14, forecast_horizon=1, 
                        batch_size=32, train_ratio=0.7, val_ratio=0.15, min_sequence_size=None):
    """
    Create train/val/test data loaders
    
    Note: Adjusts sequence_length if dataset is too small to ensure we have sequences
    """
    """
    Create train/val/test data loaders
    
    Args:
        df: Preprocessed DataFrame
        sequence_length: Input sequence length
        forecast_horizon: Forecast horizon
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Split by user (if user_id exists) to prevent data leakage
    if 'user_id' in df.columns:
        users = df['user_id'].unique()
        
        # If only one user, split by time instead
        if len(users) == 1:
            print(f"⚠️  Only 1 user found, splitting by time instead of by user")
            if 'date' in df.columns:
                df = df.sort_values('date').reset_index(drop=True)
                n = len(df)
                train_end = int(n * train_ratio)
                val_end = int(n * (train_ratio + val_ratio))
                
                train_df = df.iloc[:train_end].copy()
                val_df = df.iloc[train_end:val_end].copy()
                test_df = df.iloc[val_end:].copy()
            else:
                # Random split
                train_df, temp_df = train_test_split(df, test_size=1-train_ratio, random_state=42)
                val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
            
            print(f"Split by time:")
            print(f"  Train: {len(train_df)} records")
            print(f"  Val: {len(val_df)} records")
            print(f"  Test: {len(test_df)} records")
        else:
            # Multiple users - split by user
            train_users, temp_users = train_test_split(users, test_size=1-train_ratio, random_state=42)
            val_users, test_users = train_test_split(
                temp_users, 
                test_size=(1 - val_ratio/(1-train_ratio)), 
                random_state=42
            )
            
            train_df = df[df['user_id'].isin(train_users)].copy()
            val_df = df[df['user_id'].isin(val_users)].copy()
            test_df = df[df['user_id'].isin(test_users)].copy()
            
            print(f"Split by user:")
            print(f"  Train: {len(train_users)} users, {len(train_df)} records")
            print(f"  Val: {len(val_users)} users, {len(val_df)} records")
            print(f"  Test: {len(test_users)} users, {len(test_df)} records")
    else:
        # Split by time (if date exists)
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
            n = len(df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:].copy()
        else:
            # Random split
            train_df, temp_df = train_test_split(df, test_size=1-train_ratio, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        print(f"Split by time/random:")
        print(f"  Train: {len(train_df)} records")
        print(f"  Val: {len(val_df)} records")
        print(f"  Test: {len(test_df)} records")
    
    # Adjust sequence_length if dataset is too small to ensure we have validation/test sequences
    # Check if we can create sequences from validation and test sets
    min_data_needed = sequence_length + forecast_horizon
    
    if len(val_df) < min_data_needed or len(test_df) < min_data_needed:
        # Reduce sequence length to ensure we have sequences
        max_possible_seq = min(len(train_df), len(val_df), len(test_df)) - forecast_horizon
        if max_possible_seq < sequence_length:
            adjusted_seq_len = max(7, max_possible_seq)
            print(f"⚠️  Dataset small, adjusting sequence_length from {sequence_length} to {adjusted_seq_len}")
            print(f"   This ensures we have sequences in validation ({len(val_df)} records) and test ({len(test_df)} records)")
            sequence_length = adjusted_seq_len
    
    # Create datasets
    train_dataset = MultimodalWearableDataset(train_df, sequence_length, forecast_horizon)
    val_dataset = MultimodalWearableDataset(val_df, sequence_length, forecast_horizon)
    test_dataset = MultimodalWearableDataset(test_df, sequence_length, forecast_horizon)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Test data loading
    print("Testing data loading...")
    
    try:
        df = load_processed_data()
        train_loader, val_loader, test_loader = create_data_loaders(df)
        
        # Test one batch
        for modalities, targets in train_loader:
            print("\nBatch sample:")
            print(f"  Modalities: {list(modalities.keys())}")
            for key, value in modalities.items():
                print(f"    {key}: shape {value.shape}")
            print(f"  Targets: {list(targets[0].keys())}")
            break
        
        print("\n✅ Data loading test successful!")
        
    except Exception as e:
        print(f"⚠️  Error: {e}")
        import traceback
        traceback.print_exc()

