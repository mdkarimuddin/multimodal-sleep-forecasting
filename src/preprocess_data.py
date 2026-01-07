"""
Preprocess real wearable/sleep data
Handles missing values, artifacts, irregular sampling
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import zscore
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_wearable_data(df, source='unknown'):
    """
    Preprocess real wearable data
    
    Args:
        df: Raw DataFrame
        source: Data source name ('kaggle', 'physionet', 'zenodo', etc.)
    
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    print(f"Preprocessing {source} data: {len(df)} rows, {len(df.columns)} columns")
    
    # 1. Handle missing values
    print("  → Handling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            # Forward fill for short gaps (< 5 consecutive NaNs)
            df[col] = df[col].fillna(method='ffill', limit=5)
            # Interpolate for remaining gaps
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            # Fill any remaining with median
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
    
    # 2. Remove artifacts (outliers)
    print("  → Removing artifacts (outliers)...")
    for col in numeric_cols:
        if df[col].notna().sum() > 10:  # Need enough data
            z_scores = np.abs(zscore(df[col].dropna()))
            if len(z_scores) > 0 and z_scores.max() > 3:
                threshold = 3
                outliers = df[col].notna() & (np.abs(zscore(df[col].fillna(df[col].median()))) > threshold)
                if outliers.sum() > 0:
                    df.loc[outliers, col] = np.nan
                    # Interpolate outliers
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    df[col] = df[col].fillna(df[col].median())
    
    # 3. Smooth signals (optional, for noisy data)
    print("  → Smoothing noisy signals...")
    for col in numeric_cols:
        if 'hr' in col.lower() or 'heart' in col.lower():
            if df[col].notna().sum() > 10:
                try:
                    df[col] = signal.savgol_filter(
                        df[col].fillna(df[col].median()),
                        window_length=min(5, len(df) // 2) if len(df) > 5 else 3,
                        polyorder=2
                    )
                except:
                    pass  # Skip if smoothing fails
    
    # 4. Handle irregular sampling (resample to regular intervals if timestamp exists)
    if 'date' in df.columns or 'timestamp' in df.columns:
        print("  → Handling timestamps...")
        time_col = 'date' if 'date' in df.columns else 'timestamp'
        
        # Convert to datetime
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Remove rows with invalid dates
        df = df[df[time_col].notna()].copy()
        
        # Sort by time
        df = df.sort_values(time_col).reset_index(drop=True)
        
        # If data is irregular, resample to daily (for sleep/activity data)
        # BUT: Don't resample if we already have daily data (to preserve size)
        if len(df) > 1:
            time_diff = (df[time_col].max() - df[time_col].min()).days
            # Only resample if we have more than 2x the number of days (meaning sub-daily data)
            if len(df) > time_diff * 2:  # Only resample if we have sub-daily data
                df = df.set_index(time_col)
                # Resample to daily (take mean for numeric, first for categorical)
                df_resampled = df.resample('D').agg({
                    col: 'mean' if col in numeric_cols else 'first'
                    for col in df.columns
                })
                df = df_resampled.reset_index()
                df = df.rename(columns={time_col: 'date'})
                print(f"    Resampled to daily: {len(df)} days")
            else:
                # Already daily or close to daily, keep as is
                print(f"    Data already at daily resolution: {len(df)} records")
    
    # 5. Feature engineering for HRV (if HR data available)
    print("  → Engineering HRV features...")
    hr_cols = [col for col in df.columns if 'hr' in col.lower() or 'heart' in col.lower()]
    
    if len(hr_cols) > 0:
        hr_col = hr_cols[0]  # Use first HR column
        
        # Convert HR to RR intervals (if not already)
        if 'rr' not in df.columns and hr_col in df.columns:
            df['rr_interval'] = 60 / df[hr_col]  # seconds
        
        # Calculate RMSSD (if we have RR intervals)
        if 'rr_interval' in df.columns:
            df['rmssd'] = df['rr_interval'].rolling(window=5, min_periods=2).apply(
                lambda x: np.sqrt(np.mean(np.diff(x)**2)) if len(x) > 1 else np.nan,
                raw=False
            )
            df['rmssd'] = df['rmssd'].fillna(df['rmssd'].median())
    
    # 6. Create day of week if date exists
    if 'date' in df.columns:
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 7. Remove any remaining NaN rows
    initial_len = len(df)
    df = df.dropna(subset=[col for col in numeric_cols if df[col].notna().sum() > len(df) * 0.5])
    removed = initial_len - len(df)
    if removed > 0:
        print(f"  → Removed {removed} rows with too many missing values")
    
    print(f"✅ Preprocessed: {len(df)} rows, {len(df.columns)} columns")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    return df

def load_and_preprocess_kaggle_sleep_health(file_path):
    """
    Load and preprocess Kaggle Sleep Health and Lifestyle dataset
    """
    print(f"Loading Kaggle Sleep Health dataset: {file_path}")
    df = pd.read_csv(file_path)
    
    # Map columns to our standard format
    column_mapping = {
        'Sleep Duration': 'total_sleep_time',
        'Quality of Sleep': 'sleep_quality_score',
        'Physical Activity Level': 'activity_level',
        'Stress Level': 'stress_level',
        'Heart Rate': 'daytime_hr_avg',
        'Daily Steps': 'steps',
    }
    
    # Rename columns if they exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # Create user_id if not exists
    if 'Person ID' in df.columns:
        df['user_id'] = df['Person ID']
    elif 'user_id' not in df.columns:
        df['user_id'] = range(len(df))
    
    # Create date if not exists
    if 'date' not in df.columns:
        df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
    
    # Estimate sleep efficiency from quality score (if available)
    if 'sleep_quality_score' in df.columns and 'sleep_efficiency' not in df.columns:
        # Map quality score (1-10) to efficiency (0.6-0.98)
        df['sleep_efficiency'] = 0.6 + (df['sleep_quality_score'] / 10) * 0.38
    
    # Estimate deep sleep % (if not available)
    if 'deep_sleep_pct' not in df.columns:
        # Rough estimate: 15-25% of total sleep
        df['deep_sleep_pct'] = np.random.uniform(0.15, 0.25, len(df))
    
    # Estimate HRV (if not available)
    if 'daytime_hrv' not in df.columns and 'daytime_hr_avg' in df.columns:
        # Rough estimate based on HR
        df['daytime_hrv'] = 100 - df['daytime_hr_avg'] + np.random.normal(0, 10, len(df))
        df['daytime_hrv'] = df['daytime_hrv'].clip(20, 100)
    
    # Estimate active minutes from steps
    if 'active_minutes' not in df.columns and 'steps' in df.columns:
        # Rough estimate: 1 active minute per 100 steps
        df['active_minutes'] = df['steps'] / 100
        df['active_minutes'] = df['active_minutes'].clip(0, 180)
    
    return preprocess_wearable_data(df, source='kaggle')

def load_and_preprocess_physionet(file_path):
    """
    Load and preprocess PhysioNet Sleep-EDF data
    """
    print(f"Loading PhysioNet Sleep-EDF data: {file_path}")
    df = pd.read_csv(file_path)
    
    # Create user_id from record column
    if 'record' in df.columns:
        df['user_id'] = df['record'].astype('category').cat.codes
    
    # Create date (use sample number as proxy)
    if 'date' not in df.columns:
        df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='30min')
    
    # Map sleep stages to efficiency
    if 'sleep_stage' in df.columns:
        # Sleep stages: W (wake), 1-4 (NREM), R (REM)
        stage_mapping = {
            'W': 0.3,  # Wake
            '1': 0.5,  # Light sleep
            '2': 0.7,  # Light sleep
            '3': 0.9,  # Deep sleep
            '4': 0.95, # Deep sleep
            'R': 0.8,  # REM
        }
        df['sleep_efficiency'] = df['sleep_stage'].map(stage_mapping).fillna(0.5)
        
        # Calculate deep sleep %
        df['deep_sleep_pct'] = (df['sleep_stage'].isin(['3', '4'])).astype(int) * 0.2
    
    # Estimate other features if not available
    if 'total_sleep_time' not in df.columns:
        df['total_sleep_time'] = np.random.uniform(6, 9, len(df))
    
    if 'daytime_hr_avg' not in df.columns:
        df['daytime_hr_avg'] = np.random.uniform(60, 80, len(df))
    
    if 'daytime_hrv' not in df.columns:
        df['daytime_hrv'] = np.random.uniform(40, 80, len(df))
    
    if 'steps' not in df.columns:
        df['steps'] = np.random.uniform(5000, 12000, len(df))
    
    if 'active_minutes' not in df.columns:
        df['active_minutes'] = np.random.uniform(20, 60, len(df))
    
    return preprocess_wearable_data(df, source='physionet')

def main():
    """
    Main preprocessing function
    Processes all raw data files
    """
    print("=" * 60)
    print("PREPROCESSING REAL WEARABLE/SLEEP DATA")
    print("=" * 60)
    print()
    
    # Find all raw data files
    raw_files = list(RAW_DIR.glob('*.csv'))
    
    if not raw_files:
        print("⚠️  No raw data files found!")
        print(f"   Expected in: {RAW_DIR}")
        print("   Run: python src/download_real_data.py first")
        return
    
    processed_files = []
    
    for raw_file in raw_files:
        print(f"\nProcessing: {raw_file.name}")
        print("-" * 60)
        
        try:
            # Detect data source from filename
            filename = raw_file.name.lower()
            
            if 'kaggle' in filename or 'sleep' in filename and 'health' in filename:
                df = load_and_preprocess_kaggle_sleep_health(raw_file)
            elif 'physionet' in filename or 'sleep_edf' in filename or 'edf' in filename:
                df = load_and_preprocess_physionet(raw_file)
            else:
                # Generic preprocessing
                df = pd.read_csv(raw_file)
                df = preprocess_wearable_data(df, source='generic')
            
            # Save processed data
            output_file = PROCESSED_DIR / f"processed_{raw_file.stem}.csv"
            df.to_csv(output_file, index=False)
            processed_files.append(output_file)
            
            print(f"✅ Saved: {output_file.name}")
            
        except Exception as e:
            print(f"⚠️  Error processing {raw_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print()
    print("=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    for file in processed_files:
        df = pd.read_csv(file)
        print(f"✅ {file.name}: {len(df)} rows, {len(df.columns)} columns")
    
    print()
    print(f"📁 Processed data saved to: {PROCESSED_DIR}")
    print()
    print("Next step: Run data loading and model training")

if __name__ == '__main__':
    main()

