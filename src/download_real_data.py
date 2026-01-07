"""
Download real wearable/sleep datasets from various sources
Supports: PhysioNet, Kaggle, Zenodo, GitHub
"""

import os
import pandas as pd
import numpy as np
import requests
import zipfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Create data directory
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_kaggle_sleep_health():
    """
    Download Sleep Health and Lifestyle Dataset from Kaggle
    Dataset: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
    
    Note: Requires Kaggle API setup (kaggle.json)
    Alternative: Manual download from Kaggle website
    """
    try:
        import kaggle
        
        dataset_name = 'uom190346a/sleep-health-and-lifestyle-dataset'
        print(f"Downloading {dataset_name} from Kaggle...")
        
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(DATA_DIR),
            unzip=True
        )
        
        # Find CSV file
        csv_files = list(DATA_DIR.glob('*.csv'))
        if csv_files:
            print(f"✅ Downloaded: {csv_files[0].name}")
            return str(csv_files[0])
        else:
            print("⚠️  No CSV file found")
            return None
            
    except ImportError:
        print("⚠️  Kaggle API not installed. Install with: pip install kaggle")
        print("   Or download manually from: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset")
        return None
    except Exception as e:
        print(f"⚠️  Error downloading from Kaggle: {e}")
        print("   Try manual download from Kaggle website")
        return None

def download_zenodo_aauwss():
    """
    Download Aalborg University Wearable Sleep Study (AAUWSS) from Zenodo
    URL: https://zenodo.org/records/16919071
    
    Note: This is a large dataset, may take time to download
    """
    zenodo_id = "16919071"
    base_url = f"https://zenodo.org/api/records/{zenodo_id}"
    
    try:
        print(f"Fetching AAUWSS dataset info from Zenodo...")
        response = requests.get(base_url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            files = data.get('files', [])
            
            print(f"Found {len(files)} files in dataset")
            
            # Download first CSV file (if available)
            for file_info in files:
                if file_info['key'].endswith('.csv'):
                    file_url = file_info['links']['self']
                    file_name = file_info['key']
                    output_path = DATA_DIR / file_name
                    
                    print(f"Downloading {file_name}...")
                    file_response = requests.get(file_url, timeout=300)
                    
                    if file_response.status_code == 200:
                        output_path.write_bytes(file_response.content)
                        print(f"✅ Downloaded: {file_name}")
                        return str(output_path)
            
            print("⚠️  No CSV files found in dataset")
            return None
        else:
            print(f"⚠️  Error accessing Zenodo: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"⚠️  Error downloading from Zenodo: {e}")
        print("   Try manual download from: https://zenodo.org/records/16919071")
        return None

def download_physionet_sleep_edf():
    """
    Download Sleep-EDF database from PhysioNet
    Requires: wfdb package and PhysioNet account
    
    Note: You need to register at https://physionet.org/ first
    """
    try:
        import wfdb
        
        print("Downloading Sleep-EDF database from PhysioNet...")
        print("Note: Requires PhysioNet account registration")
        
        # Get list of available records
        try:
            records = wfdb.get_record_list('sleep-edf', records='all')
            print(f"Found {len(records)} records in Sleep-EDF database")
            
            # Download first 3 records as example (to avoid large downloads)
            data_list = []
            
            for record_name in records[:3]:
                try:
                    print(f"  Downloading {record_name}...")
                    
                    # Download annotation (sleep stages)
                    ann = wfdb.rdann(record_name, 'hypnogram', pn_dir='sleep-edf')
                    
                    # Download signal
                    record = wfdb.rdrecord(record_name, pn_dir='sleep-edf')
                    
                    # Extract sleep stages
                    sleep_stages = ann.symbol
                    
                    # Extract signals
                    signals = record.p_signal
                    signal_names = record.sig_name
                    
                    # Create DataFrame
                    df = pd.DataFrame(signals, columns=signal_names)
                    df['sleep_stage'] = sleep_stages[:len(df)]
                    df['record'] = record_name
                    df['sample'] = range(len(df))
                    
                    data_list.append(df)
                    print(f"    ✅ Downloaded {record_name}: {len(df)} samples")
                    
                except Exception as e:
                    print(f"    ⚠️  Error with {record_name}: {e}")
                    continue
            
            if data_list:
                combined_df = pd.concat(data_list, ignore_index=True)
                output_path = DATA_DIR / 'sleep_edf.csv'
                combined_df.to_csv(output_path, index=False)
                print(f"✅ Saved combined data: {len(combined_df)} samples")
                return str(output_path)
            else:
                print("⚠️  No data downloaded")
                return None
                
        except Exception as e:
            print(f"⚠️  Error accessing PhysioNet: {e}")
            print("   Make sure you have:")
            print("   1. Registered at https://physionet.org/")
            print("   2. Installed wfdb: pip install wfdb")
            return None
            
    except ImportError:
        print("⚠️  wfdb not installed. Install with: pip install wfdb")
        print("   Also register at https://physionet.org/")
        return None

def create_synthetic_fallback():
    """
    Create synthetic wearable data as fallback
    Based on real data statistics
    """
    print("Creating synthetic wearable data (fallback)...")
    
    np.random.seed(42)
    # Increased size for more substantial project
    n_users = 200  # Increased from 50
    n_days = 120   # Increased from 60 (4 months of data)
    
    data = []
    
    for user_id in range(n_users):
        user_baseline_hr = np.random.uniform(55, 75)
        user_sleep_need = np.random.uniform(7, 9)
        user_activity_level = np.random.choice(['low', 'medium', 'high'])
        
        for day in range(n_days):
            from datetime import datetime, timedelta
            date = datetime(2024, 1, 1) + timedelta(days=day)
            
            # Activity
            if user_activity_level == 'low':
                steps = np.random.normal(5000, 1500)
                active_minutes = np.random.normal(20, 10)
            elif user_activity_level == 'medium':
                steps = np.random.normal(8000, 2000)
                active_minutes = np.random.normal(45, 15)
            else:
                steps = np.random.normal(12000, 2500)
                active_minutes = np.random.normal(75, 20)
            
            if date.weekday() >= 5:
                steps *= 0.8
                active_minutes *= 0.7
            
            # HR patterns
            avg_hr = user_baseline_hr + np.random.normal(15, 5)
            hrv = np.random.uniform(40, 80) + np.random.normal(0, 10)
            
            # Sleep (influenced by activity)
            recovery_need = min(1.0, active_minutes / 60)
            sleep_efficiency = np.clip(
                np.random.normal(0.85 - recovery_need*0.15, 0.05),
                0.6, 0.98
            )
            total_sleep = np.clip(
                np.random.normal(user_sleep_need - recovery_need*0.5, 0.8),
                4, 11
            )
            deep_sleep_pct = np.clip(
                np.random.normal(0.18 - recovery_need*0.05, 0.04),
                0.05, 0.30
            )
            
            data.append({
                'user_id': user_id,
                'date': date,
                'day_of_week': date.weekday(),
                'steps': max(0, steps),
                'active_minutes': max(0, active_minutes),
                'daytime_hr_avg': max(50, avg_hr),
                'daytime_hrv': max(20, hrv),
                'sleep_efficiency': sleep_efficiency,
                'total_sleep_time': total_sleep,
                'deep_sleep_pct': deep_sleep_pct,
            })
    
    df = pd.DataFrame(data)
    output_path = DATA_DIR / 'synthetic_wearable_data.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Created synthetic data: {len(df)} records")
    return str(output_path)

def main():
    """
    Main function to download real data
    Tries multiple sources, falls back to synthetic if needed
    """
    print("=" * 60)
    print("DOWNLOADING REAL WEARABLE/SLEEP DATA")
    print("=" * 60)
    print()
    
    downloaded_files = []
    
    # Try Kaggle first (easiest)
    print("1. Trying Kaggle Sleep Health Dataset...")
    kaggle_file = download_kaggle_sleep_health()
    if kaggle_file:
        downloaded_files.append(('kaggle', kaggle_file))
    print()
    
    # Try PhysioNet
    print("2. Trying PhysioNet Sleep-EDF...")
    physionet_file = download_physionet_sleep_edf()
    if physionet_file:
        downloaded_files.append(('physionet', physionet_file))
    print()
    
    # Try Zenodo
    print("3. Trying Zenodo AAUWSS...")
    zenodo_file = download_zenodo_aauwss()
    if zenodo_file:
        downloaded_files.append(('zenodo', zenodo_file))
    print()
    
    # Fallback to synthetic if no real data
    if not downloaded_files:
        print("⚠️  No real data downloaded. Creating synthetic fallback...")
        synthetic_file = create_synthetic_fallback()
        if synthetic_file:
            downloaded_files.append(('synthetic', synthetic_file))
    
    # Summary
    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for source, filepath in downloaded_files:
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✅ {source.upper()}: {os.path.basename(filepath)} ({file_size:.2f} MB)")
    
    print()
    print(f"📁 Data saved to: {DATA_DIR}")
    print()
    print("Next steps:")
    print("1. Review downloaded data")
    print("2. Run preprocessing: python src/preprocess_data.py")
    print("3. Start training: python src/training.py")

if __name__ == '__main__':
    main()

