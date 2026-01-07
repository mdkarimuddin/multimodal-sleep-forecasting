# 🚀 Quick Start Guide

## Step-by-Step Execution

### Step 1: Download Real Data (5-10 minutes)

```bash
cd /scratch/project_2010726/senior_data_scientis_Oura/multimodal-sleep-forecasting

# Download from available sources
python src/download_real_data.py
```

**What it does:**
- Tries to download from Kaggle (easiest)
- Falls back to PhysioNet (requires registration)
- Falls back to Zenodo
- Creates synthetic data if all fail

**Expected output:**
```
✅ kaggle: sleep-health-and-lifestyle-dataset.csv (0.5 MB)
```

### Step 2: Preprocess Data (2-5 minutes)

```bash
python src/preprocess_data.py
```

**What it does:**
- Handles missing values
- Removes artifacts/outliers
- Resamples to regular intervals
- Engineers features (HRV, etc.)

**Expected output:**
```
✅ Preprocessed: 374 rows, 15 columns
```

### Step 3: Train Model (10-30 minutes depending on data size)

```bash
python src/training.py
```

**What it does:**
- Creates train/val/test splits
- Trains multimodal LSTM
- Saves best model checkpoint
- Saves training history

**Expected output:**
```
Training on device: cuda
Model parameters: 45,123
Training samples: 1,234
Validation samples: 264

Epoch 1/50
Train Loss: 0.1234, Val Loss: 0.1456
  → New best model saved!
...
```

### Step 4: Evaluate Model (2-5 minutes)

```bash
python src/evaluation.py
```

**What it does:**
- Loads best model
- Evaluates on test set
- Calculates metrics (MAE, RMSE, R²)
- Creates visualizations

**Expected output:**
```
EVALUATION RESULTS
==================
Sleep Efficiency:
  MAE:  0.0423
  RMSE: 0.0567
  R²:   0.7521
  MAPE: 5.23%
```

## 📊 Expected Results

Based on real data, you should see:

- **Sleep Efficiency**: R² > 0.70, MAE < 0.05
- **Total Sleep Time**: R² > 0.65, MAE < 0.6 hours
- **Deep Sleep %**: R² > 0.60, MAE < 0.03

## 🔧 Troubleshooting

### Issue: "No data files found"
**Solution:** Run `python src/download_real_data.py` first

### Issue: "Module not found"
**Solution:** Install requirements: `pip install -r requirements.txt`

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `training.py` (change `batch_size=32` to `batch_size=16`)

### Issue: "PhysioNet access denied"
**Solution:** 
1. Register at https://physionet.org/
2. Or use Kaggle dataset (easier)

## 📁 Output Files

After running all steps, you'll have:

```
outputs/
├── training_history.json      # Training metrics
├── training_history.png       # Training plot
├── evaluation_results.json    # Test metrics
└── predictions_scatter.png    # Prediction plots

models/
└── best_lstm_forecaster.pt   # Saved model
```

## ✅ Next Steps

1. **Review results** in `outputs/` directory
2. **Experiment** with hyperparameters
3. **Add Transformer** model (see `ENHANCED_PROJECT_PLAN.md`)
4. **Deploy to AWS** (see `aws/` directory)
5. **Push to GitHub** (see `GITHUB_SETUP_GUIDE.md`)

## 🎯 Success Criteria

You've successfully completed the project if:

- ✅ Data downloaded and preprocessed
- ✅ Model trained (validation loss decreasing)
- ✅ Test R² > 0.65 for at least one metric
- ✅ Visualizations created
- ✅ Code is clean and documented

---

**Ready to start? Run Step 1! 🚀**







