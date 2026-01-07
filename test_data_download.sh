#!/bin/bash
#SBATCH --job-name=test_download
#SBATCH --account=project_2010726
#SBATCH --partition=small
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=outputs/test_download_%j.out
#SBATCH --error=outputs/test_download_%j.err

# Initialize Lmod
source /usr/share/lmod/lmod/init/bash || source /appl/lmod/lmod/init/bash

# Load modules
module load python-data/3.10-22.09 || module load python-data
module load pytorch/2.4 || module load pytorch

# Set Python path
export PYTHONUNBUFFERED=1
PYTHON_CMD=$(which python3)

echo "=========================================="
echo "Testing Data Download on Puhti"
echo "=========================================="
echo "Python: $PYTHON_CMD"
echo "Date: $(date)"
echo "=========================================="
echo ""

cd /scratch/project_2010726/senior_data_scientis_Oura/multimodal-sleep-forecasting

# Test download
echo "Testing data download..."
python3 src/download_real_data.py

echo ""
echo "Checking downloaded files:"
ls -lh data/raw/ 2>/dev/null || echo "No files in data/raw/"

echo ""
echo "Test completed!"







