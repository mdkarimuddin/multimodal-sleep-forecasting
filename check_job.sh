#!/bin/bash
# Quick script to check job status and output

JOB_ID=$(squeue -u $USER -h -o %i | head -1)

if [ -z "$JOB_ID" ]; then
    echo "No running jobs found. Checking recent completed jobs..."
    JOB_ID=$(ls -t outputs/slurm_*.out 2>/dev/null | head -1 | grep -o '[0-9]\+' | head -1)
fi

if [ -z "$JOB_ID" ]; then
    echo "No jobs found. Submit a job first with: sbatch run_on_puhti.sh"
    exit 1
fi

echo "Job ID: $JOB_ID"
echo "Status:"
squeue -j $JOB_ID 2>/dev/null || echo "Job completed or not found"

echo ""
echo "Latest output (last 30 lines):"
echo "----------------------------------------"
if [ -f "outputs/slurm_${JOB_ID}.out" ]; then
    tail -30 outputs/slurm_${JOB_ID}.out
else
    echo "Output file not found yet"
fi

echo ""
echo "Latest errors (last 20 lines):"
echo "----------------------------------------"
if [ -f "outputs/slurm_${JOB_ID}.err" ]; then
    tail -20 outputs/slurm_${JOB_ID}.err
else
    echo "Error file not found yet"
fi







