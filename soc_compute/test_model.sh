#!/bin/bash

#SBATCH --job-name=test_captcha

## Change based on length of job and `sinfo` partitions available
#SBATCH --partition=gpu

## Request for a specific type of node
## Commented out for now, change if you need one
##SBATCH --constraint xgpe

## gpu:1 ==> any gpu. For e.g., gpu:a100-40:1 gets you one of the A100 GPU shared instances
#SBATCH --gres=gpu:a100-40:1

## Must change this based on how long job will take. We are just expecting 30 seconds for now
#SBATCH --time=00:50:00

## Probably no need to change anything here
#SBATCH --ntasks=1

## May want to change this depending on how much host memory you need
## #SBATCH --mem-per-cpu=10G

## Just useful logfile names
#SBATCH --output=captcha%j.slurmlog
#SBATCH --error=captcha%j.slurmlog


echo "Job is running on $(hostname), started at $(date)"

# Get some output about GPU status
nvidia-smi 

source ~/cs4243/remote/cnn_env/bin/activate

python run_test.py

echo -e "\n====> Finished running.\n"

echo -e "\nJob completed at $(date)"
