#!/bin/sh

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=Antibody_GCNN

## General partitions: all-HiPri, bigmem-HiPri   --   (12 hour limit)
##                     all-LoPri, bigmem-LoPri, gpuq  (5 days limit)
## Restricted: CDS_q, CS_q, STATS_q, HH_q, GA_q, ES_q, COS_q  (10 day limit)
#SBATCH --partition=gpuq 
#SBATCH --gres=gpu:1

## Separate output and error messages into 2 files.
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file

## Slurm can send you updates via email
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=cguerra5@gmu.edu     # Put your GMU email address here

## Specify how much memory your job needs. (2G is the default)
#SBATCH --mem=2G        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=0-12:00  # Total time needed for job: Days-Hours:Minutes


## Load the relevant modules needed for the job
module load cuda/10.1

## Run your program or script
cd /home/cguerra5/graph-protein-distance-prediction/src

epochs=30
lr=0.0001
batch_size=64
num_blocks='10'
h5file='/home/cguerra5/graph-protein-distance-prediction/data/ab_pdbs.h5'
class_weight_file='/home/cguerra5/graph-protein-distance-prediction/data/antibody_weights.p'

lr_str=${lr//\./p}
python -u gcnn_cli.py --epochs $epochs --lr $lr --num_blocks $num_blocks --h5file $h5file --class_weight_file $class_weight_file \
    --save_file /scratch/cguerra5/Antibody_GCNN_epochs${epochs}_lr${lr_str}_batch_size${batch_size}_num_blocks${num_blocks}.p

