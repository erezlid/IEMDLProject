#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 0-10:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name training_model_gaussian_noise			### name of the job
#SBATCH --output job-%J.out		### output log for running job - %J for job number
#SBATCH --gpus=rtx_4090:1	### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1

# Note: the following 4 lines are commented out
##SBATCH --mail-user=user@post.bgu.ac.il	### user's email for sending job status messages
##SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=24G				### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"


### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate CapDec				### activate a conda environment, replace my_env with your conda environment
#jupyter lab					### this command executes jupyter lab – replace with your own command
#python my_code.py my_arg1
python train.py --data /sise/home/erezlid/CapDec/data/coco/verified_split_COCO_train_set_with_text_not_norm.pkl --val_pt /sise/home/erezlid/CapDec/data/coco/COCO_val_set_single_cap_per_sample_with_text_not_norm.pkl --out_dir ./coco_train/hyperparameters_tuning/num_layers_24 --noise_variance 0.05 --num_layers 24

