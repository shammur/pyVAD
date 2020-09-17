#!/bin/bash

#SBATCH -A shchowdhury@hbku.edu.qa
#SBATCH -J VADtest
#SBATCH -o evalvad_out.txt
#SBATCH -e evalvad_err.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:0
#SBATCH -c 40 #number of CPUs needed


module load cuda10.1/toolkit gcc6 slurm cmake
source ~/anaconda3/bin/activate ~/anaconda3/envs/vad
#pip install -r ./requirements.txt --ignore-installed


SLURM_SUBMIT_DIR="/alt/asr/shchowdhury/vad/vad_simple_pipeline"


WORK_PATH="/alt/asr/shchowdhury/vad/vad_simple_pipeline"

INPUT_FILE=$WORK_PATH'/filelist/wav.lst' #hum_pred_filelist.txt'
#hum_inseg_filelist.txt  hum_pred_filelist.txt  pred_inseg_filelist.txt
OUTFILE=$WORK_PATH'/eval_out/hum_pred_merged_eval.txt'

REF_DIR=$WORK_PATH"/ref/"
PRED_DIR=$WORK_PATH"/"$1
TASK_ID="VAD_TEST_1"
python src/evaluation.py -i $INPUT_FILE -o $OUTFILE  -r $REF_DIR -p $PRED_DIR -d 0.5



