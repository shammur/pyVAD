#!/bin/bash

#SBATCH -A shchowdhury@hbku.edu.qa
#SBATCH -J VADtest
#SBATCH -o vad_out.txt
#SBATCH -e vad_err.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:0
#SBATCH -c 40 #number of CPUs needed


module load cuda10.1/toolkit gcc6 slurm cmake
source ~/anaconda3/bin/activate ~/anaconda3/envs/vad
#pip install -r ./requirements.txt --ignore-installed


SLURM_SUBMIT_DIR="/alt/asr/shchowdhury/vad/vad_simple_pipeline"


WORK_PATH="/alt/asr/shchowdhury/vad/vad_simple_pipeline"

in=$1
out=$2
THRESHOLD=$3


TASK_ID="VAD_TEST_1"
INPUT_FOLDER=$WORK_PATH"/"$in
OUTPUT_FOLDER=$WORK_PATH"/"$out #data_out_merged/"
SMOOTHING_FACTOR=1.0
WEIGHT_FACTOR=0.4
#MODEL_FILE=

VAD=1


mkdir -p $OUTPUT_FOLDER
if [ $THRESHOLD -eq 0 ]; then

python src/vad_simple.py VAD_simple -i $INPUT_FOLDER -o $OUTPUT_FOLDER --smoothing $SMOOTHING_FACTOR --weight $WEIGHT_FACTOR --classifier /alt/asr/shchowdhury/vad/vad_simple_pipeline/models/svm_rbf_sm
echo "Done Spliting"
else

python src/vad_simple.py VAD_simple_merged -i $INPUT_FOLDER -o $OUTPUT_FOLDER --smoothing $SMOOTHING_FACTOR --weight $WEIGHT_FACTOR --classifier /alt/asr/shchowdhury/vad/vad_simple_pipeline/models/svm_rbf_sm -t $THRESHOLD
echo "Done Spliting and Merged by Threshold"
fi


