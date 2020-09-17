# pyVAD
A simple VAD pipeline based on pyAudioAnalysis
# Running Simple Speech Segmentation
To run the please use python version 3.7, install dependencies

To install the requirements:
```
pip install -r requirements.txt
pip install -r requirements_pyAudio.txt #if needed
```

The pipeline can be used in two ways, either using by giving a maximum duration of segments (merging small segments) or as it is given by the silence based segmentation.

##For getting raw-segmentation (based on silence boundary) use:
`python src/vad_simple.py VAD_simple -i $INPUT_FOLDER -o $OUTPUT_FOLDER --smoothing $SMOOTHING_FACTOR --weight $WEIGHT_FACTOR --classifier /alt/asr/shchowdhury/vad/vad_simple_pipeline/models/svm_rbf_sm`

here: INPUT_FOLDER = folder that has the original audio,
e.g.
$WORK_PATH"/data/"
ls data -->
prep_AToUwCP1wHY.wav  prep_hsjH22Mp6xo.wav  prep_M7fecxjjmL4.wav

OUTPUT_FOLDER = location where one folder per audio (prep_AToUwCP1wHY) is created. Inside prep_AToUwCP1wHY/
segmented audios along with `segments` and `segment_details` file is created

e.g
ls data_out_merged/ -->
prep_AToUwCP1wHY  prep_czhUGrb1Rms
ls data_out_merged/prep_AToUwCP1wHY -->
prep_AToUwCP1wHY_993.550_1003.150.wav
segments
-----------

File format for `segments` file:
<segment_file_name> <original_audio_file_name> <start_dur> <end_dur>
e.g.
prep_AToUwCP1wHY_25.100_34.400 prep_AToUwCP1wHY 25.100 34.400

##For getting merged-segmentation (based on silence boundary and a threshold for maximum duration) use:

`python src/vad_simple.py VAD_simple_merged -i $INPUT_FOLDER -o $OUTPUT_FOLDER --smoothing $SMOOTHING_FACTOR --weight $WEIGHT_FACTOR --classifier /alt/asr/shchowdhury/vad/vad_simple_pipeline/models/svm_rbf_sm -t 10.0`

Options:

`For VAD_simple
"VAD_simple" : "Remove silence segments from a recording in the filder")

-i, "--input", required=True, help="input audio file location"

-o, "--output", required=True, help="output audio file location")
                        
-w, "--weight", type=float, default=0.5, help="weight factor in (0, 1)"

--classifier", required=True, help="Classifier to use (filename)"

-s, "--smoothing", type=float, default=1.0,
                           help="smoothing window size in seconds.")
                           

For VAD_simple_merged:

-i, "--input", required=True, help="input audio file location"

-o, "--output", required=True, help="output audio file location"

    
-w, "--weight", type=float, default=0.5, help="weight factor in (0, 1)

--classifier", required=True, help="Classifier to use (filename)"

-t, --threshold, type=float, default=10.0

-s, "--smoothing", type=float, default=1.0,
                           help="smoothing window size in seconds.")
                           
-t, "--threshold", type=float, default=10.0, help="max segment size in seconds.")`

