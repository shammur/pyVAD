from __future__ import print_function
import argparse
import os
import numpy as np
from collections import deque
import glob
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO
import scipy.io.wavfile as wavfile
# from src import evaluation as eval


def vadFolderWrapper(inputFolder, outFolder, smoothingWindow, weight, model_name):

    if not os.path.isfile(model_name):
        print("fileClassification: input model_name not found!")

    # segfile=open(os.path.join(outFolder,'segments'),'w+')
    # segfile2 = open(os.path.join(outFolder, 'segments_details'), 'w+')

    classifier, mean, std, classes, mid_window, mid_step, short_window, \
    short_step, compute_beat = aT.load_model(model_name)

    types = ('*.wav', '*.mp3')

    wavFilesList = []
    for files in types:
        print(inputFolder + files)
        wavFilesList.extend(glob.glob((inputFolder + files)))
    wavFilesList = sorted(wavFilesList)

    if len(wavFilesList) == 0:
        print("No WAV files found!")
        return
    for wavFile in wavFilesList:
        # print(wavFile)
        if not os.path.isfile(wavFile):
            raise Exception("Input audio file not found!")
        base = os.path.splitext(os.path.basename(wavFile))[0]
        folder=outFolder+base+'/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        segfile = open(os.path.join(folder, 'segments'), 'w+')
        segfile2 = open(os.path.join(folder, 'segments_details'), 'w+')
        stack = deque()

        [fs, x] = audioBasicIO.read_audio_file(wavFile)
        segmentLimits = aS.silence_removal(x, fs, 0.05, 0.05, smoothingWindow, weight, False)

        for i, st in enumerate(segmentLimits):


            signal = audioBasicIO.stereo_to_mono(x[int(fs * st[0]):int(fs * st[1])])
            if fs == 0:
                continue

            if signal.shape[0] / float(fs) < mid_window:
                mid_window = signal.shape[0] / float(fs)

            # feature extraction:
            mid_features, s, _ = \
                aF.mid_feature_extraction(signal, fs,
                                          mid_window * fs,
                                          mid_step * fs,
                                          round(fs * short_window),
                                          round(fs * short_step))
            # long term averaging of mid-term statistics
            mid_features = mid_features.mean(axis=1)
            if compute_beat:
                # print('in here3')
                beat, beat_conf = aF.beat_extraction(s, short_step)
                mid_features = np.append(mid_features, beat)
                mid_features = np.append(mid_features, beat_conf)
            feature_vector = (mid_features - mean) / std  # normalization
            # class_id = -1
            # probability = -1
            class_id = classifier.predict(feature_vector.reshape(1, -1))[0]
            # probability = classifier.predict_proba(feature_vector.reshape(1, -1))[0]
            # print(class_id, type(class_id))
            label=classes[int(class_id)]
            seg = [st[0], st[1], label]
            stack.append(seg)
        for sn in stack:

            strName =base + "_" + "{:.3f}".format(sn[0]) + "_" + "{:.3f}".format(sn[1])
            if sn[2] =='speech':
                strOut = folder + base+"_"+"{:.3f}".format(sn[0])+"_"+"{:.3f}".format(sn[1])+".wav"

                wavfile.write(strOut, fs, x[int(fs * sn[0]):int(fs * sn[1])])
                segfile.write(strName+' '+base+' '+"{:.3f}".format(sn[0])+' '+"{:.3f}".format(sn[1])+"\n")
            segfile2.write(strName + ' ' + "{:.3f}".format(sn[0]) + ' ' + "{:.3f}".format(sn[1]) + ' ' +sn[2] + "\n")

    segfile.close()
    segfile2.close()

def vadFolderWrapperMergedByTh(inputFolder, outFolder, smoothingWindow, weight, model_name, threshold):

    if not os.path.isfile(model_name):
        print("fileClassification: input model_name not found!")



    classifier, mean, std, classes, mid_window, mid_step, short_window, \
    short_step, compute_beat = aT.load_model(model_name)

    types = ('*.wav', '*.mp3')

    wavFilesList = []
    for files in types:
        print(inputFolder + files)
        wavFilesList.extend(glob.glob((inputFolder + files)))
    wavFilesList = sorted(wavFilesList)
    if len(wavFilesList) == 0:
        print("No WAV files found!")
        return
    for wavFile in wavFilesList:
        # print(wavFile)
        if not os.path.isfile(wavFile):
            raise Exception("Input audio file not found!")
        base = os.path.splitext(os.path.basename(wavFile))[0]
        folder = outFolder + base + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        segfile = open(os.path.join(folder, 'segments'), 'w+')
        segfile2 = open(os.path.join(folder, 'segments_details'), 'w+')

        stack = deque()

        [fs, x] = audioBasicIO.read_audio_file(wavFile)
        segmentLimits = aS.silence_removal(x, fs, 0.05, 0.05, smoothingWindow, weight, False)
        merge=True

        for i, st in enumerate(segmentLimits):


            signal = audioBasicIO.stereo_to_mono(x[int(fs * st[0]):int(fs * st[1])])
            # print('in here', len(segmentLimits), st[0],st[1],classes, type(st))
            if fs == 0:
                continue
                # audio file IO problem
                # return -1, -1, -1

            if signal.shape[0] / float(fs) < mid_window:
                mid_window = signal.shape[0] / float(fs)

            # feature extraction:
            mid_features, s, _ = \
                aF.mid_feature_extraction(signal, fs,
                                          mid_window * fs,
                                          mid_step * fs,
                                          round(fs * short_window),
                                          round(fs * short_step))
            # long term averaging of mid-term statistics
            mid_features = mid_features.mean(axis=1)
            if compute_beat:
                # print('in here3')
                beat, beat_conf = aF.beat_extraction(s, short_step)
                mid_features = np.append(mid_features, beat)
                mid_features = np.append(mid_features, beat_conf)
            feature_vector = (mid_features - mean) / std  # normalization
            # class_id = -1
            # probability = -1
            class_id = classifier.predict(feature_vector.reshape(1, -1))[0]
            # probability = classifier.predict_proba(feature_vector.reshape(1, -1))[0]
            print(class_id, type(class_id))
            label=classes[int(class_id)]

            print(label)
            if label=='speech':
                dur=st[1]-st[0]
                # print('in hereas')
                if merge == True:
                    seg_prev=[]
                    # print('in hereasq12')
                    if len(stack) >0:
                        seg_prev = stack.pop()


                    if len(seg_prev) >0 and st[1]-seg_prev[0] > threshold:
                        # print('in hereas4')
                        seg = [st[0], st[1], label]
                        stack.append(seg_prev)
                        stack.append(seg)
                        merge = True
                    elif len(seg_prev) >0:
                        # print('in hereasqw345')
                        seg = [seg_prev[0], st[1], label]
                        stack.append(seg)
                        merge = True
                    else:
                        seg = [st[0], st[1], label]
                        stack.append(seg)
                        merge = True
                else:
                    # print('in hereas2')
                    seg = [st[0], st[1], label]
                    stack.append(seg)
                    merge = True

            else:
                merge = False
            print(i, merge)
        # print(len(segmentLimits), len(stack))
        for sn in stack:
            # print(type(wavFile), sn[0].shape, sn[1].shape, type(sn[0]), type(sn[1]))

            strName = base + "_" + "{:.3f}".format(sn[0]) + "_" + "{:.3f}".format(sn[1])
            if sn[2] == 'speech':
                strOut = folder + base + "_" + "{:.3f}".format(sn[0]) + "_" + "{:.3f}".format(sn[1]) + ".wav"

                wavfile.write(strOut, fs, x[int(fs * sn[0]):int(fs * sn[1])])
                segfile.write(strName + ' ' + base + ' ' + "{:.3f}".format(sn[0]) + ' ' + "{:.3f}".format(sn[1]) + "\n")
            segfile2.write(strName + ' ' + "{:.3f}".format(sn[0]) + ' ' + "{:.3f}".format(sn[1]) + ' ' + sn[2] + "\n")
    segfile.close()
    segfile2.close()


def silenceRemovalWrapper(inputFile, outFolder, smoothingWindow, weight):
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")

    [fs, x] = audioBasicIO.read_audio_file(inputFile)
    segmentLimits = aS.silence_removal(x, fs, 0.05, 0.05,
                                       smoothingWindow, weight, True)
    for i, s in enumerate(segmentLimits):
        strOut = outFolder+"{0:s}_{1:.3f}-{2:.3f}.wav".format(inputFile[0:-4], s[0], s[1])
        wavfile.write(strOut, fs, x[int(fs * s[0]):int(fs * s[1])])


def silenceRemovalFolderWrapper(inputFolder, outFolder, smoothingWindow, weight):
    types = ('*.wav', '*.mp3')
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob((inputFolder + files)))
    wavFilesList = sorted(wavFilesList)
    if len(wavFilesList) == 0:
        print("No WAV files found!")
        return
    for wavFile in wavFilesList:
        silenceRemovalWrapper(wavFile, outFolder, smoothingWindow, weight)


def classifySMFolderWrapper(inputFolder, outputFile, model_type, model_name,
                          outputMode=False):
    if not os.path.isfile(model_name):
        raise Exception("Input model_name not found!")
    types = ('*.wav','*.mp3')
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob((inputFolder + files)))
    wavFilesList = sorted(wavFilesList)
    if len(wavFilesList) == 0:
        print("No WAV files found!")
        return
    Results = []
    for wavFile in wavFilesList:
        [Result, P, classNames] = aT.file_classification(wavFile, model_name,
                                                         model_type)
        Result = int(Result)
        Results.append(Result)
        if outputMode:
            outfile=open(outputFile, 'w+')
            outfile.write("{0:s}\t{1:s}".format(wavFile, classNames[Result])+"\n")
            # print("{0:s}\t{1:s}".format(wavFile, classNames[Result]))
    Results = np.array(Results)

    # print distribution of classes:
    [Histogram, _] = np.histogram(Results,
                                     bins=np.arange(len(classNames) + 1))
    for i, h in enumerate(Histogram):
        print("{0:20s}\t\t{1:d}".format(classNames[i], h))


def classifySMFileWrapper(inputFile, model_type, model_name):
    if not os.path.isfile(model_name):
        raise Exception("Input model_name not found!")
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")

    [Result, P, classNames] = aT.file_classification(inputFile, model_name,
                                                     model_type)
    print("{0:s}\t{1:s}".format("Class", "Probability"))
    for i, c in enumerate(classNames):
        print("{0:s}\t{1:.2f}".format(c, P[i]))
    print("Winner class: " + classNames[int(Result)])


def segmentclassifySMFileWrapper(inputWavFile, model_name, model_type):
    if not os.path.isfile(model_name):
        raise Exception("Input model_name not found!")
    if not os.path.isfile(inputWavFile):
        raise Exception("Input audio file not found!")
    gtFile = ""
    if inputWavFile[-4::]==".wav":
        gtFile = inputWavFile.replace(".wav", ".segments")
    if inputWavFile[-4::]==".mp3":
        gtFile = inputWavFile.replace(".mp3", ".segments")
    aS.mid_term_file_classification(inputWavFile, model_name, model_type, True, gtFile)







def parse_arguments():
    parser = argparse.ArgumentParser(description="A silence removal and S/M classifier script "
                                                 "based on pyAudioAnalysis library")
    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks",
        dest="task", metavar="")

    vadFolder = tasks.add_parser("VAD_simple",help="Remove silence segments from a recording in the filder")
    vadFolder.add_argument("-i", "--input", required=True, help="input audio file location")
    vadFolder.add_argument("-o", "--output", required=True, help="output audio file location")
    vadFolder.add_argument("-s", "--smoothing", type=float, default=1.0,
                        help="smoothing window size in seconds.")
    vadFolder.add_argument("-w", "--weight", type=float, default=0.5,
                        help="weight factor in (0, 1)")
    vadFolder.add_argument("--classifier", required=True,
                             help="Classifier to use (filename)")


    vadFolderM = tasks.add_parser("VAD_simple_merged", help="Remove silence segments from a recording in the filder")
    vadFolderM.add_argument("-i", "--input", required=True, help="input audio file location")
    vadFolderM.add_argument("-o", "--output", required=True, help="output audio file location")
    vadFolderM.add_argument("-s", "--smoothing", type=float, default=1.0,
                           help="smoothing window size in seconds.")
    vadFolderM.add_argument("-w", "--weight", type=float, default=0.5,
                           help="weight factor in (0, 1)")
    vadFolderM.add_argument("--classifier", required=True,
                           help="Classifier to use (filename)")
    vadFolderM.add_argument("-t", "--threshold", type=float, default=10.0,
                           help="max segment size in seconds.")


    classFile = tasks.add_parser("classifyFile",
                                 help="Classify a file using an "
                                      "existing classifier")
    classFile.add_argument("-i", "--input", required=True,
                           help="Input audio file")
    classFile.add_argument("--model", choices=["svm", "svm_rbf", "knn",
                                               "randomforest",
                                               "gradientboosting",
                                               "extratrees"],
                           required=True, help="Classifier type (svm or knn or"
                                               " randomforest or "
                                               "gradientboosting or "
                                               "extratrees)")
    classFile.add_argument("--classifier", required=True,
                           help="Classifier to use (path)")

    classFolder = tasks.add_parser("classifyFolder")
    classFolder.add_argument("-i", "--input", required=True,
                             help="Input folder")
    classFolder.add_argument("-o", "--output", required=True,
                             help="Output folder")
    classFolder.add_argument("--model", choices=["svm", "svm_rbf", "knn",
                                                 "randomforest",
                                                 "gradientboosting",
                                                 "extratrees"],
                             required=True, help="Classifier type")
    classFolder.add_argument("--classifier", required=True,
                             help="Classifier to use (filename)")
    classFolder.add_argument("--details", action="store_true",
                             help="Plot details (otherwise only "
                                  "counts per class are shown)")

    silrem = tasks.add_parser("silenceRemoval",
                              help="Remove silence segments from a recording")
    silrem.add_argument("-i", "--input", required=True, help="input audio file")
    silrem.add_argument("-o", "--output", required=True, help="output audio file location")
    silrem.add_argument("-s", "--smoothing", type=float, default=1.0,
                        help="smoothing window size in seconds.")
    silrem.add_argument("-w", "--weight", type=float, default=0.5,
                        help="weight factor in (0, 1)")

    silremFolder = tasks.add_parser("silenceRemovalFolder",
                              help="Remove silence segments from a recording in the filder")
    silremFolder.add_argument("-i", "--input", required=True, help="input audio file location")
    silremFolder.add_argument("-o", "--output", required=True, help="output audio file location")
    silremFolder.add_argument("-s", "--smoothing", type=float, default=1.0,
                        help="smoothing window size in seconds.")
    silremFolder.add_argument("-w", "--weight", type=float, default=0.5,
                        help="weight factor in (0, 1)")


    segmentationEvaluation = tasks.add_parser("segmentationEvaluation", help=
    "Segmentation - classification "
    "evaluation for a list of WAV "
    "files and CSV ground-truth "
    "stored in a folder")
    segmentationEvaluation.add_argument("-i", "--input", required=True,
                                        help="Input audio folder")
    segmentationEvaluation.add_argument("--model",
                                        choices=["svm", "knn", "hmm"],
                                        required=True, help="Model type")
    segmentationEvaluation.add_argument("--modelName", required=True,
                                        help="Model path")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.task == "VAD_simple":
        vadFolderWrapper(args.input,args.output, args.smoothing, args.weight, args.classifier)
    elif args.task == "VAD_simple_merged":
        vadFolderWrapperMergedByTh(args.input, args.output, args.smoothing, args.weight, args.classifier, args.threshold)
    elif args.task == "silenceRemoval":
        # Detect non-silent segments in a WAV file and
        # output to seperate WAV files
        silenceRemovalWrapper(args.input,args.output, args.smoothing, args.weight)
    elif args.task == "silenceRemovalFolder":
        # Detect non-silent segments in a WAV file and
        # output to seperate WAV files
        silenceRemovalFolderWrapper(args.input, args.output, args.smoothing, args.weight)
    elif args.task == "classifyFile":
        # Apply audio classifier on audio file
        classifySMFileWrapper(args.input, args.model, args.classifier)
    elif args.task == "classifyFolder":
        # Classify every WAV file in a given path
        classifySMFolderWrapper(args.input,args.output, args.model, args.classifier,
                              args.details)
    # elif args.task == "segmentationEvaluation":
    #     # Evaluate segmentation-classification for a list of WAV files
    #     # (and ground truth CSVs) stored in a folder
    #     segmentationEvaluation(args.input, args.modelName, args.model)


