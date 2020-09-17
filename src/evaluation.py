import os, sys
import numpy as np
import csv,argparse
import collections
from pyAudioAnalysis import audioSegmentation as aS



def read_segmentation_gt(gt_file):
    """
    This function reads a segmentation ground truth file,
    following a simple CSV format with the following columns:
    <segment start>,<segment end>,<class label>

    ARGUMENTS:
     - gt_file:       the path of the CSV segment file
    RETURNS:
     - seg_start:     a np array of segments' start positions
     - seg_end:       a np array of segments' ending positions
     - seg_label:     a list of respective class labels (strings)
    """
    print(gt_file)
    with open(gt_file, 'rt') as f_handle:
        reader = csv.reader(f_handle, delimiter=' ')
        start_end_times = {}

        start_times = []
        end_times = []
        labels = []
        for row in reader:

            if len(row) == 4:
                start_end_times[float(row[2])]=float(row[3])
        sorted_start = collections.OrderedDict(sorted(start_end_times.items()))
        start_point=0.0
        sp_count=len(start_end_times)
        for start, end in sorted_start.items():
            if start!=start_point:
                start_times.append(start_point)
                end_times.append(start)
                labels.append('ns')
                start_times.append(start)
                end_times.append(end)
                labels.append('speech')
                start_point=end
            else:
                start_times.append(start)
                end_times.append(end)
                labels.append('speech')
                start_point = end


        # start_times = []
        # end_times = []
        # labels = []
        # for row in reader:
        #     # if type== 'p' and len(row) == 4:
        #     #     start_times.append(float(row[1]))
        #     #     end_times.append(float(row[2]))
        #     #     labels.append((row[3]))
        #     if len(row) == 4:
        #         start_times.append(float(row[2]))
        #         end_times.append(float(row[3]))
        #         labels.append(('speech'))
    return np.array(start_times), np.array(end_times), labels, sp_count




def XXread_segmentation_gt(gt_file):
    """
    This function reads a segmentation ground truth file,
    following a simple CSV format with the following columns:
    <segment start>,<segment end>,<class label>

    ARGUMENTS:
     - gt_file:       the path of the CSV segment file
    RETURNS:
     - seg_start:     a np array of segments' start positions
     - seg_end:       a np array of segments' ending positions
     - seg_label:     a list of respective class labels (strings)
    """

    with open(gt_file, 'rt') as f_handle:
        reader = csv.reader(f_handle, delimiter='\t')
        start_times = []
        end_times = []
        labels = []
        for row in reader:
            if len(row) == 3:
                start_times.append(float(row[0]))
                end_times.append(float(row[1]))
                labels.append((row[2]))
    return np.array(start_times), np.array(end_times), labels

def segments_to_labels(start_times, end_times, labels, window):
    """
    This function converts segment endpoints and respective segment
    labels to fix-sized class labels.
    ARGUMENTS:
     - start_times:  segment start points (in seconds)
     - end_times:    segment endpoints (in seconds)
     - labels:       segment labels
     - window:      fix-sized window (in seconds)
    RETURNS:
     - flags:    np array of class indices
     - class_names:    list of classnames (strings)
    """
    flags = []
    class_names = list(set(labels))
    index = window / 2.0
    while index < end_times[-1]:
        for i in range(len(start_times)):
            if start_times[i] < index <= end_times[i]:
                break
        flags.append(class_names.index(labels[i]))
        index += window
    return np.array(flags), class_names


def load_ground_truth_segments(gt_file, window):
    """
    This function reads a gold label and predicted labesl,
    following a simple CSV format with the following columns:
    <segment start>,<segment end>,<class label>

    ARGUMENTS:
     - gt_file:       the path of the CSV segment file
     - window:      fix-sized window (in seconds) to segment
    RETURNS:

     """

    seg_start, seg_end, seg_labels, sp_count = read_segmentation_gt(gt_file)
    labels, class_names = segments_to_labels(seg_start, seg_end, seg_labels,
                                             window)
    labels_temp = []
    for index, label in enumerate(labels):
        # "align" labels with GT
        if class_names[labels[index]] in class_names:
            labels_temp.append(class_names.index(class_names[
                                                     labels[index]]))
        else:
            labels_temp.append(-1)
    labels = np.array(labels_temp)
    return labels, class_names, sp_count


def load_ground_truth(gt_file, labels, class_names, mid_step, plot_results):
    accuracy = 0
    cm = np.array([])
    labels_gt = np.array([])
    if os.path.isfile(gt_file):
        # load ground truth and class names
        labels_gt, class_names_gt, sp_count = load_ground_truth_segments(gt_file,
                                                               mid_step)
        # map predicted labels to ground truth class names
        # Note: if a predicted label does not belong to the ground truth
        #       classes --> -1
        labels_new = []
        for il, l in enumerate(labels):
            if class_names[int(l)] in class_names_gt:
                labels_new.append(class_names_gt.index(class_names[int(l)]))
            else:
                labels_new.append(-1)
        labels_new = np.array(labels_new)
        cm = calculate_confusion_matrix(labels_new, labels_gt, class_names_gt)

        accuracy = aS.plot_segmentation_results(labels_new, labels_gt,
                                        class_names, mid_step, not plot_results)
        if accuracy >= 0:
            print("Overall Accuracy: {0:.2f}".format(accuracy))

    return labels_gt, class_names, accuracy, cm, sp_count

def calculate_confusion_matrix(predictions, ground_truth, classes):
    cm = np.zeros((len(classes), len(classes)))
    for index in range(min(predictions.shape[0], ground_truth.shape[0])):
        cm[int(ground_truth[index]), int(predictions[index])] += 1
    return cm

def read_predfile(pred_file, step_window):
    seg_start, seg_end, seg_labels, sp_count_ref = read_segmentation_gt(pred_file)

    labels, class_names = segments_to_labels(seg_start, seg_end, seg_labels, step_window)
    print(class_names)
    return labels, class_names,sp_count_ref


def run_evluation(gt_file, pred_file, step_window):
    labels_pred, class_names_pred, sp_count_pred = read_predfile(pred_file,step_window)
    labels_gt, class_names, accuracy, cm, sp_count_ref= load_ground_truth(gt_file,labels_pred,class_names_pred, step_window, False)
    return accuracy , cm, sp_count_pred, sp_count_ref




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Do Segmentation Evaluation')
    parser.add_argument('-i', '--input', help='Input File name list',
                        required=True)
    parser.add_argument('-o', '--output', help='Outfile',
                        required=True)
    parser.add_argument('-d', '--window',  default=0.5,
                        help="Small window to split audio for evaluation in seconds")
    parser.add_argument('-r', '--reference', help='path for reference',
                        required=True)
    parser.add_argument('-p', '--pred', help='path for predicted segments',
                        required=True)

    args = parser.parse_args()
    filelist=args.input
    outfile=args.output
    refpath = args.reference
    predpath = args.pred
    step_window = float(args.window)

    # gt_file='/alt/asr/shchowdhury/vad/vad_simple_pipeline/ref/prep_AToUwCP1wHY/segments'
    # pred_file='/alt/asr/shchowdhury/vad/vad_simple_pipeline/data_out_raw2/prep_AToUwCP1wHY/segments_details'

    inlist=open(filelist,'r+').readlines()

    fout=open(outfile,'w+')

    for inl in inlist:

        # gt_file,pred_file=inl.split("\t") [0], (inl.split("\t") [1]).strip()

        acc,cm,sp_count_pred,sp_count_ref=run_evluation(refpath+inl.strip()+'/segments',predpath+inl.strip()+'/segments',step_window)
        print(type(cm),cm)
        fout.write(inl+"\n")
        fout.write('Accuracy: '+str(acc) + "\n")
        fout.write('CM: ' + np.array_str(cm, precision = 6, suppress_small = True) + "\n")
        fout.write('Spcounts: ' + str(sp_count_pred) +"\t"+str(sp_count_ref)+ "\n")

    fout.close()




