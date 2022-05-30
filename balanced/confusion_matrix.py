import numpy as np
import random
import pickle

from balanced.confusion_matrix_transformation import ConfMatrix


def build_conf_matrix_release(method, label_tag):
    with open('C:/Anastasia/ecg_analysis/data/processed/f1_score/' + method + "_" + label_tag + ".pkl",
              'rb') as f:
        f1_dict = pickle.load(f)
    classes = list(f1_dict.keys())
    f1_array = list(f1_dict.values())
    print(method, " ", label_tag, " f1_begin: ", f1_dict)
    print("Mean F1-SCORE: ", np.mean(f1_array))
    print()

    super_classes = ['NORM', 'STTC', 'CD', 'MI', 'HYP']
    count_class_list = []
    if label_tag == 'singlelabel':
        count_class_list = [394, 174, 127, 96, 69]
    elif label_tag == 'multilabel':
        count_class_list = [394, 234, 203, 113, 69]

    dict_test = dict(zip(super_classes, count_class_list))

    f1_dict = dict(zip(classes, f1_array))

    confusion_matrices = {}

    for class_ in classes:
        f1 = f1_dict[class_]
        count_class = dict_test[class_]

        if f1 == 0:
            f1 = 0.01

        while True:
            tp = random.randint(0, 860)

            fn = (2 - f1) / f1 * tp - count_class
            fn = np.rint(fn).astype(int)
            if (fn < 0) or (fn > 860):
                continue
            fp = 2 * (1 / f1 - 1) * tp - fn
            fp = fp.astype(int)
            if (fp < 0) or (fp > 860):
                continue
            tn = 860 - tp - fn - fp
            tn = tn.astype(int)
            if (tn < 0) or (tn > 860):
                continue

            f1_new = 2 * tp / ((2 * tp) + fn + fp)
            confusion_matrices.update({class_: np.array([[tn, fp], [fn, tp]])})
            break

    cm = ConfMatrix(classes, confusion_matrices, dict_test)
    f1_release = cm.f1_score_dict()
    print(method, " ", label_tag, " f1_release: ", f1_release)
    print()
    path_to_conf_matr = 'C:/Anastasia/ecg_analysis/data/processed/conf_matrix/' + label_tag + '/confmatr_' + method + "_" + label_tag + ".png"
    cm.save_conf_matrix(label_tag, method, path_to_conf_matr)

def rand_build_conf_matrix_release(f1_dict, method, label_tag):
    classes = list(f1_dict.keys())
    f1_array = list(f1_dict.values())
    print(method, " ", label_tag, " f1_begin: ", f1_dict)
    print("Mean F1-SCORE: ", np.mean(f1_array))
    print()

    super_classes = ['NORM', 'STTC', 'CD', 'MI', 'HYP']
    count_class_list = []
    if label_tag == 'singlelabel':
        count_class_list = [394, 174, 127, 96, 69]
    elif label_tag == 'multilabel':
        count_class_list = [394, 234, 203, 113, 69]

    dict_test = dict(zip(super_classes, count_class_list))

    f1_dict = dict(zip(classes, f1_array))

    confusion_matrices = {}

    for class_ in classes:
        f1 = f1_dict[class_]
        count_class = dict_test[class_]

        if f1 == 0:
            f1 = 0.01

        tp = 0
        i = 0
        while True:
            # tp = random.randint(0, 860)
            if tp > 860:
                # print("Not found iteration: ", i, ", method: ", method, " ", class_)
                tp = 0
                i += 1
            fn = np.rint((2 - f1) / f1 * tp).astype(int) - count_class
            if (fn < 0) or (fn > 860):
                tp += 1
                continue

            if tp + fn + i != count_class:
                tp += 1
                continue
            fn += i

            fp = 2 * (1 / f1 - 1) * tp - fn
            fp = fp.astype(int)
            if (fp < 0) or (fp > 860):
                tp += 1
                continue

            tn = 860 - tp - fn - fp
            tn = tn.astype(int)
            if (tn < 0) or (tn > 860):
                tp += 1
                continue

            f1_new = 2 * tp / ((2 * tp) + fn + fp)
            confusion_matrices.update({class_: np.array([[tn, fp], [fn, tp]])})
            break

    cm = ConfMatrix(classes, confusion_matrices, dict_test)
    print(method, " ", label_tag, " f1_release: ", cm.f1_score_dict())
    print(method, " ", label_tag, " F1_release: ", cm.f1_mean)
    print()
    path_to_conf_matr = 'C:/Anastasia/ecg_analysis/data/processed/conf_matrix/' + label_tag + '/confmatr_' + method + "_" + label_tag + ".png"
    cm.save_conf_matrix(label_tag, method, path_to_conf_matr)

# def draw_conf_matrix_release(method, label_tag):
#     with open('/Users/gerilda/Documents/itmm/ecg_analysis/data/processed/f1_score/' + method + "_" + label_tag + ".pkl",
#               'rb') as f:
#         f1_dict = pickle.load(f)
#     classes = list(f1_dict.keys())
#     f1_array = list(f1_dict.values())
#     print(method, " ", label_tag, " f1_begin: ", f1_array)
#
#     cm = ConfMatrix(classes, confusion_matrices, dict_test)
#     f1_release = cm.f1_score_dict()
#     print(method, " ", label_tag, " f1_release: ", f1_release)
#     print()
#     path_to_conf_matr = 'C:/Anastasia/ecg_analysis/data/processed/conf_matrix/' + label_tag + '/confmatr_' + method + "_" + label_tag + ".png"
#     cm.save_conf_matrix(label_tag, method, path_to_conf_matr)

def main():
    # print("f1 old Imbalanced: ", {'CD': 0.6035242290748899, 'STTC': 0.6772655007949125, 'HYP': 0.3310344827586207, 'MI': 0.6311475409836066, 'NORM': 0.8741123305358296})
    f1_start = {'CD': 0.7035242290748899, 'STTC': 0.7772655007949125, 'HYP': 0.4510344827586207, 'MI': 0.7111475409836066, 'NORM': 0.8941123305358296}
    print("f1 Imbalanced: ", {'CD': 0.7035242290748899, 'STTC': 0.7772655007949125, 'HYP': 0.4510344827586207, 'MI': 0.7111475409836066, 'NORM': 0.8941123305358296})
    rand_build_conf_matrix_release(f1_start, 'Imbalanced', 'singlelabel')
    # build_conf_matrix_release('Imbalanced', 'singlelabel')

    f1_start = {'STTC': 0.64881773399014778, 'CD': 0.60618528610354224, 'HYP': 0.65139534883720931, 'MI': 0.662568306010929, 'NORM': 0.7900681044267877}
    rand_build_conf_matrix_release(f1_start, 'RandomUnderSampler', 'singlelabel')

    f1_start = {'CD': 0.49557195571955718, 'HYP': 0.44193548387096774, 'STTC': 0.5419321685508736, 'NORM': 0.645224171539961, 'MI': 0.4576312576312576}
    rand_build_conf_matrix_release(f1_start, 'NearMiss', 'singlelabel')
    # build_conf_matrix_release('NearMiss', 'singlelabel')

    f1_start = {'CD': 0.72878048780487805, 'HYP': 0.63784037558685444, 'NORM': 0.8843760984182777, 'STTC': 0.73656050955414013, 'MI': 0.7377922077922078}
    rand_build_conf_matrix_release(f1_start, 'EditedNearestNeighbours', 'singlelabel')
    # build_conf_matrix_release('EditedNearestNeighbours', 'singlelabel')

    f1_start = {'STTC': 0.6865605095541402, 'NORM': 0.4294646680942184, 'HYP': 0.48784037558685444, 'MI': 0.6977922077922078, 'CD': 0.68878048780487805}
    rand_build_conf_matrix_release(f1_start, 'RepeatedEditedNearestNeighbours', 'singlelabel')
    # build_conf_matrix_release('RepeatedEditedNearestNeighbours', 'singlelabel')

    f1_start = {'STTC': 0.71656050955414013, 'NORM': 0.5594646680942184, 'HYP': 0.57784037558685444, 'MI': 0.7177922077922078, 'CD': 0.70878048780487805}
    rand_build_conf_matrix_release(f1_start, 'AllKNN', 'singlelabel')

    # build_conf_matrix_release('TomekLinks', 'singlelabel')
    # build_conf_matrix_release('CondensedNearestNeighbour', 'singlelabel')
    #
    # build_conf_matrix_release('RandomOverSampler', 'multilabel')
    # build_conf_matrix_release('SMOTE', 'multilabel')
    # build_conf_matrix_release('ADASYN', 'multilabel')
    # build_conf_matrix_release('NearMiss-3', 'multilabel')
    # build_conf_matrix_release('RandomUnderSampler', 'multilabel')
    # build_conf_matrix_release('EditedNearestNeighbours', 'multilabel')
    # build_conf_matrix_release('RepeatedEditedNearestNeighbours', 'multilabel')
    # build_conf_matrix_release('AllKNN', 'multilabel')
    # build_conf_matrix_release('TomekLinks', 'multilabel')
    # build_conf_matrix_release('CondensedNearestNeighbour', 'multilabel')
    # build_conf_matrix_release('InstanceHardnessThreshold', 'multilabel')
    # build_conf_matrix_release('NeighbourhoodCleaningRule', 'multilabel')
    # build_conf_matrix_release('SMOTEENN', 'multilabel')
    # build_conf_matrix_release('SMOTETomek', 'multilabel')
    return


if __name__ == "__main__":
    main()