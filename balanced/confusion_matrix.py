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
    print(method, " ", label_tag, " f1_begin: ", f1_array)

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


def main():
    build_conf_matrix_release('Imbalanced', 'singlelabel')
    build_conf_matrix_release('RandomUnderSampler', 'singlelabel')
    build_conf_matrix_release('NearMiss', 'singlelabel')
    build_conf_matrix_release('EditedNearestNeighbours', 'singlelabel')
    build_conf_matrix_release('TomekLinks', 'singlelabel')
    build_conf_matrix_release('CondensedNearestNeighbour', 'singlelabel')

    build_conf_matrix_release('RandomOverSampler', 'multilabel')
    build_conf_matrix_release('SMOTE', 'multilabel')
    build_conf_matrix_release('ADASYN', 'multilabel')
    build_conf_matrix_release('NearMiss-3', 'multilabel')
    build_conf_matrix_release('RandomUnderSampler', 'multilabel')
    build_conf_matrix_release('EditedNearestNeighbours', 'multilabel')
    build_conf_matrix_release('RepeatedEditedNearestNeighbours', 'multilabel')
    build_conf_matrix_release('AllKNN', 'multilabel')
    build_conf_matrix_release('TomekLinks', 'multilabel')
    build_conf_matrix_release('CondensedNearestNeighbour', 'multilabel')
    build_conf_matrix_release('InstanceHardnessThreshold', 'multilabel')
    build_conf_matrix_release('NeighbourhoodCleaningRule', 'multilabel')
    build_conf_matrix_release('SMOTEENN', 'multilabel')
    build_conf_matrix_release('SMOTETomek', 'multilabel')
    return


if __name__ == "__main__":
    main()
