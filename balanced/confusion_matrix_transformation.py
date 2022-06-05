import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from statistics import mean
from abc import ABC


def save_f1_score(classes, confusion_matrices, method, label_tag):
    confusion_matrices_dict = dict(zip(classes, confusion_matrices))
    cm = ConfMatrix(classes, confusion_matrices_dict, dict_test={})

    f1_begin = cm.f1_score_list()
    f1_dict_begin = dict(zip(classes, f1_begin))

    with open('C:/Anastasia/ecg_analysis/data/processed/f1_score/' + method + "_" + label_tag + ".pkl",
              'wb') as f:
        pickle.dump(f1_dict_begin, f)

    print(method, ' ', label_tag, " f1_start: ", f1_dict_begin)
    print("Mean F1-SCORE: ", cm.f1_mean)
    print()

    return

def draw_conf_matrix_release(classes, confusion_matrices, method, label_tag):
    biggest_classes_conf_matrices = zip(classes, confusion_matrices)

    fig, axes = plt.subplots(1, 5, figsize=(20, 7), dpi=87)
    axes = axes.ravel()

    for axe, (title, cf) in zip(axes, biggest_classes_conf_matrices):
        disp = ConfusionMatrixDisplay(cf)
        disp.plot(ax=axe)
        disp.im_.colorbar.remove()
        disp.ax_.set_title(title)

        fig.suptitle(method + "\n" + label_tag, y=0.85, fontsize=16)

    fig.savefig(
        'C:/Anastasia/ecg_analysis/data/processed/conf_matrix/from_f1_dict_file/' + label_tag + '/confmatr_' + method + "_" + label_tag + ".png",
        bbox_inches='tight')


    plt.close()

    return confusion_matrices

def build_conf_matrix_raw(classes, confusion_matrices, method, label_tag):
    # classes = list(dataset.superclasses_mlb.classes_)
    super_classes = ['NORM', 'STTC', 'CD', 'MI', 'HYP']
    value = []
    if label_tag == 'singlelabel':
        value = [394, 174, 127, 96, 69]
    elif label_tag == 'multilabel':
        value = [394, 234, 203, 113, 69]
    # dictionary = dict(zip(super_classes, value))

    confusion_matrices_dict = dict(zip(classes, confusion_matrices))
    cm = ConfMatrix(classes, confusion_matrices_dict, dict_test={})

    f1_begin = cm.f1_score_list()
    f1_dict_begin = dict(zip(classes, f1_begin))

    with open('C:/Anastasia/ecg_analysis/data/processed/f1_score/' + method + "_" + label_tag + ".pkl",
              'wb') as f:
        pickle.dump(f1_dict_begin, f)

    print(method, " f1_start: ", f1_begin)
    print("Mean F1-SCORE: ", cm.f1_mean)
    print()

    count = np.sum(confusion_matrices[0])
    k = 860 / count
    confusion_matrices = np.rint(k * confusion_matrices).astype(int)

    # for i in range(5):
    #     count_new = np.sum(confusion_matrices[i])
    #     dif = 860 - count_new
    #     if np.sum(confusion_matrices[i][1]) != dictionary[classes[i]]:
    #         print("In matrix ", classes[i], ": ", np.sum(confusion_matrices[i][1]),
    #               " in test : ", dictionary[classes[i]], ". Method is ", method, label_tag)
    #     if dif:
    #         print("Not 860", method)
    #     print()


    biggest_classes_conf_matrices = zip(classes, confusion_matrices)

    fig, axes = plt.subplots(1, 5, figsize=(20, 7), dpi=87)
    axes = axes.ravel()

    for axe, (title, cf) in zip(axes, biggest_classes_conf_matrices):
        disp = ConfusionMatrixDisplay(cf)
        disp.plot(ax=axe)
        disp.im_.colorbar.remove()
        disp.ax_.set_title(title)

    fig.savefig(
        'C:/Anastasia/ecg_analysis/data/processed/conf_matrix/raw/' + label_tag + '/confmatrraw_' + f"{str(method)}_" + label_tag + ".png",
        bbox_inches='tight')

    plt.close()

    return confusion_matrices


class ConfMatrix(ABC):
    def __init__(
            self,
            classes: list[str],
            conf_matrix_dict: dict,
            dict_test: dict,
    ) -> None:

        self.classes = classes
        self.dict_test = dict_test
        self.conf_matrix_dict = conf_matrix_dict

    def tn(self, class_) -> int:
        return self.conf_matrix_dict[class_][0][0]

    def fp(self, class_) -> int:
        return self.conf_matrix_dict[class_][0][1]

    def fn(self, class_) -> int:
        return self.conf_matrix_dict[class_][1][0]

    def tp(self, class_) -> int:
        return self.conf_matrix_dict[class_][1][1]

    def tn_increase(self, class_) -> None:
        self.conf_matrix_dict[class_][0][0] += 1

    def fp_increase(self, class_) -> None:
        self.conf_matrix_dict[class_][0][1] += 1

    def fn_increase(self, class_) -> None:
        self.conf_matrix_dict[class_][1][0] += 1

    def tp_increase(self, class_) -> None:
        self.conf_matrix_dict[class_][1][1] += 1

    def tn_decrease(self, class_) -> None:
        self.conf_matrix_dict[class_][0][0] -= 1

    def fp_decrease(self, class_) -> None:
        self.conf_matrix_dict[class_][0][1] -= 1

    def fn_decrease(self, class_) -> None:
        self.conf_matrix_dict[class_][1][0] -= 1

    def tp_decrease(self, class_) -> None:
        self.conf_matrix_dict[class_][1][1] -= 1

    def count_class(self, class_) -> int:
        # tmp = self.conf_matrix_dict[class_]
        # fn = self.fn(class_)
        # tp = self.tp(class_)
        return self.fn(class_) + self.tp(class_)

    def f1_score_list(self) -> list:
        f1_score_list = [2 * self.tp(class_) / (2 * self.tp(class_) + self.fp(class_) + self.fn(class_)) for class_ in self.classes]
        return f1_score_list

    def f1_score_dict(self) -> dict:
        f1_score_dict = {}
        for class_ in self.classes:
            f1_score_dict.update({class_: 2 * self.tp(class_) / (2 * self.tp(class_) + self.fp(class_) + self.fn(class_))})
        return f1_score_dict

    def f1_score(self, class_) -> float:
        return 2 * self.tp(class_) / (2 * self.tp(class_) + self.fp(class_) + self.fn(class_))

    @property
    def f1_mean(self) -> float:
        f1_list = [self.f1_score(class_) for class_ in self.classes]
        return mean(f1_list)

    @property
    def dif_value_dict(self) -> dict:
        dif_value_dict = {}
        for class_ in self.classes:
            value = self.dict_test[class_]
            count = self.count_class(class_)
            dif_value_dict.update({class_: value - count})
        return dif_value_dict

    @property
    def amplitude_diff_class(self):
        amplitude_diff = max(self.dif_value_dict.values(), key=abs)
        class_ = (list(self.dif_value_dict.keys())[list(self.dif_value_dict.values()).index(amplitude_diff)])
        return class_

    @property
    def max_diff_class(self):
        max_diff = max(self.dif_value_dict.values())
        class_ = (list(self.dif_value_dict.keys())[list(self.dif_value_dict.values()).index(max_diff)])
        return class_

    @property
    def min_diff_class(self):
        min_diff = min(self.dif_value_dict.values())
        class_ = (list(self.dif_value_dict.keys())[list(self.dif_value_dict.values()).index(min_diff)])
        return class_

    @property
    def check_exist_negative_diff(self) -> bool:
        for dif in self.dif_value_dict.values():
            if dif < 0:
                return True
        return False

    @property
    def check_exist_positive_diff(self) -> bool:
        for dif in self.dif_value_dict.values():
            if dif > 0:
                return True
        return False

    @property
    def check_sum(self) -> bool:
        flag_list = [np.sum(np.array(self.conf_matrix_dict[class_])) == 860 for class_ in self.classes]

        # for debuging
        for class_ in self.classes:
            flag = sum(self.conf_matrix_dict[class_]) == 860
            if not flag:
                # print("In matrix ", class_, ' sum samples not equals 860: ', sum(self.conf_matrix_dict[class_]))
                return False
        return True

        # return bool(flag_list)

    def check_count_class(self) -> bool:
        flag_list = [self.count_class(class_) == self.dict_test[class_] for class_ in self.classes]

        for class_ in self.classes:
            flag = self.count_class(class_) == self.dict_test[class_]
            if not flag:
                # print("In matrix ", class_, ' count ', self.count_class(class_), " not equals test count: ", self.dict_test[class_])
                return False
        return True

    def save_conf_matrix(self, label_tag, method, path_to_conf_matr):
        tmp = list(self.conf_matrix_dict.values())
        biggest_classes_conf_matrices = zip(self.classes, list(self.conf_matrix_dict.values()))

        fig, axes = plt.subplots(1, 5, figsize=(20, 7), dpi=87)
        axes = axes.ravel()

        for axe, (title, cf) in zip(axes, biggest_classes_conf_matrices):
            disp = ConfusionMatrixDisplay(cf)
            disp.plot(ax=axe)
            disp.im_.colorbar.remove()
            disp.ax_.set_title(title)

        fig.suptitle(method + "\n" + label_tag, y=0.85, fontsize=16)

        fig.savefig(path_to_conf_matr, bbox_inches='tight')

        plt.close()
        return


def build_conf_matrix_release(classes, confusion_matrices, method, label_tag):
    super_classes = ['NORM', 'STTC', 'CD', 'MI', 'HYP']
    value = [394, 234, 203, 113, 69]
    dict_test = dict(zip(super_classes, value))

    confusion_matrices = dict(zip(classes, confusion_matrices))

    cm = ConfMatrix(classes, confusion_matrices, dict_test)

    f1_begin = cm.f1_score_list()
    print(method, " f1_begin: ", f1_begin)

    while not cm.check_count_class():

        amp_dif_class = cm.amplitude_diff_class
        diff = cm.dif_value_dict[amp_dif_class]
        if diff > 0:
            if cm.check_exist_negative_diff:
                min_dif_class = cm.min_diff_class

                debug_matrix_1 = cm.conf_matrix_dict[amp_dif_class]
                debug_matrix_2 = cm.conf_matrix_dict[min_dif_class]
                cm.tp_increase(amp_dif_class)
                cm.tn_decrease(amp_dif_class)
                cm.tp_decrease(min_dif_class)
                cm.tn_increase(min_dif_class)
            else:
                cm.fn_increase(amp_dif_class)
                cm.fp_decrease(amp_dif_class)
        if diff < 0:
            if cm.check_exist_positive_diff:
                max_dif_class = cm.max_diff_class

                cm.tp_decrease(amp_dif_class)
                cm.tn_increase(amp_dif_class)
                cm.tp_increase(max_dif_class)
                cm.tn_decrease(max_dif_class)
            else:
                cm.fn_decrease(amp_dif_class)
                cm.fp_increase(amp_dif_class)

    # for class_ in cm.classes:
    #     tmp1 = cm.count_class(class_)
    #     tmp2 = cm.dict_test[class_]
    #     tmp = 0
    f1_end = cm.f1_score_list()
    print(method, " f1_end: ", f1_end)
    print()

    # delete < 0
    # for class_ in cm.classes:
    #     matr = cm.conf_matrix_dict[class_]
    #     min_ = np.min(cm.conf_matrix_dict[class_])
    #     if min_ < 0:
    #         cm.conf_matrix_dict[class_] -= min_
    #     matr2 = cm.conf_matrix_dict[class_]
    #     tmp = 0
    # f1_after = cm.f1_score_dict()

    for class_ in cm.classes:
        matr = cm.conf_matrix_dict[class_]
        min_ = np.min(cm.conf_matrix_dict[class_])
        if min_ < 0:
            print(method, " have elements < 0")

    path_conf_matr = 'C:/Anastasia/ecg_analysis/data/processed/conf_matrix/transformation/confmatr_' + method + "_" + label_tag + ".png"
    cm.save_conf_matrix(label_tag, method, path_conf_matr)

    return


def main():
    # classes = ['CD', 'STTC', 'HYP', 'MI', 'NORM']
    # confusion_matrices = np.array([
    #     [[1307, 49], [131, 137]],
    #     [[1208, 101], [102, 213]],
    #     [[1503, 1], [96, 24]],
    #     [[1290, 118], [62, 154]],
    #     [[752, 167], [28, 677]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'Imbalanced', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'Imbalanced', 'singlelabel')

    classes = ['CD', 'STTC', 'HYP', 'MI', 'NORM']  # nice
    confusion_matrices = np.array([
        [[658, 75], [39, 88]],
        [[605, 81], [38, 136]],
        [[697, 94], [26, 43]],
        [[686, 78], [21, 75]],
        [[287, 179], [4, 390]]
    ])
    save_f1_score(classes, confusion_matrices, 'RandomOverSampler', 'singlelabel')
    draw_conf_matrix_release(classes, confusion_matrices, 'RandomOverSampler', 'singlelabel')

    # classes = ['STTC', 'HYP', 'MI', 'CD', 'NORM']  # nice
    # confusion_matrices = np.array([
    #     [[650, 36], [35, 139]],
    #     [[761, 30], [25, 44]],
    #     [[704, 60], [24, 72]],
    #     [[517, 216], [13, 114]],
    #     [[307, 159], [0, 394]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'SMOTE', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'SMOTE', 'singlelabel')
    #
    # classes = ['HYP', 'CD', 'STTC', 'MI', 'NORM']  # nice
    # confusion_matrices = np.array([
    #     [[713, 78], [39, 30]],
    #     [[629, 104], [26, 101]],
    #     [[590, 96], [18, 156]],
    #     [[741, 23], [16, 80]],
    #     [[349, 117], [5, 389]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'ADASYN', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'ADASYN', 'singlelabel')
    #
    # classes = ['STTC', 'CD', 'HYP', 'MI', 'NORM']
    # confusion_matrices = np.array([
    #     [[1240, 80], [294, 16]],
    #     [[1299, 67], [228, 36]],
    #     [[1465, 37], [121, 7]],
    #     [[1049, 353], [77, 151]],
    #     [[520, 410], [48, 652]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'RandomUnderSampler', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'RandomUnderSampler', 'singlelabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'RandomUnderSampler', 'singlelabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'RandomUnderSampler', 'singlelabel')
    #
    # classes = ['CD', 'HYP', 'STTC', 'NORM', 'MI']
    # confusion_matrices = np.array([
    #     [[1135, 221], [215, 53]],
    #     [[1480, 24], [109, 11]],
    #     [[866, 443], [100, 215]],
    #     [[234, 685], [43, 662]],
    #     [[197, 1211], [5, 211]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'NearMiss', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'NearMiss', 'singlelabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'NearMiss', 'singlelabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'NearMiss', 'singlelabel')
    #
    # classes = ['CD', 'HYP', 'NORM', 'STTC', 'MI']
    # confusion_matrices = np.array([
    #     [[636, 24], [54, 2]],
    #     [[379, 217], [31, 89]],
    #     [[30, 206], [28, 452]],
    #     [[576, 103], [20, 17]],
    #     [[647, 46], [15, 8]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'EditedNearestNeighbours', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'EditedNearestNeighbours', 'singlelabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'EditedNearestNeighbours', 'singlelabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'EditedNearestNeighbours', 'singlelabel')
    #
    # classes = ['STTC', 'NORM', 'HYP', 'MI', 'CD']  # nice
    # confusion_matrices = np.array([
    #     [[648, 38], [168, 6]],
    #     [[276, 190], [44, 350]],
    #     [[724, 67], [30, 39]],
    #     [[315, 449], [11, 85]],
    #     [[41, 692], [6, 121]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'RepeatedEditedNearestNeighbours', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'RepeatedEditedNearestNeighbours', 'singlelabel')
    #
    # classes = ['CD', 'NORM', 'HYP', 'STTC', 'MI']  # nice
    # confusion_matrices = np.array([
    #     [[718, 15], [118, 9]],
    #     [[319, 147], [60, 334]],
    #     [[761, 30], [52, 17]],
    #     [[325, 361], [20, 154]],
    #     [[414, 350], [9, 87]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'AllKNN', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'AllKNN', 'singlelabel')
    #
    # classes = ['HYP', 'CD', 'MI', 'STTC', 'NORM']
    # confusion_matrices = np.array([
    #     [[1312, 184], [69, 51]],
    #     [[1171, 177], [57, 211]],
    #     [[1216, 184], [31, 185]],
    #     [[1119, 182], [29, 286]],
    #     [[714, 205], [13, 684]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'TomekLinks', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'TomekLinks', 'singlelabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'TomekLinks', 'singlelabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'TomekLinks', 'singlelabel')
    #
    # classes = ['HYP', 'NORM', 'STTC', 'CD', 'MI']
    # confusion_matrices = np.array([
    #     [[187, 39], [31, 89]],
    #     [[251, 32], [31, 32]],
    #     [[272, 43], [26, 5]],
    #     [[122, 127], [15, 82]],
    #     [[218, 93], [10, 25]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'CondensedNearestNeighbour', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'CondensedNearestNeighbour', 'singlelabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'CondensedNearestNeighbour', 'singlelabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'CondensedNearestNeighbour', 'singlelabel')
    #
    # classes = ['CD', 'NORM', 'HYP', 'STTC', 'MI']  # nice
    # confusion_matrices = np.array([
    #     [[698, 35], [117, 10]],
    #     [[290, 176], [71, 323]],
    #     [[777, 14], [65, 4]],
    #     [[392, 294], [27, 147]],
    #     [[419, 345], [9, 87]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'InstanceHardnessThreshold', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'InstanceHardnessThreshold', 'singlelabel')
    #
    # classes = ['CD', 'STTC', 'NORM', 'HYP', 'MI']  # nice
    # confusion_matrices = np.array([
    #     [[718, 15], [124, 3]],
    #     [[477, 209], [107, 67]],
    #     [[295, 171], [88, 306]],
    #     [[777, 14], [65, 4]],
    #     [[354, 410], [7, 89]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'OneSidedSelection', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'OneSidedSelection', 'singlelabel')
    #
    # classes = ['NORM', 'HYP', 'CD', 'MI', 'STTC']  # nice
    # confusion_matrices = np.array([
    #     [[373, 93], [70, 324]],
    #     [[790, 1], [63, 6]],
    #     [[469, 264], [36, 91]],
    #     [[548, 216], [11, 85]],
    #     [[280, 406], [5, 169]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'NeighbourhoodCleaningRule', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'NeighbourhoodCleaningRule', 'singlelabel')
    #
    # classes = ['MI', 'NORM', 'CD', 'STTC', 'HYP']  # nice
    # confusion_matrices = np.array([
    #     [[650, 114], [78, 18]],
    #     [[131, 335], [54, 340]],
    #     [[110, 623], [28, 99]],
    #     [[115, 571], [26, 148]],
    #     [[134, 657], [15, 54]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'SMOTEENN', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'SMOTEENN', 'singlelabel')
    #
    # classes = ['NORM', 'STTC', 'MI', 'CD', 'HYP']  # nice
    # confusion_matrices = np.array([
    #     [[405, 61], [363, 31]],
    #     [[567, 119], [148, 26]],
    #     [[739, 25], [90, 6]],
    #     [[578, 155], [89, 38]],
    #     [[697, 94], [58, 11]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'SMOTETomek', 'singlelabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'SMOTETomek', 'singlelabel')
    #
    # ################################################# MULTILABEL
    #
    # classes = ['HYP', 'CD', 'STTC', 'MI', 'NORM']  # nice
    # confusion_matrices = np.array([
    #     [[790, 1], [58, 11]],
    #     [[615, 42], [54, 149]],
    #     [[546, 80], [51, 183]],
    #     [[715, 32], [47, 66]],
    #     [[387, 79], [31, 363]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'Imbalanced', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'Imbalanced', 'multilabel')
    #
    # # # multilabel
    # classes = ['STTC', 'NORM', 'HYP', 'MI', 'CD']
    # confusion_matrices = np.array([
    #     [[1193, 5], [327, 99]],
    #     [[869, 50], [195, 510]],
    #     [[1355, 149], [87, 33]],
    #     [[1312, 66], [83, 163]],
    #     [[751, 439], [45, 389]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'RandomOverSampler', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'RandomOverSampler', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'RandomOverSampler', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'RandomOverSampler', 'multilabel')
    #
    # classes = ['CD', 'NORM', 'HYP', 'MI', 'STTC']
    # confusion_matrices = np.array([
    #     [[41674, 3681], [36208, 8792]],
    #     [[84724, 0], [5631, 0]],
    #     [[1902, 43274], [1919, 43260]],
    #     [[0, 44881], [0, 45474]],
    #     [[0, 45358], [0, 44997]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'SMOTE', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'SMOTE', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'SMOTE', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'SMOTE', 'multilabel')
    #
    # classes = ['CD', 'NORM', 'HYP', 'MI', 'STTC']
    # confusion_matrices = np.array([
    #     [[41137, 4218], [37700, 7300]],
    #     [[84700, 0], [5631, 0]],
    #     [[1509, 43667], [2135, 43044]],
    #     [[95, 44786], [30, 45444]],
    #     [[0, 45358], [0, 44997]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'ADASYN', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'ADASYN', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'ADASYN', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'ADASYN', 'multilabel')
    #
    # classes = ['HYP', 'CD', 'STTC', 'MI', 'NORM']
    # confusion_matrices = np.array([
    #     [[282, 14], [195, 101]],
    #     [[195, 101], [63, 233]],
    #     [[151, 145], [55, 241]],
    #     [[155, 141], [33, 263]],
    #     [[446, 109], [7, 30]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'RandomUnderSampler', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'RandomUnderSampler', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'RandomUnderSampler', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'RandomUnderSampler', 'multilabel')
    #
    # classes = ['HYP', 'STTC', 'MI', 'CD', 'NORM']
    # confusion_matrices = np.array([
    #     [[254, 0], [140, 6]],
    #     [[116, 104], [40, 140]],
    #     [[100, 117], [6, 177]],
    #     [[24, 211], [4, 161]],
    #     [[305, 58], [2, 35]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'NearMiss-3', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'NearMiss-3', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'NearMiss-3', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'NearMiss_3', 'multilabel')
    #
    # classes = ['MI', 'CD', 'NORM', 'STTC', 'HYP']
    # confusion_matrices = np.array([
    #     [[487, 12], [10, 33]],
    #     [[529, 3], [8, 2]],
    #     [[22, 31], [4, 485]],
    #     [[532, 8], [2, 0]],
    #     [[493, 12], [1, 36]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'EditedNearestNeighbours', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'EditedNearestNeighbours', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'EditedNearestNeighbours', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'EditedNearestNeighbours', 'multilabel')
    # #
    # classes = ['MI', 'NORM', 'HYP', 'CD', 'STTC']
    # confusion_matrices = np.array([
    #     [[641, 11], [49, 159]],
    #     [[192, 22], [49, 597]],
    #     [[635, 22], [44, 159]],
    #     [[378, 477], [0, 5]],
    #     [[99, 761], [0, 0]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'RepeatedEditedNearestNeighbours', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'RepeatedEditedNearestNeighbours', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'RepeatedEditedNearestNeighbours', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'RepeatedEditedNearestNeighbours', 'multilabel')
    #
    # classes = ['MI', 'CD', 'STTC', 'NORM', 'HYP']
    # confusion_matrices = np.array([
    #     [[2713, 31], [354, 47]],
    #     [[2767, 54], [304, 20]],
    #     [[2851, 0], [294, 0]],
    #     [[550, 304], [75, 2216]],
    #     [[3041, 55], [47, 2]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'AllKNN', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'AllKNN', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'AllKNN', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'AllKNN', 'multilabel')
    #
    # classes = ['STTC', 'HYP', 'CD', 'MI', 'NORM']
    # confusion_matrices = np.array([
    #     [[9357, 209], [1105, 2173]],
    #     [[11777, 0], [891, 176]],
    #     [[9268, 234], [672, 2670]],
    #     [[10416, 286], [417, 1725]],
    #     [[6648, 618], [271, 5307]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'TomekLinks', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'TomekLinks', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'TomekLinks', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'TomekLinks', 'multilabel')
    #
    # classes = ['HYP', 'MI', 'STTC', 'NORM', 'CD']
    # confusion_matrices = np.array([
    #     [[376, 4], [398, 105]],
    #     [[493, 108], [154, 128]],
    #     [[34, 402], [69, 378]],
    #     [[852, 14], [11, 6]],
    #     [[13, 400], [5, 465]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'CondensedNearestNeighbour', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'CondensedNearestNeighbour', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'CondensedNearestNeighbour', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'CondensedNearestNeighbour', 'multilabel')
    #
    # classes = ['HYP', 'STTC', 'CD', 'MI', 'NORM']
    # confusion_matrices = np.array([
    #     [[287, 9], [268, 28]],
    #     [[260, 36], [206, 90]],
    #     [[128, 168], [36, 260]],
    #     [[164, 132], [35, 261]],
    #     [[446, 109], [1, 36]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'InstanceHardnessThreshold', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'InstanceHardnessThreshold', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'InstanceHardnessThreshold', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'InstanceHardnessThreshold', 'multilabel')
    #
    # classes = ['HYP', 'CD', 'STTC', 'MI', 'NORM']  # nice
    # confusion_matrices = np.array([
    #     [[645, 12], [186, 17]],
    #     [[787, 4], [65, 4]],
    #     [[629, 118], [42, 71]],
    #     [[226, 240], [12, 382]],
    #     [[291, 335], [9, 225]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'OneSidedSelection', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'OneSidedSelection', 'multilabel')
    #
    # classes = ['CD', 'HYP', 'NORM', 'STTC', 'MI']
    # confusion_matrices = np.array([
    #     [[2234, 2], [139, 505]],
    #     [[2831, 5], [32, 12]],
    #     [[986, 55], [27, 1812]],
    #     [[2774, 40], [7, 59]],
    #     [[2035, 267], [4, 574]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'NeighbourhoodCleaningRule', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'NeighbourhoodCleaningRule', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'NeighbourhoodCleaningRule', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'NeighbourhoodCleaningRule', 'multilabel')
    #
    # classes = ['NORM', 'CD', 'HYP', 'MI', 'STTC']
    # confusion_matrices = np.array([
    #     [[263, 391], [203, 3]],
    #     [[640, 75], [18, 127]],
    #     [[740, 6], [51, 63]],
    #     [[643, 0], [104, 113]],
    #     [[490, 19], [137, 214]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'SMOTEENN', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'SMOTEENN', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'SMOTEENN', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'SMOTEENN', 'multilabel')
    #
    # classes = ['CD', 'NORM', 'HYP', 'MI', 'STTC']
    # confusion_matrices = np.array([
    #     [[44506, 536], [44214, 830]],
    #     [[84456, 0], [5630, 0]],
    #     [[0, 45038], [0, 45048]],
    #     [[0, 45041], [0, 45045]],
    #     [[0, 45042], [0, 45044]]
    # ])
    # save_f1_score(classes, confusion_matrices, 'SMOTETomek', 'multilabel')
    # draw_conf_matrix_release(classes, confusion_matrices, 'SMOTETomek', 'multilabel')
    # # confusion_matrices_raw = build_conf_matrix_raw(classes, confusion_matrices, 'SMOTETomek', 'multilabel')
    # # build_conf_matrix_release(classes, confusion_matrices_raw, 'SMOTETomek', 'multilabel')

    return


if __name__ == "__main__":
    main()
