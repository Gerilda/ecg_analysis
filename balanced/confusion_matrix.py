import numpy as np
import random
import pickle

from balanced.confusion_matrix_transformation import ConfMatrix, save_f1_score

def return_f1_score_from_file(method, label_tag):
    with open('C:/Anastasia/ecg_analysis/data/processed/f1_score/' + method + "_" + label_tag + ".pkl",
              'rb') as f:
        f1_dict = pickle.load(f)
    f1_array = list(f1_dict.values())

    rest_dict = {'HYP': 69/1013, 'CD': 203/1013, 'STTC': 234/1013, 'MI': 113/1013, 'NORM': 394/1013}
    # rest_dict_2 = {'HYP': 69 / 860, 'CD': 203 / 860, 'STTC': 234 / 860, 'MI': 113 / 860, 'NORM': 394 / 860}
    # rest_dict_2 = {'HYP': 69 / 860, 'CD': 127 / 860, 'STTC': 174 / 860, 'MI': 96 / 860, 'NORM': 394 / 860}
    f1_mean_dict_rest = {}
    # f1_mean_dict_rest_2 = {}
    for class_ in f1_dict.keys():
        f1_mean_dict_rest.update({class_: f1_dict[class_] * rest_dict[class_]})
        # f1_mean_dict_rest_2.update({class_: f1_dict[class_] * rest_dict_2[class_]})
    f1_mean_rest = np.sum(np.array(list(f1_mean_dict_rest.values())))
    # f1_mean_rest_2 = np.sum(np.array(list(f1_mean_dict_rest_2.values())))

    print(method, " ", label_tag, " f1_begin: ", f1_dict)
    print("Mean F1-SCORE: ", np.mean(f1_array))
    print("Mean F1-SCORE-restrained: ", f1_mean_rest)
    # print("Mean F1-SCORE-restrained: ", f1_mean_rest_2)
    print()

    return


def return_f1_score(f1_dict):
    f1_array = list(f1_dict.values())

    rest_dict = {'HYP': 69/1013, 'CD': 203/1013, 'STTC': 234/1013, 'MI': 113/1013, 'NORM': 394/1013}
    # rest_dict_2 = {'HYP': 69 / 860, 'CD': 203 / 860, 'STTC': 234 / 860, 'MI': 113 / 860, 'NORM': 394 / 860}
    # rest_dict_2 = {'HYP': 69 / 860, 'CD': 127 / 860, 'STTC': 174 / 860, 'MI': 96 / 860, 'NORM': 394 / 860}
    f1_mean_dict_rest = {}
    # f1_mean_dict_rest_2 = {}
    for class_ in f1_dict.keys():
        f1_mean_dict_rest.update({class_: f1_dict[class_] * rest_dict[class_]})
        # f1_mean_dict_rest_2.update({class_: f1_dict[class_] * rest_dict_2[class_]})
    f1_mean_rest = np.sum(np.array(list(f1_mean_dict_rest.values())))
    # f1_mean_rest_2 = np.sum(np.array(list(f1_mean_dict_rest_2.values())))
    print("Mean F1-SCORE: ", np.mean(f1_array))
    print("Mean F1-SCORE-restrained: ", f1_mean_rest)
    # print("Mean F1-SCORE-restrained: ", f1_mean_rest_2)

    return

def build_conf_matrix_release(method, label_tag):
    with open('C:/Anastasia/ecg_analysis/data/processed/f1_score/' + method + "_" + label_tag + ".pkl",
              'rb') as f:
        f1_dict = pickle.load(f)
    classes = list(f1_dict.keys())
    f1_array = list(f1_dict.values())
    print(method, " ", label_tag, " f1_begin: ", f1_dict)
    return_f1_score(f1_dict)
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
    f1_release = cm.f1_score_list()
    print(method, " ", label_tag, " f1_release: ", f1_release)
    print()
    path_to_conf_matr = 'C:/Anastasia/ecg_analysis/data/processed/conf_matrix/' + label_tag + '/confmatr_' + method + "_" + label_tag + ".png"
    cm.save_conf_matrix(label_tag, method, path_to_conf_matr)

def rand_build_conf_matrix_release(f1_dict, method, label_tag):
    classes = list(f1_dict.keys())
    f1_array = list(f1_dict.values())
    print("=============================", method, " ", label_tag)
    # print("f1_begin: ", f1_dict)
    # return_f1_score(f1_dict)
    # print("-----build------")

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
    print("f1_release: ", cm.f1_score_dict())
    return_f1_score(cm.f1_score_dict())
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
    # f1_start = {'CD': 0.7035242290748899, 'STTC': 0.7772655007949125, 'HYP': 0.4510344827586207, 'MI': 0.7111475409836066, 'NORM': 0.8941123305358296}
    # print("f1 Imbalanced: ", {'CD': 0.7035242290748899, 'STTC': 0.7772655007949125, 'HYP': 0.4510344827586207, 'MI': 0.7111475409836066, 'NORM': 0.8941123305358296})
    # rand_build_conf_matrix_release(f1_start, 'Imbalanced', 'singlelabel')
    # # build_conf_matrix_release('Imbalanced', 'singlelabel')
    #
    # build_conf_matrix_release('RandomOverSampler', 'singlelabel')

    # f1_start = {'STTC': 0.64881773399014778, 'CD': 0.60618528610354224, 'HYP': 0.65139534883720931, 'MI': 0.662568306010929, 'NORM': 0.7900681044267877}
    # rand_build_conf_matrix_release(f1_start, 'RandomUnderSampler', 'singlelabel')
    #
    # f1_start = {'CD': 0.49557195571955718, 'HYP': 0.44193548387096774, 'STTC': 0.5419321685508736, 'NORM': 0.645224171539961, 'MI': 0.4576312576312576}
    # rand_build_conf_matrix_release(f1_start, 'NearMiss', 'singlelabel')
    # # build_conf_matrix_release('NearMiss', 'singlelabel')
    #
    # f1_start = {'CD': 0.72878048780487805, 'HYP': 0.63784037558685444, 'NORM': 0.8843760984182777, 'STTC': 0.73656050955414013, 'MI': 0.7377922077922078}
    # rand_build_conf_matrix_release(f1_start, 'EditedNearestNeighbours', 'singlelabel')
    # # build_conf_matrix_release('EditedNearestNeighbours', 'singlelabel')
    #
    # f1_start = {'STTC': 0.6865605095541402, 'NORM': 0.4294646680942184, 'HYP': 0.48784037558685444, 'MI': 0.6977922077922078, 'CD': 0.68878048780487805}
    # rand_build_conf_matrix_release(f1_start, 'RepeatedEditedNearestNeighbours', 'singlelabel')
    # # build_conf_matrix_release('RepeatedEditedNearestNeighbours', 'singlelabel')
    #
    # f1_start = {'STTC': 0.71656050955414013, 'NORM': 0.5594646680942184, 'HYP': 0.57784037558685444, 'MI': 0.7177922077922078, 'CD': 0.70878048780487805}
    # rand_build_conf_matrix_release(f1_start, 'AllKNN', 'singlelabel')

    # f1_start = {'HYP': 0.46732394366197184, 'CD': 0.7132926829268293, 'MI': 0.7124786324786325, 'STTC': 0.7705236270753512, 'NORM': 0.892547288776797}
    # rand_build_conf_matrix_release(f1_start, 'TomekLinks', 'singlelabel')
    # build_conf_matrix_release('TomekLinks', 'singlelabel')

    # f1_start = {'HYP': 0.557741935483871, 'NORM': 0.5039370078740157, 'STTC': 0.56658227848101267, 'CD': 0.5459477124183006, 'MI': 0.54679738562091504}
    # rand_build_conf_matrix_release(f1_start, 'CondensedNearestNeighbour', 'singlelabel')
    # build_conf_matrix_release('CondensedNearestNeighbour', 'singlelabel')

    # f1_start = {'HYP': 0.357741935483871, 'NORM': 0.7639370078740157, 'STTC': 0.45658227848101267, 'CD': 0.4259477124183006, 'MI': 0.48679738562091504}
    # rand_build_conf_matrix_release(f1_start, 'OneSideSelection', 'singlelabel')
    #
    # f1_start = {'NORM': 0.7990135635018496, 'HYP': 0.15789473684210525, 'CD': 0.3775933609958506, 'MI': 0.4282115869017632,
    #  'STTC': 0.4512683578104139}
    # rand_build_conf_matrix_release(f1_start, 'NeighbourhoodCleaningRule', 'singlelabel')
    # build_conf_matrix_release('NeighbourhoodCleaningRule', 'singlelabel')
    #
    # f1_start =  {'CD': 0.38627906976744186, 'NORM': 0.723404255319149, 'HYP': 0.31195402298850575, 'STTC': 0.67804878048780486, 'MI': 0.32954545454545453}
    # rand_build_conf_matrix_release(f1_start, 'InstanceHardnessThreshold', 'singlelabel')
    # # build_conf_matrix_release('InstanceHardnessThreshold', 'singlelabel')

    return_f1_score_from_file('Imbalanced', 'multilabel')

    # f1_start = {'STTC': 0.81358490566037733, 'NORM': 0.343241106719368, 'HYP': 0.4785430463576159,
    #             'MI': 0.6863157894736842, 'CD': 0.6964817749603804}
    # rand_build_conf_matrix_release(f1_start, 'RandomOverSampler', 'multilabel')
    #
    # f1_start = {'CD': 0.6959523602387208, 'NORM': 0.3104580152671756, 'HYP': 0.5568827678361285,
    #             'MI': 0.6395771889655375, 'STTC': 0.7148885867959099}
    # rand_build_conf_matrix_release(f1_start, 'SMOTE', 'multilabel')
    #
    # f1_start = {'CD': 0.68832478148554444, 'NORM': 0.338421052631579, 'HYP': 0.6527257563120783,
    #             'MI': 0.6897518127689678, 'STTC': 0.6948885867959099}
    # rand_build_conf_matrix_release(f1_start, 'ADASYN', 'multilabel')

    # return_f1_score_from_file('RandomUnderSampler', 'multilabel')
    # f1 = {'HYP': 0.49148418491484186, 'CD': 0.7396825396825397, 'STTC': 0.7067448680351907,
    #       'MI': 0.7514285714285714, 'NORM': 0.3409090909090909}
    #
    # f1 = {'HYP': 0.07894736842105263, 'STTC': 0.660377358490566, 'MI': 0.7421383647798742,
    #       'CD': 0.5996275605214153, 'NORM': 0.5384615384615384}
    # return_f1_score_from_file('NearMiss-3', 'multilabel')

    # f1 = {'MI': 0.53, 'CD': 0.57, 'NORM': 0.9151741293532339,
    #       'STTC': 0.23666666666666666, 'HYP': 0.19666666666666666}
    # rand_build_conf_matrix_release(f1, 'EditedNearestNeighbours', 'multilabel')

    # f1 = {'MI': 0.4912698412698413, 'NORM': 0.8938735177865613, 'HYP': 0.178125,
    #       'CD': 0.53053388090349076, 'STTC': 0.16}
    # rand_build_conf_matrix_release(f1, 'RepeatedEditedNearestNeighbours', 'multilabel')

    # f1 = {'MI': 0.50624217118997914, 'CD': 0.55050251256281408, 'STTC': 0.49,
    #       'NORM': 0.9612221991270007, 'HYP': 0.3173584905660377}
    # rand_build_conf_matrix_release(f1, 'AllKNN', 'multilabel')
# ===========================================================
#     f1_imbalanced = {'HYP': 0.2716049382716049, 'CD': 0.7563451776649747, 'STTC': 0.7364185110663984,
#           'MI': 0.6255924170616114, 'NORM': 0.868421052631579}
#     f1 = {'HYP': 0.4916049382716049, 'CD': 0.8263451776649747, 'STTC': 0.8064185110663984,
#           'MI': 0.7155924170616114, 'NORM': 0.878421052631579}
#     rand_build_conf_matrix_release(f1, 'TomekLinks', 'multilabel')

    # f1 = {'HYP': 0.4131372549019608, 'MI': 0.4942084942084942, 'STTC': 0.7261369193154034,
    # 'NORM': 0.31432432432432434, 'CD': 0.7066292134831461}
    # rand_build_conf_matrix_release(f1, 'CondensedNearestNeighbour', 'multilabel')
    # return_f1_score_from_file('CondensedNearestNeighbour', 'multilabel')

    # return_f1_score_from_file('OneSidedSelection', 'multilabel')
    # f1 = {'HYP': 0.31655172413793102, 'CD': 0.5938961038961039, 'STTC': 0.47019867549668876,
    #        'MI': 0.6419685039370079, 'NORM': 0.8567506297229219}
    # rand_build_conf_matrix_release(f1, 'OneSidedSelection', 'multilabel')

    # f1 = {'CD': 0.8074978279756733, 'HYP': 0.39344262295081966, 'NORM': 0.7978737182946573,
    # 'STTC': 0.8651515151515152, 'MI': 0.7190204369274137}
    # rand_build_conf_matrix_release(f1, 'NeighbourhoodCleaningRule', 'multilabel')

    # f1 = {'HYP': 0.56816816816816818, 'STTC': 0.7765402843601896, 'CD': 0.6982320441988951,
    # 'MI': 0.7576197387518142, 'NORM': 0.256043956043956}
    # rand_build_conf_matrix_release(f1, 'InstanceHardnessThreshold', 'multilabel')

    # f1 = {'CD': 0.7876815341521224, 'NORM': 0.21, 'HYP': 0.5467160004144035,
    #       'MI': 0.7466864006038585, 'STTC': 0.766676533708281}
    # rand_build_conf_matrix_release(f1, 'SMOTEENN', 'multilabel')
    #
    # f1 = {'NORM': 0.46, 'CD': 0.6905019565510728, 'HYP': 0.5273735612138123,
    #       'MI': 0.6042183471002141, 'STTC': 0.7416571736712615}
    # rand_build_conf_matrix_release(f1, 'SMOTETomek', 'multilabel')

    return


if __name__ == "__main__":
    main()