import os
import ast
import sys

# import seaborn as sns
from functools import partial
from collections import Counter
from datetime import datetime

from balanced.balanced_classification import balanced_classification
from ecg_analysis.dataset import PtbXlClassesSuperclasses, PtbXl

import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, \
    OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold, \
    TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.linear_model import LogisticRegression

from ecg_analysis.dataset import PtbXlClassesSuperclasses

import matplotlib.pyplot as plt


def counter_dict_class(tabular_column):
    """Creates the counter of classes in the tabular by column"""

    all_labels = []
    if type(tabular_column[0]) is tuple or type(tabular_column[0]) is list:
        all_labels = [label for labels in tabular_column for label in labels]
    elif type(tabular_column[0]) is str:
        all_labels = [label for label in tabular_column]
    return Counter(all_labels)


def one_label_preparing(probs, counter):
    residuary_prob = probs
    tmp = sys.maxsize

    for prob in probs:
        current = counter[prob]
        if current < tmp:
            tmp = current
            residuary_prob = prob

    return (residuary_prob,)


def create_pie(classes, method, label_tag, class_tag, processed_data_folder):
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]

    fig, ax = plt.subplots()

    ax.set(aspect="equal", title=f"{str(method)}\n" + label_tag + ", " + class_tag)

    # ax.title(f"{str(method)}\n" + label_tag + ", " + class_tag)

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{v:d}'.format(v=val)

        return my_autopct

    ax.pie(classes.values(),
           labels=classes.keys(),
           # autopct='%1.1f%%',
           autopct=make_autopct(classes.values()),
           shadow=True,
           colors=colors,
           wedgeprops={"edgecolor": "black",
                       'linewidth': 1,
                       'antialiased': True}
           )

    pie_label_tag = "single_label" if label_tag == "Single-label" else "multi_label"

    fig.savefig(
        processed_data_folder + r"/pies/multilabel/pie_" + f"{str(method)}_" + pie_label_tag + "_" + class_tag + ".png",
        bbox_inches='tight')

    ax.clear()

    return plt


class PtbXlClassesSuperclassesMultilabelBalanced(PtbXlClassesSuperclasses):
    def __init__(
            self,
            raw_data_folder: str,
            processed_data_folder: str,
            ptbxl_dataset_filename: str,
            scp_statements_filename: str,
            classes_mlb_filename: str,
            superclasses_mlb_filenames: str,
            tabular_filename: str,
            waves_filename: str,
            threshold: int,
            sampling_rate: int = 100,
            batch_size: int = 64,
            balanced_batch_size: int = 128
    ) -> None:
        super().__init__(
            raw_data_folder,
            processed_data_folder,
            ptbxl_dataset_filename,
            scp_statements_filename,
            classes_mlb_filename,
            superclasses_mlb_filenames,
            tabular_filename,
            waves_filename,
            threshold,
            sampling_rate,
            batch_size,
        )

        tabular = pd.read_csv(os.path.join(processed_data_folder, tabular_filename))
        tabular["multilabel_superclass"] = tabular["superclass"].apply(ast.literal_eval)

        tabular["multilabel_superclass"] = tabular["multilabel_superclass"].apply(lambda x: ('/'.join(x),))

        self.balanced_multilabel_encoder = self.multilabel_encoder(tabular)

        # encoding only for balancing for train
        self.y_train_multilabel = self.prepare_labels_encoding(tabular, self.balanced_multilabel_encoder)

        counter = counter_dict_class(tabular["multilabel_superclass"][self.labels_indices_train])
        print("Counter train after prepare multi-labels", counter)
        create_pie(counter, "Imblanced", "Multi-label", "subclasses", self.processed_data_folder)

        # not needed
        # y_train_split_multilabel = split_multilabels(tabular["multilabel_superclass"][self.labels_indices_train])
        # counter = counter_dict_class(y_train_split_multilabel)
        # create_pie(counter, "Imblanced", "Multi-label", "superclasses", self.processed_data_folder)

    def make_balanced_train_dataloader(self, x_resampled, y_resampled) -> DataLoader:
        return DataLoader(
            PtbXl(x_resampled, y_resampled),
            batch_size=self.batch_size
        )

    def make_balanced_val_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self._waves_val, self.y_val),
            batch_size=self.batch_size
        )

    def make_balanced_test_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self._waves_test, self.y_test),
            batch_size=self.batch_size
        )

    def make_balanced_test_test_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self._waves_train, self.y_train),
            batch_size=self.batch_size
        )

    def multilabel_encoder(self, tabular: pd.DataFrame):
        le = MultiLabelBinarizer()
        classes = Counter(tabular["multilabel_superclass"]).keys()
        le.fit(classes)
        print(le.classes_)
        return le

    def prepare_labels_encoding(self, tabular: pd.DataFrame, le) -> np.ndarray:
        labels_encoding = le.transform(tabular["multilabel_superclass"].to_numpy())
        labels_encoding_train = labels_encoding[self.labels_indices_train]
        return labels_encoding_train

    # decoding from [0, ..., 0, 0, 0, 1] to CD/HYP
    def prepare_label_decoding(self, y, le):
        return le.inverse_transform(y)

    def balanced_by_imbalanced_learn_method(self, method):
        print("Balansed by: ", method)
        if not (
                os.path.exists(os.path.join(
                    self.processed_data_folder + r"/x_resampled_multilabel", f"multilabel_{str(method)}.npy")
                )
                and os.path.exists(os.path.join(
            self.processed_data_folder + r"/y_resampled_multilabel", f"multilabel_{str(method)}.npy")
        )
        ):
            X = self._waves_train
            y = self.y_train_multilabel

            X_2d = np.reshape(X, (-1, 12000))

            X_resampled_2d, y_resampled_multilabel = method.fit_resample(X_2d, y)

            # inverse_transform
            y_resampled_decoding_label = self.prepare_label_decoding(y_resampled_multilabel,
                                                                     self.balanced_multilabel_encoder)

            # image
            y_resampled_decoded_counter = counter_dict_class(y_resampled_decoding_label)
            print(y_resampled_decoded_counter)
            create_pie(y_resampled_decoded_counter, str(method), "Multi-label", "subclasses",
                       self.processed_data_folder)

            y_resampled_decoded_label = split_multilabels(y_resampled_decoding_label)
            y_resampled = self.superclasses_mlb.transform(y_resampled_decoded_label)
            X_resampled = np.reshape(X_resampled_2d, (-1, 12, 1000))

            np.save(os.path.join(self.processed_data_folder + r"/x_resampled_multilabel",
                                 f"multilabel_{str(method)}"),
                    X_resampled)
            np.save(os.path.join(self.processed_data_folder + r"/y_resampled_multilabel",
                                 f"multilabel_{str(method)}"),
                    y_resampled)

        with open(os.path.join(
                self.processed_data_folder + r"/x_resampled_multilabel",
                f"multilabel_{str(method)}.npy"
        ), "rb") as x_file:
            X_resampled = np.load(x_file)

        with open(os.path.join(
                self.processed_data_folder + r"/y_resampled_multilabel",
                f"multilabel_{str(method)}.npy"
        ), "rb") as y_file:
            y_resampled = np.load(y_file)

        # for pie picture - not needed
        # y_resampled_inv_trans = self.superclasses_mlb.inverse_transform(y_resampled)
        # y_resampled_counter = counter_dict_class(y_resampled_inv_trans)
        # print("End balancing: ", y_resampled_counter)
        #
        # # plt.bar(y_resampled_counter.keys(), y_resampled_counter.values())
        #
        # create_pie(y_resampled_counter, str(method), "Multi-label", "superclasses", self.processed_data_folder)

        return X_resampled, y_resampled


# from CD/HYP to [['CD','HYP']]
def split_multilabels(labels_decoding):
    labels_decoded = []
    for label in labels_decoding:
        labels_decoded.append(label[0].split('/'))
    # labels_decoded.append([label.split('/') for label in labels_decoding])
    return labels_decoded


def main():
    path_ecg_analysis = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset = PtbXlClassesSuperclassesMultilabelBalanced(
        path_ecg_analysis + r"/data/raw",
        path_ecg_analysis + r"/data/processed",
        "ptbxl_database.csv",
        "scp_statements.csv",
        "classes_mlb.pkl",
        "superclasses_mlb.pkl",
        "tabular.csv",
        "waves",
        threshold=100,
        sampling_rate=100,
        batch_size=64,
        balanced_batch_size=128
    )


    balanced_classification(dataset, dataset._waves_train, dataset.y_train, 'Without_multilabel', 10)
    balanced_classification(dataset, dataset._waves_train, dataset.y_train, 'Without_multilabel', 10)
    balanced_classification(dataset, dataset._waves_train, dataset.y_train, 'Without_multilabel', 9)
    balanced_classification(dataset, dataset._waves_train, dataset.y_train, 'Without_multilabel', 9)
    balanced_classification(dataset, dataset._waves_train, dataset.y_train, 'Without_multilabel', 9)

    # over-sampling RandomOverSampler

    # X_resampled_ros, y_resampled_ros = dataset.balanced_by_imbalanced_learn_method(
    #     RandomOverSampler(sampling_strategy='not majority'))
    # balanced_classification(dataset, X_resampled_ros, y_resampled_ros, 'RandomOverSampler_multilabel', 30)
    # balanced_classification(dataset, X_resampled_ros, y_resampled_ros, 'RandomOverSampler_multilabel', 30)
    # balanced_classification(dataset, X_resampled_ros, y_resampled_ros, 'RandomOverSampler_multilabel', 30)
    #
    # # SMOTE
    # # train_test
    smote_4 = SMOTE(sampling_strategy='not majority', k_neighbors=4)
    X_resampled_smote, y_resampled_smote = dataset.balanced_by_imbalanced_learn_method(
        SMOTE(sampling_strategy='not majority', k_neighbors=4))
    balanced_classification(dataset, X_resampled_smote, y_resampled_smote, 'SMOTE_4_neighbors', 10)
    # balanced_classification(dataset, X_resampled_smote, y_resampled_smote, 'SMOTE_4_neighbors', 10)
    # balanced_classification(dataset, X_resampled_smote, y_resampled_smote, 'SMOTE_4_neighbors', 10)
    # balanced_classification(dataset, X_resampled_smote, y_resampled_smote, 'SMOTE_4_neighbors', 10)

    X_resampled_adasyn3, y_resampled_adasyn3 = dataset.balanced_by_imbalanced_learn_method(
        ADASYN(sampling_strategy='not majority', n_neighbors=3))
    X_resampled_adasyn4, y_resampled_adasyn4 = dataset.balanced_by_imbalanced_learn_method(
        ADASYN(sampling_strategy='not majority', n_neighbors=4))
    X_resampled_adasyn5, y_resampled_adasyn5 = dataset.balanced_by_imbalanced_learn_method(
        ADASYN(sampling_strategy='not majority', n_neighbors=5))
    balanced_classification(dataset, X_resampled_adasyn3, y_resampled_adasyn3, 'ADASYN_3_neighbors', 20)
    balanced_classification(dataset, X_resampled_adasyn4, y_resampled_adasyn4, 'ADASYN_4_neighbors', 20)
    balanced_classification(dataset, X_resampled_adasyn5, y_resampled_adasyn5, 'ADASYN_5_neighbors', 20)
    # balanced_classification(dataset, X_resampled_adasyn3, y_resampled_adasyn3, 'ADASYN_3_neighbors', 20)
    # balanced_classification(dataset, X_resampled_adasyn4, y_resampled_adasyn4, 'ADASYN_4_neighbors', 20)
    # balanced_classification(dataset, X_resampled_adasyn5, y_resampled_adasyn5, 'ADASYN_5_neighbors', 20)
    # balanced_classification(dataset, X_resampled_adasyn3, y_resampled_adasyn3, 'ADASYN_3_neighbors', 20)
    # balanced_classification(dataset, X_resampled_adasyn4, y_resampled_adasyn4, 'ADASYN_4_neighbors', 20)
    # balanced_classification(dataset, X_resampled_adasyn5, y_resampled_adasyn5, 'ADASYN_5_neighbors', 20)

    # ---------------------------------------------------------------------------------------------------

    # over-sampling controlled
    # RandomUnderSampler
    X_resampled_rus, y_resampled_rus = dataset.balanced_by_imbalanced_learn_method(
        RandomUnderSampler(sampling_strategy='not minority'))
    balanced_classification(dataset, X_resampled_rus, y_resampled_rus, 'RandomUnderSampler_multilabel', 20)
    # balanced_classification(dataset, X_resampled_rus, y_resampled_rus, 'RandomUnderSampler_multilabel', 20)
    # balanced_classification(dataset, X_resampled_rus, y_resampled_rus, 'RandomUnderSampler_multilabel', 20)
    # balanced_classification(dataset, X_resampled_rus, y_resampled_rus, 'RandomUnderSampler_multilabel', 20)

    # NearMiss
    X_resampled_nm1, y_resampled_nm1 = dataset.balanced_by_imbalanced_learn_method(NearMiss(version=1))
    X_resampled_nm2, y_resampled_nm2 = dataset.balanced_by_imbalanced_learn_method(NearMiss(version=2))
    X_resampled_nm3, y_resampled_nm3 = dataset.balanced_by_imbalanced_learn_method(NearMiss(version=3))
    X_resampled_nm3_neighbor, y_resampled_nm3_neighbor = dataset.balanced_by_imbalanced_learn_method(
        NearMiss(version=3, n_neighbors_ver3=5))
    balanced_classification(dataset, X_resampled_nm1, y_resampled_nm1, 'NearMiss-1_multilabel', 20)
    # balanced_classification(dataset, X_resampled_nm1, y_resampled_nm1, 'NearMiss-1_multilabel', 20)
    # balanced_classification(dataset, X_resampled_nm1, y_resampled_nm1, 'NearMiss-1_multilabel', 20)
    balanced_classification(dataset, X_resampled_nm2, y_resampled_nm2, 'NearMiss-2_multilabel', 20)
    # balanced_classification(dataset, X_resampled_nm2, y_resampled_nm2, 'NearMiss-2_multilabel', 20)
    # balanced_classification(dataset, X_resampled_nm2, y_resampled_nm2, 'NearMiss-2_multilabel', 20)
    balanced_classification(dataset, X_resampled_nm3, y_resampled_nm3, 'NearMiss-3_multilabel', 20)
    # balanced_classification(dataset, X_resampled_nm3, y_resampled_nm3, 'NearMiss-3_multilabel', 20)
    # balanced_classification(dataset, X_resampled_nm3, y_resampled_nm3, 'NearMiss-3_multilabel', 20)
    balanced_classification(dataset, X_resampled_nm3_neighbor, y_resampled_nm3_neighbor,
                            'NearMiss-3__multilabel_n_neighbors_ver3=14', 20)
    # balanced_classification(dataset, X_resampled_nm3_neighbor, y_resampled_nm3_neighbor,
    #                         'NearMiss-3__multilabel_n_neighbors_ver3=14', 20)

    # over-sampling cleaning
    # EditedNearestNeighbours
    enn_2_mode = EditedNearestNeighbours(n_neighbors=2, kind_sel='mode')
    X_resampled_enn1, y_resampled_enn1 = dataset.balanced_by_imbalanced_learn_method(EditedNearestNeighbours(
        n_neighbors=2, kind_sel='mode'))
    balanced_classification(dataset, X_resampled_enn1, y_resampled_enn1,
                            'EditedNearestNeighbours_multilabel_n_neighbors=2_kind_sel=mode', 20)
    # balanced_classification(dataset, X_resampled_enn1, y_resampled_enn1,
    #                         'EditedNearestNeighbours_multilabel_n_neighbors=2_kind_sel=mode', 20)
    # balanced_classification(dataset, X_resampled_enn1, y_resampled_enn1,
    #                         'EditedNearestNeighbours_multilabel_n_neighbors=2_kind_sel=mode', 20)
    enn_3_all = EditedNearestNeighbours(n_neighbors=3, kind_sel='all')
    X_resampled_enn2, y_resampled_enn2 = dataset.balanced_by_imbalanced_learn_method(EditedNearestNeighbours(
        n_neighbors=3, kind_sel='all'))
    balanced_classification(dataset, X_resampled_enn2, y_resampled_enn2,
                            'EditedNearestNeighbours_multilabel_n_neighbors=3_kind_sel=all', 20)
    # balanced_classification(dataset, X_resampled_enn2, y_resampled_enn2,
    #                         'EditedNearestNeighbours_multilabel_n_neighbors=3_kind_sel=all', 20)
    # balanced_classification(dataset, X_resampled_enn2, y_resampled_enn2,
    #                         'EditedNearestNeighbours_multilabel_n_neighbors=3_kind_sel=all', 20)

    # RepeatedEditedNearestNeighbours (over-sampling cleaning)
    X_resampled_renn, y_resampled_renn = dataset.balanced_by_imbalanced_learn_method(
        RepeatedEditedNearestNeighbours(
            n_neighbors=5, kind_sel='all', max_iter=100))
    balanced_classification(dataset, X_resampled_renn, y_resampled_renn,
                            'RepeatedEditedNearestNeighbours_kind_sel=all_max_iter=100', 20)
    # balanced_classification(dataset, X_resampled_renn, y_resampled_renn,
    #                         'RepeatedEditedNearestNeighbours_kind_sel=all_max_iter=100', 20)
    # balanced_classification(dataset, X_resampled_renn, y_resampled_renn,
    #                         'RepeatedEditedNearestNeighbours_kind_sel=all_max_iter=100', 20)

    # AllKNN (over-sampling cleaning)
    X_resampled_allknn, y_resampled_allknn = dataset.balanced_by_imbalanced_learn_method(
        AllKNN())
    balanced_classification(dataset, X_resampled_allknn, y_resampled_allknn,
                            'AllKNN_sampling_kind_sel=all_n_neighbors=3', 20)
    # balanced_classification(dataset, X_resampled_allknn, y_resampled_allknn,
    #                         'AllKNN_sampling_kind_sel=all_n_neighbors=3', 20)
    # balanced_classification(dataset, X_resampled_allknn, y_resampled_allknn,
    #                         'AllKNN_sampling_kind_sel=all_n_neighbors=3', 20)

    # NeighbourhoodCleaningRule (over-sampling)
    X_resampled_ncr, y_resampled_ncr = dataset.balanced_by_imbalanced_learn_method(
        NeighbourhoodCleaningRule(kind_sel='mode'))
    balanced_classification(dataset, X_resampled_ncr, y_resampled_ncr,
                            'NeighbourhoodCleaningRule_multilabel_kind_sel=mode', 20)
    # balanced_classification(dataset, X_resampled_ncr, y_resampled_ncr,
    #                         'NeighbourhoodCleaningRule_multilabel_kind_sel=mode', 20)
    # balanced_classification(dataset, X_resampled_ncr, y_resampled_ncr,
    #                         'NeighbourhoodCleaningRule_multilabel_kind_sel=mode', 20)

    # TomekLinks (over-sampling cleaning)
    # параметров у метода нет (majority не работает)
    X_resampled_tomeklinks, y_resampled_tomeklinks = dataset.balanced_by_imbalanced_learn_method(TomekLinks())
    balanced_classification(dataset, X_resampled_tomeklinks, y_resampled_tomeklinks, 'TomekLinks_multilabel', 20)
    # balanced_classification(dataset, X_resampled_tomeklinks, y_resampled_tomeklinks, 'TomekLinks_multilabel', 20)
    # balanced_classification(dataset, X_resampled_tomeklinks, y_resampled_tomeklinks, 'TomekLinks_multilabel', 20)

    # CondensedNearestNeighbour (over-sampling cleaning)
    # X_resampled_cnn, y_resampled_cnn = dataset.balanced_by_imbalanced_learn_method(
    #     CondensedNearestNeighbour())
    # balanced_classification(dataset, X_resampled_cnn, y_resampled_cnn, 'CondensedNearestNeighbour', 20)
    # balanced_classification(dataset, X_resampled_cnn, y_resampled_cnn, 'CondensedNearestNeighbour', 20)
    # balanced_classification(dataset, X_resampled_cnn, y_resampled_cnn, 'CondensedNearestNeighbour', 20)

    X_resampled_cnn, y_resampled_cnn = dataset.balanced_by_imbalanced_learn_method(
        CondensedNearestNeighbour(n_neighbors=1))
    balanced_classification(dataset, X_resampled_cnn, y_resampled_cnn, 'CondensedNearestNeighbour_1_neighbors', 20)
    # balanced_classification(dataset, X_resampled_cnn, y_resampled_cnn, 'CondensedNearestNeighbour_1_neighbors', 20)
    # balanced_classification(dataset, X_resampled_cnn, y_resampled_cnn, 'CondensedNearestNeighbour_1_neighbors', 20)
    #
    # X_resampled_cnn, y_resampled_cnn = dataset.balanced_by_imbalanced_learn_method(
    #     CondensedNearestNeighbour(n_neighbors=5))
    # balanced_classification(dataset, X_resampled_cnn, y_resampled_cnn, 'CondensedNearestNeighbour_5_neighbors', 20)
    # balanced_classification(dataset, X_resampled_cnn, y_resampled_cnn, 'CondensedNearestNeighbour_5_neighbors', 20)
    # balanced_classification(dataset, X_resampled_cnn, y_resampled_cnn, 'CondensedNearestNeighbour_5_neighbors', 20)

    # InstanceHardnessThreshold (over-sampling cleaning)
    X_resampled_iht, y_resampled_iht = dataset.balanced_by_imbalanced_learn_method(
        InstanceHardnessThreshold(estimator=LogisticRegression(solver='lbfgs', multi_class='auto'), cv=10))
    balanced_classification(dataset, X_resampled_iht, y_resampled_iht,
                            'InstanceHardnessThreshold_multilabel_LogisticRegression_cv_10', 20)
    # balanced_classification(dataset, X_resampled_iht, y_resampled_iht,
    #                         'InstanceHardnessThreshold_multilabel_LogisticRegression_cv_10', 20)
    # balanced_classification(dataset, X_resampled_iht, y_resampled_iht,
    #                         'InstanceHardnessThreshold_multilabel_LogisticRegression_cv_10', 20)

    # combined method
    # SMOTEENN
    # X_resampled_smoteenn, y_resampled_smoteenn = dataset.balanced_by_imbalanced_learn_method(
    #     SMOTEENN(sampling_strategy='not minority', smote=smote_4, enn=enn_2_mode))
    # balanced_classification(dataset, X_resampled_smoteenn, y_resampled_smoteenn,
    #                         'SMOTEENN_multilabel_smote_4_enn_2_mode', 20)
    # balanced_classification(dataset, X_resampled_smoteenn, y_resampled_smoteenn,
    #                         'SMOTEENN_multilabel_smote_4_enn_2_mode', 20)
    # balanced_classification(dataset, X_resampled_smoteenn, y_resampled_smoteenn,
    #                         'SMOTEENN_multilabel_smote_4_enn_2_mode', 20)

    X_resampled_smoteenn, y_resampled_smoteenn = dataset.balanced_by_imbalanced_learn_method(
        SMOTEENN(sampling_strategy='not minority'))
    balanced_classification(dataset, X_resampled_smoteenn, y_resampled_smoteenn, 'SMOTEENN_multilabel', 20)
    # balanced_classification(dataset, X_resampled_smoteenn, y_resampled_smoteenn, 'SMOTEENN_multilabel', 20)
    # balanced_classification(dataset, X_resampled_smoteenn, y_resampled_smoteenn, 'SMOTEENN_multilabel', 20)

    # SMOTETomek
    X_resampled_smotetomek, y_resampled_smotetomek = dataset.balanced_by_imbalanced_learn_method(
        SMOTETomek())
    balanced_classification(dataset, X_resampled_smotetomek, y_resampled_smotetomek, 'SMOTETomek_multilabel', 20)
    # balanced_classification(dataset, X_resampled_smotetomek, y_resampled_smotetomek, 'SMOTETomek_multilabel', 20)
    # balanced_classification(dataset, X_resampled_smotetomek, y_resampled_smotetomek, 'SMOTETomek_multilabel', 20)


    # X_resampled_smotetomek, y_resampled_smotetomek = dataset.balanced_by_imbalanced_learn_method(
    #     SMOTETomek(sampling_strategy='not minority', smote=smote_4, tomek=TomekLinks()))
    # balanced_classification(dataset, X_resampled_smotetomek, y_resampled_smotetomek, 'SMOTETomek_multilabel_smote_4',
    #                         20)
    # balanced_classification(dataset, X_resampled_smotetomek, y_resampled_smotetomek, 'SMOTETomek_multilabel_smote_4',
    #                         20)
    # balanced_classification(dataset, X_resampled_smotetomek, y_resampled_smotetomek, 'SMOTETomek_multilabel_smote_4',
    #                         20)


if __name__ == "__main__":
    main()
