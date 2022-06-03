import os
import ast
import sys

# import seaborn as sns
from functools import partial
from collections import Counter
from datetime import datetime

from balanced.balanced_classification import balanced_classification
# from balanced.confusion_matrix import conf_matrix
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

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MultiLabelBinarizer, normalize

from balanced.autoencoder import ECG_NN, buil_compile_fit_ECG_NN
from balanced.autoencoder_dataset import AutoencoderDataset

from ecg_analysis.dataset import PtbXlClassesSuperclasses

import matplotlib.pyplot as plt


class PtbXlClassesSuperclassesBalanced(PtbXlClassesSuperclasses):
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

        tabular["superclass_one_label"] = tabular["superclass"].apply(ast.literal_eval)

        # для картинки
        counter = counter_dict_class(tabular["superclass_one_label"])
        print("Counter before prepare one-labels", counter)

        # для избавления от мультилейблинга
        one_label_preparing_counter = partial(one_label_preparing, counter=counter)
        tabular["superclass_one_label"] = tabular["superclass_one_label"].apply(one_label_preparing_counter)

        # для картинки
        counter = counter_dict_class(tabular["superclass_one_label"])
        print("Counter all after prepare one-labels", counter)
        counter = counter_dict_class(tabular["superclass_one_label"][self.labels_indices_train])
        print("Counter train after prepare one-labels", counter)
        create_pie(counter, "Imblanced", "Single-label", "superclasses", self.processed_data_folder)

        # for test.npy
        counter = counter_dict_class_test(tabular["superclass_one_label"][self.labels_indices_test])
        print("Counter train after prepare one-labels", counter)

        # Save processed tabular data
        os.makedirs(processed_data_folder, exist_ok=True)
        tabular.to_csv(os.path.join(processed_data_folder, tabular_filename))

        self.labels = self.prepare_mono_labels(tabular)

    @property
    def y_train(self) -> np.ndarray:
        return self.labels[self.labels_indices_train]

    @property
    def y_val(self) -> np.ndarray:
        return self.labels[self.labels_indices_val]

    @property
    def y_test(self) -> np.ndarray:
        return self.labels[self.labels_indices_test]

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

    def prepare_mono_labels(self, tabular: pd.DataFrame) -> np.ndarray:
        return self.superclasses_mlb.transform(tabular["superclass_one_label"].to_numpy())


    def balanced_by_imbalanced_learn_method(self, method):
        # # method_name = str(method)
        # method_name = 'ENN_majority_mode_1'
        print("Balansed by: ", method)
        if not (
                os.path.exists(os.path.join(
                    self.processed_data_folder + r"/x_resampled_singlelabel", f"{str(method)}.npy")
                    # self.processed_data_folder + r"/x_resampled_singlelabel" + method_name + ".npy")
                )
                and os.path.exists(os.path.join(
                    self.processed_data_folder + r"/y_resampled_singlelabel", f"{str(method)}.npy")
                # self.processed_data_folder + r"/y_resampled_singlelabel" + method_name + ".npy")
                )
        ):
            X = self._waves_test
            y = self.y_test

            # X = np.concatenate((self._waves_test, self._waves_val))
            # y = np.concatenate((self.y_test, self.y_val))

            X_2d = np.reshape(X, (-1, 12000))
            # X_2d_normalize = normalize(X_2d)

            X_resampled_2d, y_resampled = method.fit_resample(X_2d, y)
            X_resampled = np.reshape(X_resampled_2d, (-1, 12, 1000))

            # np.save(os.path.join(self.processed_data_folder + r"/x_resampled_singlelabel", method_name),
            np.save(os.path.join(self.processed_data_folder + r"/x_resampled_singlelabel", f"{str(method)}"),
                    X_resampled)
            # np.save(os.path.join(self.processed_data_folder + r"/y_resampled_singlelabel", method_name),
            np.save(os.path.join(self.processed_data_folder + r"/y_resampled_singlelabel", f"{str(method)}"),
                    y_resampled)

        # with open(os.path.join(self.processed_data_folder + r"/x_resampled_singlelabel", method_name + ".npy"), "rb") \
        with open(os.path.join(self.processed_data_folder + r"/x_resampled_singlelabel", f"{str(method)}.npy"),
                  "rb") \
                as x_file:
            X_resampled = np.load(x_file)

        # with open(os.path.join(self.processed_data_folder + r"/y_resampled_singlelabel", method_name + ".npy"), "rb") \
        with open(os.path.join(self.processed_data_folder + r"/x_resampled_singlelabel", f"{str(method)}.npy"),
                  "rb") \
                    as y_file:
            y_resampled = np.load(y_file)

        # для картинки

        y_resampled_inv_trans = self.superclasses_mlb.inverse_transform(y_resampled)
        y_resampled_counter = counter_dict_class(y_resampled_inv_trans)
        print("End balancing: ", y_resampled_counter)

        # plt.bar(y_resampled_counter.keys(), y_resampled_counter.values())

        # create_pie(y_resampled_counter, method_name, "Single-label", "superclasses", self.processed_data_folder)
        create_pie(y_resampled_counter, str(method), "Single-label", "superclasses", self.processed_data_folder)
        # self.create_figure(X_resampled, y_resampled, y_resampled_counter, method, "Single-label", "superclasses")

        return X_resampled, y_resampled


def counter_dict_class(tabular_column):
    """Creates the counter of classes in the tabular by column"""

    all_labels = []
    if type(tabular_column[0]) is tuple or type(tabular_column[0]) is list:
        all_labels = [label for labels in tabular_column for label in labels]
    elif type(tabular_column[0]) is str:
        all_labels = [label for label in tabular_column]
    # ax = sns.countplot(all_labels, order=[k for k, _ in result.most_common()], log=True)
    # ax.set_title('Number of classes with a class label')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
    return Counter(all_labels)

def counter_dict_class_test(tabular_column):
    all_labels = []
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

    ax.set(aspect="equal", title=method + label_tag + ", " + class_tag)

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
        processed_data_folder + r"/pies/singlelabel/pie_" + method + '_' + pie_label_tag + "_" + class_tag + ".png",
        bbox_inches='tight')

    ax.clear()

    return plt


# from CD/HYP to [['CD','HYP']]
def split_multilabels(labels_decoding):
    labels_decoded = []
    for label in labels_decoding:
        labels_decoded.append(label[0].split('/'))
    # labels_decoded.append([label.split('/') for label in labels_decoding])
    return labels_decoded


def main():
    path_ecg_analysis = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset = PtbXlClassesSuperclassesBalanced(
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

    # balanced_classification(dataset, dataset._waves_train, dataset.y_train, 'Imbalanced_singlelabel', 1)

    # over-sampling RandomOverSampler_overfit
    X_resampled_ros, y_resampled_ros = dataset.balanced_by_imbalanced_learn_method(RandomOverSampler(sampling_strategy='not majority', random_state=0))
    balanced_classification(dataset, X_resampled_ros, y_resampled_ros, 'RandomOverSampler_singlelabel', 9)

    # #
    # # SMOTE
    X_resampled_smote, y_resampled_smote = dataset.balanced_by_imbalanced_learn_method(
        SMOTE(sampling_strategy='not majority', random_state=0, k_neighbors=4))
    balanced_classification(dataset, X_resampled_smote, y_resampled_smote, 'SMOTE_singlelabel_4_neighbors', 12)
    #
    # # ADASYN_overfit
    X_resampled_adasyn, y_resampled_adasyn = dataset.balanced_by_imbalanced_learn_method(
        ADASYN(sampling_strategy='not majority', n_neighbors=8, random_state=0))

    balanced_classification(dataset, X_resampled_adasyn, y_resampled_adasyn, 'ADASYN_singlelabel_8_neighbors', 13)



    # # under-sampling RandomUnderSampler
    # X_resampled_rus, y_resampled_rus = dataset.balanced_by_imbalanced_learn_method(RandomUnderSampler())
    # X_resampled_rus, y_resampled_rus = dataset.balanced_by_imbalanced_learn_method(
    #      RandomUnderSampler(sampling_strategy='majority', random_state=0))
    # balanced_classification(dataset, X_resampled_rus, y_resampled_rus, 'RandomUnderSampler_singlelabel_majority', 10)


    # # NearMiss
    # X_resampled_nm1, y_resampled_nm1 = dataset.balanced_by_imbalanced_learn_method(NearMiss(version=1))
    # balanced_classification(dataset, X_resampled_nm1, y_resampled_nm1, 'NearMiss-1', 13)
    # X_resampled_nm2, y_resampled_nm2 = dataset.balanced_by_imbalanced_learn_method(NearMiss(version=2))
    # balanced_classification(dataset, X_resampled_nm2, y_resampled_nm2, 'NearMiss-2', 13)
    # X_resampled_nm3_26, y_resampled_nm3_26 = dataset.balanced_by_imbalanced_learn_method(
    #     NearMiss(version=3, n_neighbors_ver3=26))
    # balanced_classification(dataset, X_resampled_nm3_26, y_resampled_nm3_26,
    #                         'NearMiss-3', 13)


    # ENN
    # X_resampled_enn1, y_resampled_enn1 = dataset.balanced_by_imbalanced_learn_method(EditedNearestNeighbours(
    #     sampling_strategy='majority', n_neighbors=1, kind_sel='mode'))
    # balanced_classification(dataset, X_resampled_enn1, y_resampled_enn1,
    #                         'EditedNearestNeighbours_singlelabel_not minority', 11)
    # X_resampled_enn2, y_resampled_enn2 = dataset.balanced_by_imbalanced_learn_method(EditedNearestNeighbours(
    #     sampling_strategy='not minority', n_neighbors=3, kind_sel='all'))
    # balanced_classification(dataset, X_resampled_enn2, y_resampled_enn2, 'EditedNearestNeighbours_singlelabel_not minority', 11)
    # balanced_classification(dataset, X_resampled_enn2, y_resampled_enn2,
    #                         'EditedNearestNeighbours_singlelabel_not minority', 11)
    # balanced_classification(dataset, X_resampled_enn2, y_resampled_enn2,
    #                         'EditedNearestNeighbours_singlelabel_not minority', 11)

    # RENN
    # X_resampled_renn, y_resampled_renn = dataset.balanced_by_imbalanced_learn_method(
    #     RepeatedEditedNearestNeighbours(
    #         sampling_strategy='majority', n_neighbors=3, kind_sel='mode', max_iter=100
    #     ))
    # balanced_classification(dataset, X_resampled_renn, y_resampled_renn, 'RepeatedEditedNearestNeighbours_singlelabel', 8)


    # AllKNN
    # X_resampled_allknn, y_resampled_allknn = dataset.balanced_by_imbalanced_learn_method(
    #     AllKNN(sampling_strategy='majority', kind_sel='mode'))
    # balanced_classification(dataset, X_resampled_allknn, y_resampled_allknn, 'AllKNN_majority', 13)
    # X_resampled_allknn, y_resampled_allknn = dataset.balanced_by_imbalanced_learn_method(
    #     AllKNN(sampling_strategy='not minority', kind_sel='mode'))
    # balanced_classification(dataset, X_resampled_allknn, y_resampled_allknn, 'AllKNN_not minority', 13)


    # # # TomekLinks
    # X_resampled_tomek, y_resampled_tomek = dataset.balanced_by_imbalanced_learn_method(
    #         TomekLinks(sampling_strategy='not minority'))
    # balanced_classification(dataset, X_resampled_tomek, y_resampled_tomek,
    #                         'TomekLinks_singlelabel_sampling_strategy=not minority', 9)
    #
    # X_resampled_tomek, y_resampled_tomek = dataset.balanced_by_imbalanced_learn_method(
    #     TomekLinks(sampling_strategy='majority'))
    # balanced_classification(dataset, X_resampled_tomek, y_resampled_tomek,
    #                         'TomekLinks_singlelabel_sampling_strategy=majority', 16)
    #
    # X_resampled_tomeklinks, y_resampled_tomeklinks = dataset.balanced_by_imbalanced_learn_method(TomekLinks())
    # balanced_classification(dataset, X_resampled_tomeklinks, y_resampled_tomeklinks, 'TomekLinks', 8)


    # CondensedNearestNeighbour
    # X_resampled_cnn, y_resampled_cnn = dataset.balanced_by_imbalanced_learn_method(
    #     CondensedNearestNeighbour(sampling_strategy='majority', n_neighbors=2))
    # balanced_classification(dataset, X_resampled_cnn, y_resampled_cnn,
    #                         'CondensedNearestNeighbour_majority_2_neighbors', 3)
    # balanced_classification(dataset, X_resampled_cnn, y_resampled_cnn,
    #                         'CondensedNearestNeighbour_majority_2_neighbors', 3)
    # X_resampled_cnn2, y_resampled_cnn2 = dataset.balanced_by_imbalanced_learn_method(
    #     CondensedNearestNeighbour(sampling_strategy='majority', n_neighbors=3))
    # balanced_classification(dataset, X_resampled_cnn2, y_resampled_cnn2,
    #                         'CondensedNearestNeighbour_majority_3_neighbors', 2)
    # balanced_classification(dataset, X_resampled_cnn2, y_resampled_cnn2,
    #                         'CondensedNearestNeighbour_majority_3_neighbors', 2)
    # balanced_classification(dataset, X_resampled_cnn2, y_resampled_cnn2,
    #                         'CondensedNearestNeighbour_majority_3_neighbors', 1)
    # balanced_classification(dataset, X_resampled_cnn2, y_resampled_cnn2,
    #                         'CondensedNearestNeighbour_majority_3_neighbors', 1)
    # X_resampled_cnn4, y_resampled_cnn4 = dataset.balanced_by_imbalanced_learn_method(
    #     CondensedNearestNeighbour(sampling_strategy='not minority', n_neighbors=5))
    # balanced_classification(dataset, X_resampled_cnn4, y_resampled_cnn4, 'CondensedNearestNeighbour_not_minority', 22)

    # # InstanceHardnessThreshold
    # X_resampled_iht, y_resampled_iht = dataset.balanced_by_imbalanced_learn_method(
    #     InstanceHardnessThreshold(estimator=LogisticRegression(solver='lbfgs', multi_class='auto')))
    # balanced_classification(dataset, X_resampled_iht, y_resampled_iht, 'InstanceHardnessThreshold', 12)
    # balanced_classification(dataset, X_resampled_iht, y_resampled_iht, 'InstanceHardnessThreshold', 12)
    # balanced_classification(dataset, X_resampled_iht, y_resampled_iht, 'InstanceHardnessThreshold', 12)
    # balanced_classification(dataset, X_resampled_iht, y_resampled_iht, 'InstanceHardnessThreshold', 12)


    # # OneSidedSelection
    # X_resampled_oss, y_resampled_oss = dataset.balanced_by_imbalanced_learn_method(
    #     OneSidedSelection(sampling_strategy='majority'))
    # balanced_classification(dataset, X_resampled_oss, y_resampled_oss, 'OneSidedSelection', 10)
    # X_resampled_oss, y_resampled_oss = dataset.balanced_by_imbalanced_learn_method(
    #     OneSidedSelection(sampling_strategy='not minority'))
    # balanced_classification(dataset, X_resampled_oss, y_resampled_oss, 'OneSidedSelection', 10)
    #
    # # NeighbourhoodCleaningRule
    # X_resampled_ncr, y_resampled_ncr = dataset.balanced_by_imbalanced_learn_method(
    #     NeighbourhoodCleaningRule(sampling_strategy='majority'))
    # balanced_classification(dataset, X_resampled_ncr, y_resampled_ncr, 'NeighbourhoodCleaningRule_majority', 20)
    # X_resampled_ncr, y_resampled_ncr = dataset.balanced_by_imbalanced_learn_method(
    #     NeighbourhoodCleaningRule(sampling_strategy='majority', n_neighbors=1))
    # balanced_classification(dataset, X_resampled_ncr, y_resampled_ncr, 'NeighbourhoodCleaningRule_majority_1', 20)
    # X_resampled_ncr, y_resampled_ncr = dataset.balanced_by_imbalanced_learn_method(
    #     NeighbourhoodCleaningRule(sampling_strategy='majority', n_neighbors=2))
    # balanced_classification(dataset, X_resampled_ncr, y_resampled_ncr, 'NeighbourhoodCleaningRule_majority_2', 20)
    # X_resampled_ncr, y_resampled_ncr = dataset.balanced_by_imbalanced_learn_method(
    #     NeighbourhoodCleaningRule(sampling_strategy='majority', n_neighbors=10))
    # balanced_classification(dataset, X_resampled_ncr, y_resampled_ncr, 'NeighbourhoodCleaningRule_majority_10', 20)

    # X_resampled_ncr, y_resampled_ncr = dataset.balanced_by_imbalanced_learn_method(
    #     NeighbourhoodCleaningRule(sampling_strategy='not minority'))
    # balanced_classification(dataset, X_resampled_ncr, y_resampled_ncr, 'NeighbourhoodCleaningRule_not_minority', 20)
    # balanced_classification(dataset, X_resampled_ncr, y_resampled_ncr, 'NeighbourhoodCleaningRule_not_minority', 10)

    # # ClusterCentroids
    # X_resampled_cc, y_resampled_cc = dataset.balanced_by_imbalanced_learn_method(
    #     ClusterCentroids(sampling_strategy='not minority'))
    # balanced_classification(dataset, X_resampled_cc, y_resampled_cc, 'ClusterCentroids', 10)
    #
    # X_resampled_cc2, y_resampled_cc2 = dataset.balanced_by_imbalanced_learn_method(
    #     ClusterCentroids(sampling_strategy='not minority', voting='soft'))
    # X_resampled_cc3, y_resampled_cc3 = dataset.balanced_by_imbalanced_learn_method(
    #     ClusterCentroids(sampling_strategy='majority'))
    # X_resampled_cc4, y_resampled_cc4 = dataset.balanced_by_imbalanced_learn_method(
    #     ClusterCentroids(sampling_strategy='majority', voting='soft'))
    # X_resampled_cc5, y_resampled_cc5 = dataset.balanced_by_imbalanced_learn_method(
    #     ClusterCentroids(sampling_strategy='majority', voting='hard', random_state=0))
    # balanced_classification(dataset, X_resampled_cc, y_resampled_cc, 'ClusterCentroids_not_minority', 5)
    # balanced_classification(dataset, X_resampled_cc2, y_resampled_cc2, 'ClusterCentroids_not_minority_soft', 3)
    # balanced_classification(dataset, X_resampled_cc3, y_resampled_cc3, 'ClusterCentroids_majority', 1)
    # balanced_classification(dataset, X_resampled_cc4, y_resampled_cc4, 'ClusterCentroid_majority_soft', 5)
    # balanced_classification(dataset, X_resampled_cc5, y_resampled_cc5, 'ClusterCentroids_majority_hard', 1)


    #
    # # combinated methods
    # smote = SMOTE(sampling_strategy='not majority', random_state=0, k_neighbors=4)
    # enn = EditedNearestNeighbours(sampling_strategy='not minority', n_neighbors=3, kind_sel='mode')
    # # smote = SMOTE(sampling_strategy='not majority')
    # # enn = EditedNearestNeighbours()
    # # X_resampled_smoteenn, y_resampled_smoteenn = dataset.balanced_by_imbalanced_learn_method(
    # #     SMOTEENN(smote=smote, enn=enn))
    # X_resampled_smoteenn, y_resampled_smoteenn = dataset.balanced_by_imbalanced_learn_method(
    #     SMOTEENN(sampling_strategy='not majority'))

    #
    # X_resampled_smotetomek, y_resampled_smotetomek = dataset.balanced_by_imbalanced_learn_method(SMOTETomek(random_state=0))
    # balanced_classification(dataset, X_resampled_smotetomek, y_resampled_smotetomek, 'SMOTETomek_singlelabel', 1)
    # balanced_classification(dataset, X_resampled_smotetomek, y_resampled_smotetomek, 'SMOTETomek_singlelabel', 1)
    # balanced_classification(dataset, X_resampled_smotetomek, y_resampled_smotetomek, 'SMOTETomek_singlelabel', 1)
    # balanced_classification(dataset, X_resampled_smotetomek, y_resampled_smotetomek, 'SMOTETomek_singlelabel', 1)

    # smote = SMOTE(sampling_strategy='not majority', k_neighbors=4)
    # tomek = TomekLinks(sampling_strategy='not majority')
    # X_resampled_smotetomek, y_resampled_smotetomek = dataset.balanced_by_imbalanced_learn_method(
    #     SMOTETomek(sampling_strategy='not minority', smote=smote, tomek=tomek))
    # balanced_classification(dataset, X_resampled_smotetomek, y_resampled_smotetomek,
    #                         'SMOTETomek_smote_4_singlelabel', 15)

if __name__ == "__main__":
    main()
