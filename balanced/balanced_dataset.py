import os
import ast
import sys

# import seaborn as sns
from functools import partial
from collections import Counter
from datetime import datetime

from ecg_analysis.dataset import PtbXlClassesSuperclasses, PtbXl

import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, \
    OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold, \
    TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

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
        self.counter = self.counter_dict_class(tabular["superclass_one_label"])

        one_label_preparing_counter = partial(one_label_preparing, counter=self.counter)
        tabular["superclass_one_label"] = tabular["superclass_one_label"].apply(one_label_preparing_counter)

        # для картинки
        self.counter = self.counter_dict_class(tabular["superclass_one_label"])

        # Save processed tabular data
        os.makedirs(processed_data_folder, exist_ok=True)
        tabular.to_csv(os.path.join(processed_data_folder, tabular_filename))

        self.y_balance_class = tabular["superclass_one_label"]

        print("y_balance_class_counter", self.counter_dict_class(self.y_balance_class))

        self.y_balanced_label = self.prepare_labels_balance(tabular)

    @property
    def y_balance_train(self) -> np.ndarray:
        return self.y_balanced_label[self.labels_indices_train]

    @property
    def y_balance_val(self) -> np.ndarray:
        return self.y_balanced_label[self.labels_indices_val]

    @property
    def y_balance_test(self) -> np.ndarray:
        return self.y_balanced_label[self.labels_indices_test]

    def make_balanced_train_dataloader(self, x_resampled, y_resampled) -> DataLoader:
        return DataLoader(
            PtbXl(x_resampled, y_resampled),
            batch_size=self.batch_size
        )

    def make_balanced_val_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self._waves_val, self.y_balance_val),
            batch_size=self.batch_size
        )

    def make_balanced_test_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self._waves_test, self.y_balance_test),
            batch_size=self.batch_size
        )

    # def balanced_by_RandomOverSampler(self):
    #     # перевод 3d в 2d
    #     X = self._waves_train[:, 0]
    #     y = self.y_balance
    #     print(self.counter_dict_class(y))
    #
    #     ros = RandomOverSampler(random_state=0)
    #     X_resampled_NaiveRandom, y_resampled_NaiveRandom = ros.fit_resample(X, y)
    #     print(self.counter_dict_class(y_resampled_NaiveRandom))
    #
    #     return X_resampled_NaiveRandom, y_resampled_NaiveRandom

    # Old version balancing

    # def balanced_by_imbalanced_learn_method(self, method):
    #     print("Balansed by: ", method)
    #     # print(self._waves_train.shape)
    #
    #     y = self.y_balance_train
    #
    #     x, _ = method.fit_resample(self._waves_train[:, 1], y)
    #     X_resampled = np.empty(shape=[x.shape[0], 0, 1000])
    #
    #     for i in range(12):
    #         X = self._waves_train[:, i]
    #
    #         X_resampled_i, y_resampled = method.fit_resample(X, y)
    #         X_resampled_i = np.expand_dims(X_resampled_i, axis=1)
    #         X_resampled = np.append(X_resampled, X_resampled_i, axis=1)
    #
    #     # для картинки
    #     # y_resampled_inv_trans = self.superclasses_mlb.inverse_transform(y_resampled)
    #     # print(self.counter_dict_class(y_resampled_inv_trans))
    #
    #     return X_resampled, y_resampled

    def balanced_by_imbalanced_learn_method(self, method):
        print("Balansed by: ", method)
        if not (
                os.path.exists(os.path.join(
                    r"C:\Anastasia\ecg_analysis\data/processed\x_resampled", f"{str(method)}.npy")
                )
                and os.path.exists(os.path.join(
            r"C:\Anastasia\ecg_analysis\data\processed\y_resampled", f"{str(method)}.npy")
        )
        ):
            X = self._waves_train
            y = self.y_balance_train

            X_2d = np.reshape(X, (-1, 12000))

            X_resampled_2d, y_resampled = method.fit_resample(X_2d, y)

            X_resampled = np.reshape(X_resampled_2d, (-1, 12, 1000))

            np.save(os.path.join(r"C:\Anastasia\ecg_analysis\data\processed\x_resampled", f"{str(method)}"),
                    X_resampled)
            np.save(os.path.join(r"C:\Anastasia\ecg_analysis\data\processed\y_resampled", f"{str(method)}"),
                    y_resampled)

        with open(os.path.join(
                r"C:\Anastasia\ecg_analysis\data\processed\x_resampled", f"{str(method)}.npy"
        ), "rb") as x_file:
            X_resampled = np.load(x_file)

        with open(os.path.join(
                r"C:\Anastasia\ecg_analysis\data\processed\y_resampled", f"{str(method)}.npy"
        ), "rb") as y_file:
            y_resampled = np.load(y_file)

        # для картинки

        y_resampled_inv_trans = self.superclasses_mlb.inverse_transform(y_resampled)
        y_resampled_counter = self.counter_dict_class(y_resampled_inv_trans)
        print("End balancing: ", y_resampled_counter)

        # plt.bar(y_resampled_counter.keys(), y_resampled_counter.values())

        cmap = plt.get_cmap('Spectral')
        colors = [cmap(i) for i in np.linspace(0, 1, 5)]

        plt.pie(y_resampled_counter.values(),
                labels=y_resampled_counter.keys(),
                autopct='%1.1f%%',
                shadow=True,
                colors=colors,
                wedgeprops={"edgecolor": "black",
                            'linewidth': 1,
                            'antialiased': True}
                )
        plt.show()

        return X_resampled, y_resampled

    def balanced_by_imbalanced_learn_method_witn_autoencoder(self, method):
        print("Balansed by: ", method)

        y = self.y_balance_train
        # self.y_balanced_encoded_train
        print(self._waves_train.shape)

        if not os.path.exists("model_autoencoder.pt"):
            dataset_train = AutoencoderDataset(self._waves_train)
            dataset_test = AutoencoderDataset(self._waves_test)

            # train_dl = self.make_train_dataloader()
            # test_dl = self.make_test_dataloader()

            dataloader_train = torch.utils.data.DataLoader(
                dataset_train, batch_size=128, shuffle=True, pin_memory=True
            )

            dataloader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=32, shuffle=False
            )

            model_autoencoder = buil_compile_fit_ECG_NN(dataloader_train, dataset_train, dataloader_test, dataset_test)

            # Save model weights
            # os.path.join("balanced", "model_autoencoder.pt")
            torch.save(model_autoencoder.state_dict(), "model_autoencoder.pt")

        model_weight = torch.load("model_autoencoder.pt")
        model_loaded = ECG_NN(input_shape=12)
        model_loaded.load_state_dict(model_weight)

        dataset_train = AutoencoderDataset(self._waves_train)
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=128, shuffle=False, pin_memory=True
        )

        outputs = []
        model_loaded.eval()
        for i, data in enumerate(dataloader_train):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data = data.to(device)
            output = model_loaded(data).detach().numpy()
            outputs.append(output)
        print(outputs)

        # to 2d
        # outputs = np.empty(shape=[dataset_train.length//dataloader_train.batch_size, 1, 1000])
        X_2d = np.empty(shape=[0, 1, 1000])
        model_loaded.eval()
        for i, data in enumerate(dataloader_train):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data = data.to(device)
            output = model_loaded.call_encoder(data).detach().cpu().numpy()
            X_2d = np.append(X_2d, output, axis=0)
        print(X_2d.shape)

        # to 3d
        dataset_train = AutoencoderDataset(X_2d)
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=128, shuffle=True, pin_memory=True
        )
        outputs = np.empty(shape=[0, 12, 1000])
        model_loaded.eval()
        for i, data in enumerate(dataloader_train):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data = data.to(device)
            output = model_loaded.call_decoder(data).detach().cpu().numpy()
            outputs = np.append(outputs, output, axis=0)
        print(outputs.shape)

        # X_3d_forward = model_loaded(dataloader_train)

        # self._waves_train

        # X_2d = model_loaded.call_encoder(dataset_train)
        #
        # X_3d = model_loaded.call_decoder(X_2d)
        # print("self._waves_train:", self._waves_train)
        # print("X_3d:", X_3d)

        X_resampled, y_resampled = method.fit_resample(X_2d, y)
        print(self.counter_dict_class(y_resampled))

        return X_resampled, y_resampled

    @staticmethod
    # def counter_dict_class(self, tabular_column):
    def counter_dict_class(tabular_column):
        """Creates the counter of classes in the tabular by column"""

        if type(tabular_column[0]) is tuple:
            all_labels = [label for labels in tabular_column for label in labels]
        elif type(tabular_column[0]) is str:
            all_labels = [label for label in tabular_column]
        # ax = sns.countplot(all_labels, order=[k for k, _ in result.most_common()], log=True)
        # ax.set_title('Number of classes with a class label')
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
        return Counter(all_labels)

    def prepare_labels_balance(self, tabular: pd.DataFrame) -> np.ndarray:
        labels = self.superclasses_mlb.transform(tabular["superclass_one_label"].to_numpy())
        return labels


# мб стоит изменить имя на prepare_labels_balance
def y_encoding(y) -> np.ndarray:
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)
    print(le.classes_)
    print("array_encoded", y_encoded)

    # для обратного процесса
    # le.transform(self.y_balance_class)
    return y_encoded


def one_label_preparing(probs, counter):
    residuary_prob = probs
    tmp = sys.maxsize

    for prob in probs:
        current = counter[prob]
        if current < tmp:
            tmp = current
            residuary_prob = prob

    tmp = (residuary_prob,)

    return tmp


# def ratio_multiplier(y):
#     from collections import Counter
#
#     multiplier = {1: 0.7, 2: 0.95}
#     target_stats = Counter(y)
#     for key, value in target_stats.items():
#         if key in multiplier:
#             target_stats[key] = int(value * multiplier[key])
#     return target_stats


def ratio_multiplier(y):
    target_stats = Counter(y)
    for key, value in target_stats.items():
        target_stats[key] = target_stats.get('HYP')
    return target_stats


def main():
    path_ecg_analysis = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset = PtbXlClassesSuperclassesBalanced(
        r"C:\Anastasia\ecg_analysis\data\raw",
        r"C:\Anastasia\ecg_analysis\data\processed",
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

    X_resampled_ros, y_resampled_ros = dataset.balanced_by_imbalanced_learn_method(RandomOverSampler(random_state=0))

    X_resampled_smote, y_resampled_smote = dataset.balanced_by_imbalanced_learn_method(SMOTE())

    X_resampled_rus, y_resampled_rus = dataset.balanced_by_imbalanced_learn_method(RandomUnderSampler(random_state=0))

    # start_time = datetime.now()
    X_resampled_cc, y_resampled_cc = dataset.balanced_by_imbalanced_learn_method(
        ClusterCentroids(random_state=0))  # looooong time
    # print("--- %s seconds ---" % (datetime.now() - start_time))

    X_resampled_enn, y_resampled_enn = dataset.balanced_by_imbalanced_learn_method(
        EditedNearestNeighbours())  # need parameters
    X_resampled_renn, y_resampled_renn = dataset.balanced_by_imbalanced_learn_method(
        RepeatedEditedNearestNeighbours())  # need parameters

    X_resampled_allknn, y_resampled_allknn = dataset.balanced_by_imbalanced_learn_method(
        AllKNN(sampling_strategy='majority'))

    # start_time = datetime.now()
    X_resampled_cnn, y_resampled_cnn = dataset.balanced_by_imbalanced_learn_method(
        CondensedNearestNeighbour(random_state=0))  # looooong time
    # print("--- %s seconds ---" % (datetime.now() - start_time))

    X_resampled_oss, y_resampled_oss = dataset.balanced_by_imbalanced_learn_method(
        OneSidedSelection(random_state=0))  # need parameters
    X_resampled_ncr, y_resampled_ncr = dataset.balanced_by_imbalanced_learn_method(NeighbourhoodCleaningRule())
    X_resampled_iht, y_resampled_iht = dataset.balanced_by_imbalanced_learn_method(
        InstanceHardnessThreshold(random_state=0, estimator=LogisticRegression(solver='lbfgs', multi_class='auto')))
    X_resampled_smoteenn, y_resampled_smoteenn = dataset.balanced_by_imbalanced_learn_method(
        SMOTEENN(random_state=0))  # need parameters
    X_resampled_smotetomek, y_resampled_smotetomek = dataset.balanced_by_imbalanced_learn_method(
        SMOTETomek(random_state=0))
    X_resampled_tomeklinks, y_resampled_tomeklinks = dataset.balanced_by_imbalanced_learn_method(TomekLinks())

    # balanced_classification(dataset, dataset._waves_train, dataset.y_balance_train, 'Without')

    # balanced_classification(dataset, X_resampled_ros, y_resampled_ros, 'RandomOverSampler')
    # balanced_classification(dataset, X_resampled_smote, y_resampled_smote, 'SMOTE')
    # balanced_classification(dataset, X_resampled_rus, y_resampled_rus, 'RandomUnderSampler')
    # balanced_classification(dataset, X_resampled_cc, y_resampled_cc, 'ClusterCentroids')
    # balanced_classification(dataset, X_resampled_enn, y_resampled_enn, 'EditedNearestNeighbours')
    # balanced_classification(dataset, X_resampled_renn, y_resampled_renn, 'RepeatedEditedNearestNeighbours')
    # balanced_classification(dataset, X_resampled_allknn, y_resampled_allknn, 'AllKNN')
    # balanced_classification(dataset, X_resampled_cnn, y_resampled_cnn, 'CondensedNearestNeighbour')
    # balanced_classification(dataset, X_resampled_oss, y_resampled_oss, 'OneSidedSelection')
    # balanced_classification(dataset, X_resampled_ncr, y_resampled_ncr, 'NeighbourhoodCleaningRule')
    # balanced_classification(dataset, X_resampled_iht, y_resampled_iht, 'InstanceHardnessThreshold')
    # balanced_classification(dataset, X_resampled_smoteenn, y_resampled_smoteenn, 'SMOTEENN')
    # balanced_classification(dataset, X_resampled_smotetomek, y_resampled_smotetomek, 'SMOTETomek')
    # balanced_classification(dataset, X_resampled_tomeklinks, y_resampled_tomeklinks, 'TomekLinks')


if __name__ == "__main__":
    main()
