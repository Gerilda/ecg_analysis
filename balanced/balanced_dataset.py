import os
import ast
import sys
from typing import Optional

#import seaborn as sns
from functools import partial
from collections import Counter
from datetime import datetime

from ecg_analysis.dataset import PtbXlClassesSuperclasses, PtbXl

import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, EditedNearestNeighbours,\
                                    RepeatedEditedNearestNeighbours, CondensedNearestNeighbour,\
                                    OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold,\
                                    TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from balanced.autoencoder import buil_compile_fit_ECG_NN
from balanced.custom_dataset import CustomDataset


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
        tabular["y_balance"] = tabular["superclass"].apply(ast.literal_eval)

        self.counter = self.counter_dict_class(tabular["y_balance"])

        one_label_preparing_counter = partial(one_label_preparing, counter=self.counter)
        tabular["y_balance"] = tabular["y_balance"].apply(one_label_preparing_counter)

        self.counter = self.counter_dict_class(tabular["y_balance"])

        # Save processed tabular data
        os.makedirs(processed_data_folder, exist_ok=True)
        tabular.to_csv(os.path.join(processed_data_folder, tabular_filename))

        self.y_balance_class = tabular["y_balance"]
        print("y_balance_class_counter", self.counter_dict_class(self.y_balance_class))

        # print("array[]", np.expand_dims(self.y_balance_class.to_numpy(), axis=0))

        # энкодим стринги классов в 0, 1, 2, ... для y: all, train и test
        self.y_balanced_encoded_all = y_encoding(self.y_balance_class)
        self.y_balanced_encoded_train = y_encoding(self.y_balance_train)
        self.y_balanced_encoded_test = y_encoding(self.y_balance_test)

    @property
    def y_balance_train(self) -> np.ndarray:
        return self.y_balance_class[self.labels_indices_train]

    @property
    def y_balance_test(self) -> np.ndarray:
        return self.y_balance_class[self.labels_indices_test]

    def make_balanced_train_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self._waves_train, self.y_train),
            batch_size=self.balanced_batch_size
        )

    def make_balanced_test_dataloader(self) -> DataLoader:
        return DataLoader(
            PtbXl(self._waves_test, self.y_test),
            batch_size=self.balanced_batch_size
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

    def balanced_by_imbalanced_learn_method(self, method):
        print("Balansed by: ", method)
        # self._waves_train.take(axis=1)
        X_resampled = self._waves_train
        print(self._waves_train.shape)
        for i in range(12):
            X = self._waves_train[:, i]
            y = self.y_balance

            X_resampled_i, y_resampled = method.fit_resample(X, y)
            print(self.counter_dict_class(y_resampled))
            print(X_resampled_i.shape)
            # new = np.zeros()

        print(X_resampled_i)
        return X_resampled, y_resampled

    def balanced_by_imbalanced_learn_method_witn_autoencoder(self, method):
        print("Balansed by: ", method)

        X = self._waves_train
        y = self.y_balance_train

        X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size=0.33, random_state=42)

        # print(X_train.shape)
        # print(X_train[0])
        # print(y.shape)
        # print(y[0])
        #
        train_dataset = CustomDataset(X_train, y_train)
        test_dataset = CustomDataset(X_test, y_test)
        #
        # print(train_dataset[0])

        train_dl = self.make_train_dataloader()
        test_dl = self.make_test_dataloader()

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True, pin_memory=True
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=False
        )

        autoencoder = buil_compile_fit_ECG_NN(train_dataloader, train_dataset, test_dataloader, test_dataset)
        X_2d = autoencoder.call(X)

        X_resampled, y_resampled = method.fit_resample(X_2d, y)
        print(self.counter_dict_class(y_resampled))

        return X_resampled, y_resampled

    def counter_dict_class(self, tabular_column):
        """Creates the counter of classes in the tabular by column"""

        if type(tabular_column[0]) is tuple:
            all_labels = [label for labels in tabular_column for label in labels]
        elif type(tabular_column[0]) is str:
            all_labels = [label for label in tabular_column]
        # ax = sns.countplot(all_labels, order=[k for k, _ in result.most_common()], log=True)
        # ax.set_title('Number of classes with a class label')
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
        return Counter(all_labels)


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

    return residuary_prob

# def create_fit_balanced_mlbs(tabular: pd.DataFrame) ->



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
    dataset = PtbXlClassesSuperclassesBalanced(
        r"/Users/gerilda/Documents/itmm/ecg_analysis/data/raw",
        r"/Users/gerilda/Documents/itmm/ecg_analysis/data/processed",
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

    # dataset.balanced_by_imbalanced_learn_method(RandomOverSampler(random_state=0))
    # dataset.balanced_by_imbalanced_learn_method(SMOTE())
    #
    # dataset.balanced_by_imbalanced_learn_method(RandomUnderSampler(random_state=0))
    #
    # start_time = datetime.now()
    # # dataset.balanced_by_imbalanced_learn_method(ClusterCentroids(random_state=0)) # looooong time
    # print("--- %s seconds ---" % (datetime.now() - start_time))
    dataset.balanced_by_imbalanced_learn_method_witn_autoencoder(EditedNearestNeighbours())# need parameters
    # dataset.balanced_by_imbalanced_learn_method(RepeatedEditedNearestNeighbours())# need parameters
    # start_time = datetime.now()
    # # dataset.balanced_by_imbalanced_learn_method(CondensedNearestNeighbour(random_state=0))# looooong time
    # print("--- %s seconds ---" % (datetime.now() - start_time))
    # dataset.balanced_by_imbalanced_learn_method(OneSidedSelection(random_state=0))# need parameters
    # # dataset.balanced_by_imbalanced_learn_method(NeighbourhoodCleaningRule())# TypeError: bad operand type for unary ~: 'str'
    # # dataset.balanced_by_imbalanced_learn_method(InstanceHardnessThreshold(random_state=0,
    # #                                                                       estimator=LogisticRegression(
    # #                                                                                 solver='lbfgs',
    # #                                                                                 multi_class='auto')
    # #                                                                       ))# IndexError: arrays used as indices must be of integer (or boolean) type
    # dataset.balanced_by_imbalanced_learn_method(SMOTEENN(random_state=0))# need parameters
    # dataset.balanced_by_imbalanced_learn_method(SMOTETomek(random_state=0))
    dataset.balanced_by_imbalanced_learn_method(TomekLinks())


    # Create the data loaders
    # train_dl = dataset.make_train_dataloader()
    # test_dl = dataset.make_test_dataloader()
    # val_dl = dataset.make_val_dataloader()


if __name__ == "__main__":
    main()
