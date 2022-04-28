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

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, EditedNearestNeighbours,\
                                    RepeatedEditedNearestNeighbours, CondensedNearestNeighbour,\
                                    OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold
from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


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
        print(self.counter_dict_class(self.y_balance))

        # self.label_balance = self.prepare_labels(tabular)

    # def transform_labels(self, tabular: pd.DataFrame) -> np.ndarray:
    #     column_classes = np.array(list(self.counter.keys()))
    #     enc = OneHotEncoder(categories=column_classes, sparse=False)
    #
    #     # column_classes = ['NORM','CD','HYP','MI','STTC']
    #     enc.fit_transform(y[column_classes])
    #     enc.inverse_transform(y)

        #return self.classes_mlb.transform(tabular["y_balance"].to_numpy())

    @property
    def y_balance(self) -> np.ndarray:
        return self.y_balance_class[self.labels_indices_train]
        # return self.label_balance[self.labels_indices_train]

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
        X = self._waves_train[:, 0]
        y = self.y_balance
        print("Balansed by: ", method)

        X_resampled, y_resampled = method.fit_resample(X, y)
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


def one_label_preparing(probs, counter):
    residuary_prob = probs
    tmp = sys.maxsize

    for prob in probs:
        current = counter[prob]
        if current < tmp:
            tmp = current
            residuary_prob = prob

    return residuary_prob


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
    )

    dataset.balanced_by_imbalanced_learn_method(RandomOverSampler(random_state=0))
    dataset.balanced_by_imbalanced_learn_method(SMOTE())

    dataset.balanced_by_imbalanced_learn_method(RandomUnderSampler(random_state=0))

    start_time = datetime.now()
    dataset.balanced_by_imbalanced_learn_method(ClusterCentroids(random_state=0)) # looooong time
    print("--- %s seconds ---" % (datetime.now() - start_time))
    dataset.balanced_by_imbalanced_learn_method(EditedNearestNeighbours())# need parameters
    dataset.balanced_by_imbalanced_learn_method(RepeatedEditedNearestNeighbours())# need parameters
    start_time = datetime.now()
    dataset.balanced_by_imbalanced_learn_method(CondensedNearestNeighbour(random_state=0))# looooong time
    print("--- %s seconds ---" % (datetime.now() - start_time))
    dataset.balanced_by_imbalanced_learn_method(OneSidedSelection(random_state=0))# need parameters
    # dataset.balanced_by_imbalanced_learn_method(NeighbourhoodCleaningRule())# TypeError: bad operand type for unary ~: 'str'
    # dataset.balanced_by_imbalanced_learn_method(InstanceHardnessThreshold(random_state=0,
    #                                                                       estimator=LogisticRegression(
    #                                                                                 solver='lbfgs',
    #                                                                                 multi_class='auto')
    #                                                                       ))# IndexError: arrays used as indices must be of integer (or boolean) type
    dataset.balanced_by_imbalanced_learn_method(SMOTEENN(random_state=0))# need parameters
    dataset.balanced_by_imbalanced_learn_method(SMOTETomek(random_state=0))


    # Create the data loaders
    # train_dl = dataset.make_train_dataloader()
    # test_dl = dataset.make_test_dataloader()
    # val_dl = dataset.make_val_dataloader()


if __name__ == "__main__":
    main()
