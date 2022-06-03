import matplotlib.pyplot as plt
import numpy as np
import os

def create_pie(classes, method, title, label_tag, class_tag, processed_data_folder):
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]

    fig, ax = plt.subplots()
    ax.set(aspect="equal", title=title + "\n" + label_tag + ", " + class_tag)

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

    pie_label_tag = "singlelabel" if label_tag == "Single-label" else "multilabel"
    file_name = processed_data_folder + r"/pies/" + pie_label_tag + "/pie_" + method + '_' + pie_label_tag + "_" + class_tag + ".png"
    fig.savefig(
        file_name,
        bbox_inches='tight')

    ax.clear()

    return plt

def create_pie_two(classes, method, title, label_tag, class_tag, processed_data_folder):
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]

    fig, ax = plt.subplots()
    ax.set(aspect="equal", title=title + "\n" + label_tag + ", " + class_tag)

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

    pie_label_tag = "singlelabel" if label_tag == "Single-label" else "multilabel"
    file_name = processed_data_folder + r"/pies/" + pie_label_tag + "/pie_" + method + '_' + pie_label_tag + "_" + class_tag + ".png"
    fig.savefig(
        file_name,
        bbox_inches='tight')

    ax.clear()

    return plt


def main():
    processed_data_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r"/data/processed"
    # counter = {'CD': 2010, 'HYP': 1082, 'MI': 1961, 'NORM': 1684, 'STTC': 2350}
    # create_pie(counter, "RepeatedEditedNearestNeighbours(sampling_strategy='majority',n_neighbors=3, kind_sel='mode', max_iter=100)",
    #            "Single-label", "superclasses", processed_data_folder)

    # counter = {'CD': 2010, 'HYP': 1082, 'MI': 1961, 'NORM': 1725, 'STTC': 2350}
    # create_pie(counter, "AllKNN(sampling_strategy='majority',n_neighbors=3, kind_sel='mode')",
    #            "Single-label", "superclasses", processed_data_folder)

    # counter = {'NORM': 5578, 'STTC': 2313, 'CD': 1980, 'HYP': 1082, 'MI': 1922}
    # create_pie(counter, "TomekLinks", "TomekLinks(sampling_strategy='not minority')",
    #            "Single-label", "superclasses", processed_data_folder)

    # counter = {'NORM': 2054, 'STTC': 102, 'CD': 370, 'HYP': 3, 'MI': 402,
    #            'HYP/MI': 37, 'CD/MI': 280, 'CD/HYP': 2}
    # create_pie(counter, "EditedNearestNeighbours",
    #            "EditedNearestNeighbours(sampling_strategy='not minority',\nn_neighbours=5, kind_sel='mode')",
    #            "Multi-label", "subclasses", processed_data_folder)

    # counter = {'STTC': 66, 'CD': 392, 'HYP': 2, 'MI': 292,
    #            'HYP/MI': 37, 'CD/MI': 247, 'CD/MI': 280, 'CD/HYP/MI': 3, 'NORM': 1839}
    # create_pie(counter, "RepeatedEditedNearestNeighbours",
    #            "RepeatedEditedNearestNeighbours(sampling_strategy='not minority',\nn_neighbours=5, kind_sel='mode')",
    #            "Multi-label", "subclasses", processed_data_folder)

    # counter = {'NORM': 1839, 'STTC': 66, 'CD': 392, 'HYP': 2, 'MI': 292,
    #            'HYP/MI': 37, 'CD/MI': 247, 'CD/MI': 280, 'CD/HYP/MI': 3}
    # create_pie(counter, "AllKNN",
    #            "RepeatedEditedNearestNeighbours(sampling_strategy='not minority',\nn_neighbours=5, kind_sel='mode')",
    #            "Multi-label", "subclasses", processed_data_folder)

    # counter = {'NORM': 3878, 'CD': 1980, 'STTC': 1932, 'MI': 1076, 'HYP': 982,
    #            'OTHERS': 1995}
    # counter_other = {
    #     'MI/CD': 536,
    #     'STTC/HYP': 410,
    #     'STTC/CD': 377,
    #     'STTC/CD/HYP': 118,
    #     'STTC/HYP/MI': 1,
    #     'STTC/MI': 229,
    #     'CD/HYP': 155,
    #     'MI/CD/STTC': 70,
    #     'HYP/STTC/MI': 1,
    #     'MI/CD/HYP': 50,
    #     'MI/CD/HYP/STTC': 49}
    # create_pie(counter, "TomekLinks",
    #                "TomekLinks(sampling_strategy='not minority')",
    #                "Multi-label", "subclasses", processed_data_folder)
    # create_pie(counter_other, "TomekLinks",
    #            "TomekLinks(sampling_strategy='not minority')",
    #            "Multi-label", "others subclasses", processed_data_folder)

    # counter = {'NORM': 2844, 'STTC': 892, 'CD': 878, 'MI': 382, 'HYP': 216,
    #            'OTHERS': 813}
    # counter_other = {
    #     'STTC/HYP/MI': 20,#
    #     'MI/CD': 216,#
    #     'STTC/HYP': 170,#
    #     'STTC/CD': 175,#
    #     'STTC/CD/HYP': 32,#
    #     'STTC/MI': 91,#
    #     'CD/HYP': 21,#
    #     'MI/CD/STTC': 26,#
    #     'HYP/STTC/MI': 1,
    #     'MI/CD/HYP': 21,#
    #     'MI/CD/HYP/STTC': 40}
    # create_pie(counter, "OneSidedSelection",
    #            "OneSidedSelection(sampling_strategy='not minority', n_neighbours=2)",
    #            "Multi-label", "subclasses", processed_data_folder)
    # create_pie(counter_other, "OneSidedSelection",
    #            "OneSidedSelection(sampling_strategy='not minority', n_neighbours=2)",
    #            "Multi-label", "others subclasses", processed_data_folder)

    # counter = {'NORM': 1839, 'STTC': 1967, 'CD': 2010, 'MI': 1103, 'HYP': 153,
    #            'OTHERS': 2366}
    # counter_other = {
    #     'STTC/HYP/MI': 417,
    #     'MI/CD': 554,
    #     'MI/CD/HYP': 19,
    #     'STTC/HYP': 417,
    #     # 'HYP/STTC/MI': 1,
    #     'STTC/CD': 383,
    #     'STTC/CD/HYP': 121,
    #     'STTC/MI': 232,
    #     'MI/CD/HYP/STTC': 44,
    #     'CD/HYP': 157,
    #     'MI/CD/STTC': 22}
    # create_pie(counter, "NeighbourhoodCleaningRule",
    #            "NeighbourhoodCleaningRule(sampling_strategy='majority')",
    #            "Multi-label", "subclasses", processed_data_folder)
    # create_pie(counter_other, "NeighbourhoodCleaningRule",
    #            "NeighbourhoodCleaningRule(sampling_strategy='majority')",
    #            "Multi-label", "others subclasses", processed_data_folder)

    # counter = {'NORM': 2291, 'CD': 231, 'STTC': 201, 'MI': 212, 'HYP': 2,
    #            'OTHERS': 282}
    # counter_other = {
    #     'MI/CD/HYP/STTC': 12,
    #     'MI/CD': 76,
    #     'STTC/HYP': 3,
    #     'HYP/MI': 37,
    #     'STTC/CD': 3,
    #     'STTC/MI': 73,
    #     'CD/HYP': 2,
    #     'CD/MI': 76}
    # create_pie(counter, "AllKNN",
    #            "AllKNN(sampling_strategy='not_minority',\nkind_sel='mode', n_neighbours=5)",
    #            "Multi-label", "subclasses", processed_data_folder)
    # create_pie(counter_other, "AllKNN",
    #            "AllKNN(sampling_strategy='not_minority',\nkind_sel='mode', n_neighbours=5)",
    #            "Multi-label", "others subclasses", processed_data_folder)

    counter = {'NORM': 5631,
               'CD': 2010,
               'STTC': 1967,
               'MI': 1103,
               'HYP': 153,
               'OTHERS': 2170}
    counter_other = {
        'MI/CD': 554,
        'STTC/HYP/MI': 1,
        'STTC/HYP': 417,
        'HYP/STTC/MI': 1,
        'STTC/CD': 383,
        'MI/HYP': 37,
        'MI/STTC': 232,
        'CD/HYP': 157,
        'STTC/CD/HYP': 121,
        'MI/HYP/STTC': 89,
        'MI/CD/STTC': 72,
        'MI/CD/HYP': 54,
        'MI/CD/HYP/STTC': 52,
        }
    create_pie(counter, "Imbalanced",
               "Imbalanced",
               "Multi-label", "subclasses", processed_data_folder)
    create_pie(counter_other, "Imbalanced",
               "Imbalanced",
               "Multi-label", "others subclasses", processed_data_folder)

if __name__ == "__main__":
    main()
