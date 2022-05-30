import matplotlib.pyplot as plt
import numpy as np
import os

def create_pie(classes, method, label_tag, class_tag, processed_data_folder):
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]

    fig, ax = plt.subplots()
    # "RepeatedEditedNearestNeighbours(sampling_strategy='majority',n_neighbors=3, kind_sel='mode', max_iter=100)"
    ax.set(aspect="equal", title="RepeatedEditedNearestNeighbours(sampling_strategy='majority',\nn_neighbors=3, kind_sel='mode', max_iter=100)" + "\n" + label_tag + ", " + class_tag)
    # ax.set(aspect="equal", title=method + "\n" + label_tag + ", " + class_tag)

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
    counter = {'CD': 2010, 'HYP': 1082, 'MI': 1961, 'NORM': 1684, 'STTC': 2350}
    create_pie(counter, "RepeatedEditedNearestNeighbours(sampling_strategy='majority',n_neighbors=3, kind_sel='mode', max_iter=100)",
               "Single-label", "superclasses", processed_data_folder)


if __name__ == "__main__":
    main()
