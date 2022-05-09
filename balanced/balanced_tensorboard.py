import os
from datetime import datetime as dt

import matplotlib
from torch.utils.tensorboard.writer import SummaryWriter

from ecg_analysis.tracking import Stage

from ecg_analysis.tensorboard import TensorboardExperiment

def create_balanced_experiment_log_dir(root: str, method: str) -> str:
    """
    Create folder under provided root folder with name formed from timestamp.
    Return created folder path
    """

    dirname = str(method) + '_' + str(int(dt.timestamp(dt.now())))
    dirpath = os.path.join(root, dirname)
    os.makedirs(dirpath)

    return dirpath


class BalancedTensorboardExperiment(TensorboardExperiment):
    def __init__(self, log_path: str, method: str):
        super().__init__(log_path)

        self.log_dir = create_balanced_experiment_log_dir(root=log_path, method=method)
