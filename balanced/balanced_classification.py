import os

import torch

from ecg_analysis.models import ResidualConvNetMixed
from ecg_analysis.runner import Runner, run_epoch, run_test
from ecg_analysis.tensorboard import TensorboardExperiment

# Hyperparameters
EPOCH_COUNT = 6
LR = 8e-4
BATCH_SIZE = 128
LOG_PATH = "./balanced_runs"

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"{DEVICE=}")


def balanced_classification(dataset, X_resampled, y_resampled, method):

    # Create the data loaders
    train_dl = dataset.make_balanced_train_dataloader(X_resampled, y_resampled)
    test_dl = dataset.make_balanced_test_dataloader()
    val_dl = dataset.make_balanced_val_dataloader()

    # Model and optimizer
    model = ResidualConvNetMixed(
        [12, 64, 128, 256, 512],
        [2, 2, 2, 2],
        [5, 5, 3, 3],
        [0.3 for __ in range(4)],
        [128],
        5
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create the runners
    test_runner = Runner(test_dl, model, device=DEVICE)
    train_runner = Runner(train_dl, model, optimizer, device=DEVICE)
    val_runner = Runner(val_dl, model, device=DEVICE)

    # Setup the experiment tracker
    tracker = TensorboardExperiment(LOG_PATH, method)

    # Run the epochs
    for epoch_id in range(EPOCH_COUNT):
        run_epoch(
            train_runner,
            val_runner,
            tracker,
            epoch_id,
        )

        # Compute Average Epoch Metrics
        print(
            f"[Epoch: {epoch_id + 1}/{EPOCH_COUNT}]",
            f"Val Accuracy: {val_runner.avg_accuracy: 0.4f}",
            f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
            sep='/n',
            end='/n/n',
        )

        # Reset the runners
        train_runner.reset()
        val_runner.reset()
        # test_runner.reset()

        # Flush the tracker after every epoch for live updates
        tracker.flush()

    classes = (list(dataset.superclasses_mlb.classes_))

    run_test(test_runner, tracker, classes)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(tracker.log_dir, "model.pt"))

    return
