import torch

from ecg_analysis.dataset import PtbXlWrapper
from ecg_analysis.models import SimpleConv
from ecg_analysis.runner import Runner, run_epoch, run_test
from ecg_analysis.tensorboard import TensorboardExperiment

# Hyperparameters
EPOCH_COUNT = 20
LR = 8e-4
BATCH_SIZE = 128
LOG_PATH = "./runs"

# Data configuration
DATA_DIR = "data"

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{DEVICE=}")


def main():
    dataset = PtbXlWrapper(
        r"data/raw",
        r"data/processed",
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

    # Create the data loaders
    train_dl = dataset.make_train_dataloader()
    test_dl = dataset.make_test_dataloader()
    val_dl = dataset.make_val_dataloader()

    # Model and optimizer
    model = SimpleConv().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create the runners
    test_runner = Runner(test_dl, model, device=DEVICE)
    train_runner = Runner(train_dl, model, optimizer, device=DEVICE)
    val_runner = Runner(val_dl, model, device=DEVICE)

    # Setup the experiment tracker
    tracker = TensorboardExperiment(LOG_PATH)

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
            f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
            f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
            sep='/n',
            end='/n/n',
        )

        # Reset the runners
        train_runner.reset()
        test_runner.reset()

        # Flush the tracker after every epoch for live updates
        tracker.flush()

    run_test(test_runner, tracker)


if __name__ == "__main__":
    main()
