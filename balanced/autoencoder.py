# import tensorflow as tf
# from tensorflow.keras import callbacks, models, metrics, optimizers, losses, layers
# from tensorflow.keras.models import Model
import torch
from torch import nn, optim
from tqdm import tqdm
import time

from balanced.autoencoder_dataset import LRScheduler, EarlyStopping


def conv_block(
        input_size: int,
        output_size: int,
        *,
        kernel_size: int,
        dropout_p: float=0.5,
) -> nn.Sequential:
    block = nn.Sequential(
        nn.Conv1d(
            input_size,
            output_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        ),
        nn.BatchNorm1d(output_size),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_p),
    )

    return block


class ECG_NN(nn.Module):
    def __init__(self, **kwargs):
        super(ECG_NN, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(input_size=kwargs["input_shape"], output_size=6, kernel_size=5),
            conv_block(input_size=6, output_size=1, kernel_size=5),
        )

        self.decoder = nn.Sequential(
            conv_block(input_size=1, output_size=6, kernel_size=5),
            conv_block(input_size=6, output_size=kwargs["input_shape"], kernel_size=5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def call_encoder(self, x):
        encoded = self.encoder(x)
        return encoded

    def call_decoder(self, x):
        decoded = self.decoder(x)
        return decoded


# def build_tf_callbacks():
#     reduce_lr_on_plateau = callbacks.ReduceLROnPlateau(
#         monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7, verbose=1
#     )
#     early_stopping = callbacks.EarlyStopping(
#         monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, restore_best_weights=True
#     )
#     terminate_nn_nan = callbacks.TerminateOnNaN()
#     return [reduce_lr_on_plateau, early_stopping, terminate_nn_nan]

# def fit(train_loader, model, device, optimizer, criterion, epochs):
#     for epoch in range(epochs):
#         loss = 0
#         for batch_features, _ in train_loader:
#             # reshape mini-batch data to [N, 784] matrix
#             # load it to the active device
#             batch_features = batch_features.view(-1, 784).to(device)
#
#             # reset the gradients back to zero
#             # PyTorch accumulates gradients on subsequent backward passes
#             optimizer.zero_grad()
#
#             # compute reconstructions
#             outputs = model(batch_features)
#
#             # compute training reconstruction loss
#             train_loss = criterion(outputs, batch_features)
#
#             # compute accumulated gradients
#             train_loss.backward()
#
#             # perform parameter update based on current gradients
#             optimizer.step()
#
#             # add the mini-batch training loss to epoch loss
#             loss += train_loss.item()
#
#         # compute the epoch training loss
#         loss = loss / len(train_loader)
#
#         # display the epoch training loss
#         print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))


# training function
def train_step(model, train_dataloader, train_dataset, optimizer, criterion, device):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    # print(len(train_dataset))
    # print(train_dataloader.batch_size)
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / train_dataloader.batch_size))
    for i, data in prog_bar:
        counter += 1
        # data = target, X=X
        # data, target = data[0].to(device), data[0].to(device)
        data = data[0].to(device)
        # total += target.size(0)
        total += data.size(0)
        optimizer.zero_grad()
        outputs = model(data) # (batch_features)
        # loss = criterion(outputs, target)
        loss = criterion(outputs, data)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        # train_running_correct += (preds == target).sum().item()
        train_running_correct += (preds == data).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    return train_loss


# validation function
def val_step(model, test_dataloader, test_dataset, criterion, device):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(test_dataloader), total=int(len(test_dataset) / test_dataloader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            # X = X проверка в энкодере, поэтому y=X
            # data, target = data[0].to(device), data[1].to(device)
            data = data[0].to(device)
            # total += data.target(0)
            total += data.size(0)
            outputs = model(data)
            # loss = criterion(outputs, target)
            loss = criterion(outputs, data)

            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            # val_running_correct += (preds == target).sum().item()
            val_running_correct += (preds == data).sum().item()

        val_loss = val_running_loss / counter
        return val_loss


def fit(model, train_dataloader, train_dataset, test_dataloader, test_dataset, optimizer, criterion, device,
        epochs, is_lr_scheduler=True, is_early_stopping=True):
    """

    """
    # lists to store per-epoch loss and accuracy values
    train_loss, test_loss = [], []
    start = time.time()
    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = train_step(
            model, train_dataloader, train_dataset, optimizer, criterion, device
        )
        test_epoch_loss = val_step(
            model, test_dataloader, test_dataset, criterion, device
        )
        train_loss.append(train_epoch_loss)
        test_loss.append(test_epoch_loss)
        if is_lr_scheduler:
            lr_scheduler(test_epoch_loss)
        if is_early_stopping:
            early_stopping(test_epoch_loss)
            if early_stopping.early_stop:
                break

        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {test_epoch_loss:.4f}')

    end = time.time()
    print(f"Training time: {(end - start) / 60:.3f} minutes")


def buil_compile_fit_ECG_NN(train_dataloader, train_dataset, test_dataloader, test_dataset):
    autoencoder = ECG_NN(input_shape=12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # autoencoder.build([None, 1000, 12])
    model = autoencoder.to(device)

    # autoencoder.compile(
    #     optimizer=optim.Adam(learning_rate=1e-05),
    #     loss='mae'
    # )
    optimizer = optim.Adam(model.parameters(), lr=1e-05)
    # mean-squared error loss
    criterion = nn.MSELoss()

    fit(
        model,
        train_dataloader,
        train_dataset,
        test_dataloader,
        test_dataset,
        optimizer,
        criterion,
        device,
        epochs=100,
        is_lr_scheduler=True,
        is_early_stopping=True
    )
    # autoencoder.fit(
    #     x=X,
    #     y=X,
    #     batch_size=16,
    #     epochs=15,
    #     verbose='auto',
    #     # callbacks=build_tf_callbacks(),
    #     validation_split=0.2,
    #     validation_data=None,
    # )

    return autoencoder
