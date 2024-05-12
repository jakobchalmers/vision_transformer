# %%
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from data_generation import SequenceDataset, ImageDataset
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device=}")

# %%

FILE_NAME = "data/particle_dataset_500.pth"
TEST_FILE_NAME = "data/particle_test_dataset_100.pth"

print("Loading...")
sequence_dataset: SequenceDataset = torch.load(FILE_NAME)
sequence_test_dataset: SequenceDataset = torch.load(TEST_FILE_NAME)

sequence_train_loader = DataLoader(
    sequence_dataset,
    batch_size=32,
    shuffle=True,
)

sequence_test_loader = DataLoader(
    sequence_test_dataset,
    batch_size=32,
    shuffle=False,
)


image_dataset = ImageDataset(sequence_dataset)
image_test_dataset = ImageDataset(sequence_test_dataset)

image_train_loader = DataLoader(
    image_dataset,
    batch_size=32,
    shuffle=True,
)

image_test_loader = DataLoader(
    image_test_dataset,
    batch_size=32,
    shuffle=False,
)
print("Done")

# %%

try:
    del sys.modules["modules"]
    from modules import (
        PrintLayer,
        ConvolutionalDecoder,
        ConvolutionalEncoder,
        ConvolutionalAutoencoder,
    )
except KeyError:
    from modules import (
        PrintLayer,
        ConvolutionalDecoder,
        ConvolutionalEncoder,
        ConvolutionalAutoencoder,
    )


class Time2Vector(nn.Module):
    def __init__(self, seq_len, latent_dim):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.linear_weights = nn.Parameter(
            torch.rand(seq_len)
        )  # requires grad by default?
        self.linear_biases = nn.Parameter(torch.rand(seq_len))
        self.periodic_weights = nn.Parameter(torch.rand(seq_len))
        self.periodic_biases = nn.Parameter(torch.rand(seq_len))

    def forward(self, x):
        x_embedding = x.clone()
        x_embedding = x_embedding.mean(dim=-1)

        time_linear = (
            self.linear_weights * x_embedding + self.linear_biases
        ).unsqueeze(-1)
        time_periodic = torch.sin(
            self.periodic_weights * x_embedding + self.periodic_biases
        ).unsqueeze(-1)

        x = torch.cat([x, time_linear, time_periodic], dim=-1)
        return x


class TransformerPredictor(nn.Module):
    def __init__(self, latent_dim, seq_len, num_transformer_layers):
        super(TransformerPredictor, self).__init__()
        time_encoding_dim = 2
        self.time2vector = Time2Vector(seq_len=seq_len, latent_dim=latent_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim + time_encoding_dim,
                nhead=3,
                dim_feedforward=256,
                dropout=0.1,
                activation="relu",
            ),
            num_layers=num_transformer_layers,
        )

    def forward(self, x):
        x = self.time2vector(x)
        x = self.transformer(x)
        return x[:, -1, :]


# %% Train
def train(model, train_loader, criterion, optimizer, autoencoder):
    model.train()
    total_loss = 0
    for x in sequence_train_loader:
        batch_size, seq_len, _, _, _ = x.shape
        x = x.to(device)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        with torch.no_grad():
            x = autoencoder.encoder(x.permute(0, 3, 1, 2))
        x = x.unflatten(0, (batch_size, seq_len))
        x_train = x[:, :-1, :]
        y_train = x[:, -1, :]

        out = model(x_train)[:, :-2]

        loss = criterion(out, y_train)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


def test(model, test_loader, criterion, autoencoder):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for x in test_loader:
            batch_size, seq_len, _, _, _ = x.shape
            x = x.to(device)
            x = torch.flatten(x, start_dim=0, end_dim=1)
            x = autoencoder.encoder(x.permute(0, 3, 1, 2))
            x = x.unflatten(0, (batch_size, seq_len))
            x_test = x[:, :-1, :]
            y_test = x[:, -1, :]

            out = model(x_test)[:, :-2]

            loss = criterion(out, y_test)
            total_loss += loss.item()

    return total_loss / len(test_loader)


def train_for_epochs(predictor_model, autoencoder, epochs=50, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer_convolution = torch.optim.Adam(
        model_convolution.parameters(), lr=learning_rate
    )

    # scheduler_convolution = torch.optim.lr_scheduler.MultiplicativeLR(
    #     optimizer_convolution, lr_lambda=lambda epoch: 0.95
    # )

    losses = {"Train": [], "Test": []}
    for epoch in range(epochs):
        train_loss = train(
            predictor_model,
            sequence_train_loader,
            criterion,
            optimizer_convolution,
            autoencoder,
        )
        test_loss = test(predictor_model, sequence_test_loader, criterion, autoencoder)

        losses["Train"].append(train_loss)
        losses["Test"].append(test_loss)

        print(f"Epoch {epoch} Loss: {train_loss} Test Loss: {test_loss}")
        # scheduler_convolution.step()

    return losses


def image_prediction_test(predictor_model, autoencoder, sequence_loader):
    with torch.no_grad():
        x = next(iter(sequence_loader))
        original = x.clone()
        batch_size, seq_len, _, _, _ = x.shape
        x = x.to(device)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = autoencoder.encoder(x.permute(0, 3, 1, 2))
        x = x.unflatten(0, (batch_size, seq_len))
        x_test = x[:, :-1, :]
        y_test = x[:, -1, :]

        out = predictor_model(x_test)[:, :-2]

        out = autoencoder.decoder(out)
        y_test = autoencoder.decoder(y_test)

    num_rows = 4
    num_frames = 5
    fig, ax = plt.subplots(
        num_rows, num_frames + 1, figsize=(2 * (num_frames + 1), 2 * num_rows)
    )
    # Remove vertical space between subplots
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i in range(num_rows):
        for j in range(num_frames):
            ax[i, j].imshow(original[i, j - num_frames].squeeze())
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(f"{10-num_frames+j+1}th")

        ax[i, num_frames].imshow(out[i].detach().cpu().permute(1, 2, 0))
        ax[i, num_frames].axis("off")
        if i == 0:
            ax[i, num_frames].set_title("Predicted 10th")

    plt.show()


# %% Train Model with Convolutional Autoencoder

# Load autoencoder
convolutional_autoencoder: ConvolutionalAutoencoder = torch.load(
    "models/convolutional_autoencoder.pth"
)
model_convolution = TransformerPredictor(
    latent_dim=4, seq_len=9, num_transformer_layers=4
).to(device)

train_for_epochs(
    model_convolution,
    autoencoder=convolutional_autoencoder,
    epochs=50,
    learning_rate=0.01,
)


# %% Test Convolutional Autoencoder (images)
image_prediction_test(
    model_convolution,
    autoencoder=convolutional_autoencoder,
    sequence_loader=sequence_test_loader,
)


# %% Train Model with Vision Transformer Autoencoder
# TODO

# %% Test Convolutional Autoencoder (images)
# TODO


# %% Convolutional Variant: Vary number of Transformer Layers
numbers_of_transformer_layers = [1, 2, 4, 8]
convolution_models_data = {}

for i, num_layers in enumerate(numbers_of_transformer_layers):
    model_convolution = TransformerPredictor(
        latent_dim=4, seq_len=9, num_transformer_layers=num_layers
    ).to(device)
    optimizer_convolution = torch.optim.Adam(model_convolution.parameters(), lr=0.01)
    # scheduler_convolution = torch.optim.lr_scheduler.MultiplicativeLR(
    #     optimizer_convolution, lr_lambda=lambda epoch: 0.95
    # )

    epochs = 30
    losses = train_for_epochs(
        predictor_model=model_convolution,
        autoencoder=convolutional_autoencoder,
        epochs=epochs,
        learning_rate=0.01,
    )

    convolution_models_data[num_layers] = {}
    convolution_models_data[num_layers]["model"] = model_convolution
    convolution_models_data[num_layers]["losses"] = losses


# %% Convolutional Variant: Plot varying number of Transformer Layers


colors = ["orange", "blue", "green", "red", "black"]
assert len(colors) >= len(convolution_models_data)
for i, (num_layers, data) in enumerate(convolution_models_data.items()):
    losses = data["losses"]
    plt.plot(
        losses["Train"],
        linestyle="--",
        label=f"Train, {num_layers} Layer(s)",
        color=colors[i],
    )
    plt.plot(
        losses["Test"],
        label=f"Test, {num_layers} Layer(s)",
        color=colors[i],
    )

plt.title("Losses for transformer predictor for different numbers of layers")
plt.legend()
plt.show()

# %% Vision Transformer Variant: Vary number of Transformer Layers
# TODO

# %% Vision Transformer Variant: Plot varying number of Transformer Layers
# TODO
