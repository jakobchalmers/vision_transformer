# %%
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from data_generation import SequenceDataset, ImageDataset
import sys
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device=}")

# %%

FILE_NAME = "data/particle_dataset_4000.pth"
TEST_FILE_NAME = "data/particle_test_dataset_1000.pth"

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
    del sys.modules["vision_modules"]
    from modules import (
        PrintLayer,
        ConvolutionalAutoencoder,
    )
    from vision_modules import VisionTransformerAutoencoder
except KeyError:
    from modules import (
        PrintLayer,
        ConvolutionalAutoencoder,
    )
    from vision_modules import VisionTransformerAutoencoder


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
    def __init__(self, latent_dim, seq_len, num_transformer_layers, nhead):
        super(TransformerPredictor, self).__init__()
        time_encoding_dim = 2
        self.time2vector = Time2Vector(seq_len=seq_len, latent_dim=latent_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim + time_encoding_dim,
                nhead=nhead,
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
    optimizer = torch.optim.Adam(
        predictor_model.parameters(), lr=learning_rate
    )

    losses = {"Train": [], "Test": []}
    for epoch in range(epochs):
        train_loss = train(
            predictor_model,
            sequence_train_loader,
            criterion,
            optimizer,
            autoencoder,
        )
        test_loss = test(predictor_model, sequence_test_loader, criterion, autoencoder)

        losses["Train"].append(train_loss)
        losses["Test"].append(test_loss)

        print(f"Epoch {epoch} Loss: {train_loss} Test Loss: {test_loss}")
        # scheduler_convolution.step()

    return losses



def image_prediction_test(predictor_model, autoencoder, sequence_loader: DataLoader):
    x = random.sample(list(sequence_loader), 1)[0]
    with torch.no_grad():
        original = x.clone()
        batch_size, seq_len, _, _, _ = x.shape
        x = x.to(device)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = autoencoder.encoder(x.permute(0, 3, 1, 2))
        x = x.unflatten(0, (batch_size, seq_len))
        x_test = x[:, :-1, :]  # -> (batch_size, seq_len, latent_dim)
        y_test = x[:, -1, :]  # -> (batch_size, latent_dim)

        out = predictor_model(x_test)  # -> (batch_size, latent_dim + time_dim)
        out = out[:, :-2]  # -> (batch_size_latent_dim)

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

        ax[i, num_frames].imshow(out[i, :, :].detach().cpu().permute(1, 2, 0))
        ax[i, num_frames].axis("off")
        if i == 0:
            ax[i, num_frames].set_title("Predicted 10th")

    plt.show()


def sequence_prediction_test(
    predictor_model, autoencoder, sequence_loader: DataLoader, num_predictions: int
):
    x = random.sample(list(sequence_loader), 1)[0]
    with torch.no_grad():
        original = x.clone()
        batch_size, seq_len, _, _, _ = x.shape

        x = x.to(device).flatten(start_dim=0, end_dim=1)
        x = autoencoder.encoder(x.permute(0, 3, 1, 2))
        x = x.unflatten(
            0, (batch_size, seq_len)
        )  # -> (batch_size, seq_len, latent_dim)

        x_test = x[:, :-1, :]  # -> (batch_size, seq_len, latent_dim)
        for i in range(num_predictions):

            prediction = predictor_model(x_test[:, -(seq_len - 1) :, :])[
                :, :-2
            ].unsqueeze(
                1
            )  # -> (batch_size, latent_dim, d_embed)
            x_test = torch.cat(
                (x_test, prediction), dim=1
            )  # -> (batch_size, seq_len + i, d_embed)
        latent_predictions = x_test[
            :, (seq_len - 1) :, :
        ]  # -> (batch_size, num_predictions, d_embed)
        flattened_latent_predictions = latent_predictions.flatten(
            start_dim=0, end_dim=1
        )  # -> (batch_size * num_predictions, d_embed)

        flattened_predictions: torch.Tensor = autoencoder.decoder(
            flattened_latent_predictions
        )  # -> (batch_size * num_predictions, 1, 64, 64)
        predictions: torch.Tensor = flattened_predictions.unflatten(
            dim=0, sizes=(batch_size, num_predictions)
        )  # -> (batch_size, num_predictions, 1, 64, 64)
        

    num_rows = 4
    num_original_frames = 5
    fig, ax = plt.subplots(
        num_rows,
        num_original_frames + num_predictions,
        figsize=(2 * (num_original_frames + num_predictions), 2 * num_rows),
    )
    # Remove vertical space between subplots
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i in range(num_rows):
        for j in range(num_original_frames):
            ax[i, j].imshow(original[i, j - num_original_frames].squeeze())
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(f"{10-num_original_frames+j+1}th")

        for k in range(num_predictions):
            ax[i, num_original_frames + k].imshow(predictions[i, k, :, :, :].detach().cpu().permute(1, 2, 0))
            ax[i, num_original_frames + k].axis("off")
            if i == 0:
                ax[i, num_original_frames + k].set_title(f"Predicted {seq_len+k}th")

    plt.show()


# %% Train Model with Convolutional Autoencoder

# Load autoencoder
convolutional_autoencoder: ConvolutionalAutoencoder = torch.load(
    # "models/convolutional_autoencoder_4000.pth"
    "models/convolutional_autoencoder.pth"
)
model_convolution = TransformerPredictor(
    latent_dim=4, seq_len=9, num_transformer_layers=4, nhead=3,
).to(device)

# %% Train convolution variant
losses = train_for_epochs(
    model_convolution,
    autoencoder=convolutional_autoencoder,
    epochs=10,
    learning_rate=0.01,
)


# %% Test Convolutional Autoencoder for 1 prediction
image_prediction_test(
    model_convolution,
    autoencoder=convolutional_autoencoder,
    sequence_loader=sequence_test_loader,
)

# %% Test Convolutional Autoencoder for multiple predictions

sequence_prediction_test(
    predictor_model=model_convolution,
    autoencoder=convolutional_autoencoder,
    sequence_loader=sequence_test_loader,
    num_predictions=10,
)

# %% Train Model with Vision Transformer Autoencoder
import dill
from vision_modules import VisionTransformerAutoencoder
# Load autoencoder
vision_autoencoder: VisionTransformerAutoencoder = torch.load(
    "models/vision_transformer_autoencoder_4000.pth",
    pickle_module=dill,
)
model_vision = TransformerPredictor(
    latent_dim=4, seq_len=9, num_transformer_layers=4, nhead=3,
).to(device)

# # %%
# import random
# for data in sequence_test_loader:
#     data = torch.flatten(data, start_dim=0, end_dim=1)
#     original = data.clone()
#     data = data.permute(0, 3, 1, 2).to(device)
#     print(data.shape)

#     out = vision_autoencoder(data).detach().cpu().permute(0, 2, 3, 1)
#     rand = random.randint(0, 319)
#     print(rand)

#     plt.imshow(original[rand, :, :, :])
#     plt.show()
#     plt.imshow(out[rand, :, :, :])

#     break


# %%
train_for_epochs(
    model_vision,
    autoencoder=vision_autoencoder,
    epochs=10,
    learning_rate=0.01,
)

# %% Test Convolutional Autoencoder (images)
image_prediction_test(
    predictor_model=model_vision,
    autoencoder=vision_autoencoder,
    sequence_loader=sequence_test_loader,
)

# %% Test Convolutional Autoencoder for multiple predictions
sequence_prediction_test(
    predictor_model=model_vision,
    autoencoder=vision_autoencoder,
    sequence_loader=sequence_test_loader,
    num_predictions=10,
)



# %% Convolutional Variant: Vary number of Transformer Layers
numbers_of_transformer_layers = [1, 2, 4, 8]
convolution_models_data = {}

for i, num_layers in enumerate(numbers_of_transformer_layers):
    model_convolution = TransformerPredictor(
        latent_dim=4, seq_len=9, num_transformer_layers=num_layers, nhead=3
    ).to(device)
    optimizer_convolution = torch.optim.Adam(model_convolution.parameters(), lr=0.01)

    epochs = 10
    losses = train_for_epochs(
        predictor_model=model_convolution,
        autoencoder=convolutional_autoencoder,
        epochs=epochs,
        learning_rate=0.01,
    )

    epochs = 10
    losses = train_for_epochs(
        predictor_model=model_convolution,
        autoencoder=convolutional_autoencoder,
        epochs=epochs,
        learning_rate=0.001,
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
numbers_of_transformer_layers = [1, 2, 4, 8]
vision_models_data = {}

for i, num_layers in enumerate(numbers_of_transformer_layers):
    model_vision = TransformerPredictor(
        latent_dim=4, seq_len=9, num_transformer_layers=num_layers, nhead=3
    ).to(device)

    epochs = 10
    losses = train_for_epochs(
        predictor_model=model_vision,
        autoencoder=vision_autoencoder,
        epochs=epochs,
        learning_rate=0.01,
    )

    epochs = 10
    losses = train_for_epochs(
        predictor_model=model_vision,
        autoencoder=vision_autoencoder,
        epochs=epochs,
        learning_rate=0.001,
    )


    vision_models_data[num_layers] = {}
    vision_models_data[num_layers]["model"] = model_vision
    vision_models_data[num_layers]["losses"] = losses

# %% Vision Transformer Variant: Plot varying number of Transformer Layers

colors = ["orange", "blue", "green", "red", "black"]
assert len(colors) >= len(vision_models_data)
for i, (num_layers, data) in enumerate(vision_models_data.items()):
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

plt.title("Losses for Vision Transformer autoencodern frame predictor for different numbers of layers")
plt.legend()
plt.show()


# %% Load multiple particle data


MULTIPLE_PARTICLE_FILE_NAME = "data/multiple_particle_dataset_2000.pth"
MULTIPLE_PARTICLE_TEST_FILE_NAME = "data/multiple_particle_test_dataset_200.pth"

print("Loading...")
sequence_dataset: SequenceDataset = torch.load(MULTIPLE_PARTICLE_FILE_NAME)
sequence_test_dataset: SequenceDataset = torch.load(MULTIPLE_PARTICLE_TEST_FILE_NAME)

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


# %% Train Model with Convolutional Autoencoder
import dill

# Load autoencoder
convolutional_autoencoder: ConvolutionalAutoencoder = torch.load(
    "models/multiple_particle_convolutional_autoencoder_2000.pth"
)
model_convolution = TransformerPredictor(
    latent_dim=8, seq_len=9, num_transformer_layers=4, nhead=5,
).to(device)

# %%

losses = train_for_epochs(
    model_convolution,
    autoencoder=convolutional_autoencoder,
    epochs=10,
    learning_rate=0.001,
)


# %% Test Convolutional Autoencoder for 1 prediction
image_prediction_test(
    model_convolution,
    autoencoder=convolutional_autoencoder,
    sequence_loader=sequence_test_loader,
)

# %% Test Convolutional Autoencoder for multiple predictions

sequence_prediction_test(
    predictor_model=model_convolution,
    autoencoder=convolutional_autoencoder,
    sequence_loader=sequence_test_loader,
    num_predictions=10,
)