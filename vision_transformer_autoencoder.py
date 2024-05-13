# %%
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from data_generation import SequenceDataset, ImageDataset
import numpy as np
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device=}")

# %% Data Loading

FILE_NAME = "data/particle_dataset_500.pth"
TEST_FILE_NAME = "data/particle_test_dataset_100.pth"

print("Loading...")
sequence_dataset: SequenceDataset = torch.load(FILE_NAME)
sequence_test_dataset: SequenceDataset = torch.load(TEST_FILE_NAME)


image_dataset = ImageDataset(sequence_dataset)
image_test_dataset = ImageDataset(sequence_test_dataset)


train_loader = DataLoader(
    image_dataset,
    batch_size=32,
    shuffle=True,
)

test_loader = DataLoader(
    image_test_dataset,
    batch_size=32,
    shuffle=False,
)
print("Done")

# %% Functions ##############################################################

try:
    del sys.modules["modules"]
    from modules import PrintLayer, ConvolutionalDecoder
except KeyError:
    from modules import PrintLayer, ConvolutionalDecoder


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for input in train_loader:
        input: torch.Tensor = input.permute(0, 3, 1, 2).to(device)
        output = model(input)

        loss = criterion(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    mean_loss = total_loss / len(train_loader)
    return mean_loss


def test(model, data_loader, criterion, device):
    model.train()
    total_loss = 0
    for input in data_loader:
        input: torch.Tensor = input.permute(0, 3, 1, 2).to(device)
        output = model(input)

        loss = criterion(input, output)
        total_loss += loss.item()

    mean_loss = total_loss / len(data_loader)
    return mean_loss


# %% Modules

from vision_modules import VisionTransformerAutoencoder

# %% Setup for training ###########################################################

gpu_model = VisionTransformerAutoencoder(
    dim_embedding=4,
    latent_dim=4,
    patch_size=32,
    number_of_heads=1,
    number_of_transformer_layers=1,
    device=device,
).to(device)
criterion = nn.MSELoss()
# optimizer = optim.Adam(gpu_model.parameters(), lr=0.001)

initial_train_loss = test(gpu_model, train_loader, criterion, device)
print(f"{initial_train_loss=}")
initial_test_loss = test(gpu_model, test_loader, criterion, device)
print(f"{initial_test_loss=}")

last_epoch_number = 0
learning_rate = 0.01

# %% Train #########################################################################
num_epochs = 10
learning_rate = 0.0001
optimizer = optim.Adam(gpu_model.parameters(), lr=learning_rate)
for i, epoch in tqdm(enumerate(range(num_epochs))):
    train_loss = train(gpu_model, train_loader, criterion, optimizer, device)
    test_loss = test(gpu_model, test_loader, criterion, device)
    # learning_rate *= 0.9

    print(
        f"Epoch {i+1+last_epoch_number}: Train loss {train_loss}, Test Loss {test_loss}"
    )
last_epoch_number += i + 1

# %% Plot #########################################################################
for data in test_loader:
    img = data[torch.randint(low=0, high=31, size=(1,)).item(), :, :, :]
    print(img.shape)
    output_img = gpu_model(img.permute(2, 0, 1).unsqueeze(0).to(device)).cpu()
    output_img = output_img.detach().squeeze(0).permute(1, 2, 0)
    figure = plt.figure()
    subplot1 = figure.add_subplot(1, 2, 1)
    subplot1.imshow(img)
    subplot1.set_title("Original Image")

    subplot2 = figure.add_subplot(1, 2, 2)
    subplot2.imshow(output_img)
    subplot2.set_title("Output Image")

    plt.show()

    break

# %% save model
import dill
torch.save(gpu_model, "vision_transformer_autoencoder.pth", pickle_module=dill)

# %% Attention gate output ########################################################

for data in test_loader:
    img = data[torch.randint(low=0, high=31, size=(1,)).item(), :, :, :]
    print(img.shape)

    attention_gate_output = (
        gpu_model.attention_gate_output(img.permute(2, 0, 1).unsqueeze(0).to(device))
        .detach()
        .squeeze(0)
        .cpu()
    )
    print(attention_gate_output.shape)
    out_img_mean = attention_gate_output.norm(dim=1)[1:].reshape(2, 2)

    print(out_img_mean.shape)
    figure = plt.figure()
    subplot1 = figure.add_subplot(1, 2, 1)
    subplot1.imshow(img)
    subplot1.set_title("Original Image")

    subplot2 = figure.add_subplot(1, 2, 2)
    subplot2.imshow(out_img_mean)
    subplot2.set_title("Attention Gate Output")

    plt.show()
    min_val = attention_gate_output[:-1].min()
    max_val = attention_gate_output[:-1].max()

    plt.show()

    break
