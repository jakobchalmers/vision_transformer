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


# FILE_NAME = "data/multiple_particle_dataset_500.pth"
# TEST_FILE_NAME = "data/multiple_particle_test_dataset_125.pth"

FILE_NAME = "data/particle_dataset_500.pth"
TEST_FILE_NAME = "data/particle_test_dataset_100.pth"

# FILE_NAME = "data/particle_dataset_4000.pth"
# TEST_FILE_NAME = "data/particle_test_dataset_1000.pth"

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

# %%

try:
    del sys.modules["modules"]
    from modules import PrintLayer, ConvolutionalDecoder, ConvolutionalEncoder
except KeyError:
    from modules import (
        PrintLayer,
        ConvolutionalDecoder,
        ConvolutionalEncoder,
        ConvolutionalAutoencoder,
    )


def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for input in train_loader:
        input = input.to(device)
        input: torch.Tensor = input.permute(0, 3, 1, 2)
        output = model(input)

        loss = criterion(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    mean_loss = total_loss / len(train_loader)
    return mean_loss


def test(model, data_loader, criterion):
    model.train()
    total_loss = 0
    for input in data_loader:
        input = input.to(device)
        input: torch.Tensor = input.permute(0, 3, 1, 2)
        output = model(input)

        loss = criterion(input, output)
        total_loss += loss.item()

    mean_loss = total_loss / len(data_loader)
    return mean_loss


# %% Training Setup
from tqdm import tqdm
import torch.optim as optim

model = ConvolutionalAutoencoder(
    hidden_feature_dim_1=16,
    hidden_feature_dim_2=32,
    hidden_feature_dim_3=64,
    latent_dim=4,
).to(device)
criterion = nn.MSELoss()

initial_train_loss = test(model, train_loader, criterion)
print(f"{initial_train_loss=}")
initial_test_loss = test(model, test_loader, criterion)
print(f"{initial_test_loss=}")

# %% Training
num_epochs = 10

optimizer = optim.Adam(model.parameters(), lr=0.001)
for i, epoch in enumerate(tqdm(range(num_epochs))):
    train_loss = train(model, train_loader, criterion, optimizer)
    test_loss = test(model, test_loader, criterion)

    print(f"Epoch {i+1}: Train loss {train_loss}, Test Loss {test_loss}")


# %% Plot

for test_batch in test_loader:
    img = test_batch[torch.randint(low=0, high=31, size=(1,)).item(), :, :, :]

    output_img = model.forward(img.permute(2, 0, 1).unsqueeze(0).to(device)).cpu()
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


# %% Save model
torch.save(model, "models/convolutional_autoencoder.pth")
# torch.save(model, "models/multiple_particle_convolutional_autoencoder.pth")


# %% Train varying latent dimensions

latent_dimensions = [1, 2, 3, 4, 8]
final_loss_thresholds = [20, 15, 10, 5, 5]

models = []
final_losses = []
for l, latent_dimension in tqdm(enumerate(latent_dimensions)):
    while True:
        model = ConvolutionalAutoencoder(
            hidden_feature_dim_1=16,
            hidden_feature_dim_2=32,
            hidden_feature_dim_3=64,
            latent_dim=latent_dimension,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        num_epochs = 20

        for i, epoch in enumerate(tqdm(range(num_epochs))):
            train_loss = train(model, train_loader, criterion, optimizer)
            print(f"Epoch {i+1}: Train loss {train_loss}")
        test_loss = test(model, test_loader, criterion)
        if test_loss < final_loss_thresholds[l]:
            break

    print(f"{test_loss=}")

    models.append(model)
    final_losses.append(test_loss)

# %% PLot varying latent dimensions
test_batch = next(iter(test_loader))

num_inputs = 5
input_images = test_batch[torch.randint(low=0, high=31, size=(num_inputs,)), :, :, :]


model_output_image_sets = []
for i, model in enumerate(models):
    out_images = model(input_images.permute(0, 3, 1, 2).to(device)).detach().cpu()
    model_output_image_sets.append(out_images)

figure = plt.figure()
figure.suptitle("Examples for varying latent dim")
plt.axis("off")
for i in range(num_inputs):
    # Plot input images
    subplot = figure.add_subplot(num_inputs, len(models) + 1, i * (len(models) + 1) + 1)
    subplot.imshow(input_images[i])
    subplot.axis("off")

    if i == 0:
        subplot.set_title("input")

    # Plot output images
    for j in range(len(models)):
        subplot = figure.add_subplot(
            num_inputs, len(models) + 1, i * (len(models) + 1) + j + 2
        )
        subplot.imshow(model_output_image_sets[j][i].permute(1, 2, 0))
        subplot.axis("off")

        if i == 0:
            subplot.set_title(f"{latent_dimensions[j]}")

plt.show()
