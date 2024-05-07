# %%

import deeptrack as dt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

IMAGE_SIZE = 64
sequence_length = 10  # Number of frames per sequence
MIN_SIZE = 0.5e-6
MAX_SIZE = 1.5e-6
MAX_VEL = 10  # Maximum velocity. The higher the trickier!
MAX_PARTICLES = 3  # Max number of particles in each sequence. The higher the trickier!

# Defining properties of the particles
particle = dt.Sphere(
    intensity=lambda: 10 + 10 * np.random.rand(),
    radius=lambda: MIN_SIZE + np.random.rand() * (MAX_SIZE - MIN_SIZE),
    position=lambda: IMAGE_SIZE * np.random.rand(2),
    vel=lambda: MAX_VEL * np.random.rand(2),
    position_unit="pixel",
)


# Defining an update rule for the particle position
def get_position(previous_value, vel):

    newv = previous_value + vel
    for i in range(2):
        if newv[i] > IMAGE_SIZE - 1:
            newv[i] = IMAGE_SIZE - np.abs(newv[i] - IMAGE_SIZE)
            vel[i] = -vel[i]
        elif newv[i] < 0:
            newv[i] = np.abs(newv[i])
            vel[i] = -vel[i]
    return newv


particle = dt.Sequential(particle, position=get_position)

# Defining properties of the microscope
optics = dt.Fluorescence(
    NA=1,
    output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
    magnification=10,
    resolution=(1e-6, 1e-6, 1e-6),
    wavelength=633e-9,
)


# Combining everything into a dataset.
# Note that the sequences are flipped in different directions, so that each unique sequence defines
# in fact 8 sequences flipped in different directions, to speed up data generation
sequential_images = dt.Sequence(
    optics(particle ** (lambda: 1 + np.random.randint(MAX_PARTICLES))),
    sequence_length=sequence_length,
)
train_loader: dt.Sequence = (
    sequential_images >> dt.FlipUD() >> dt.FlipDiagonal() >> dt.FlipLR()
)


class SequenceDataset(Dataset):
    def __init__(self, data_generator, data_size):
        data = []
        for _ in tqdm(range(data_size)):
            sequence = data_generator.update().resolve()

            data.append(sequence)
        self.sequences = torch.tensor(data)

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        sequence = self.sequences[idx].float()
        return sequence


FILE_NAME = "particle_dataset.pth"
TEST_FILE_NAME = "particle_test_dataset.pth"
GENERATE = False
if GENERATE:
    data_size = 500
    dataset: SequenceDataset = SequenceDataset(train_loader, data_size)
    torch.save(dataset, FILE_NAME)

    test_data_size = 100
    test_dataset: SequenceDataset = SequenceDataset(train_loader, test_data_size)
    torch.save(test_dataset, TEST_FILE_NAME)

# %%

try:
    sequence_dataset: SequenceDataset = dataset
    sequence_test_dataset: SequenceDataset = test_dataset
except:
    print("Loading...")
    sequence_dataset: SequenceDataset = torch.load(FILE_NAME)
    sequence_test_dataset: SequenceDataset = torch.load(TEST_FILE_NAME)


class ImageDataset(Dataset):
    def __init__(self, sequence_dataset):
        sequences = sequence_dataset.sequences
        self.images: torch.Tensor = sequences.view(-1, *sequences.shape[2:])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = self.images[index].float()
        return image


image_dataset = ImageDataset(sequence_dataset)
image_test_dataset = ImageDataset(sequence_test_dataset)

image_dataset.images /= image_dataset.images.max()
image_test_dataset.images /= image_test_dataset.images.max()

# image_dataset.images /= 255
# image_test_dataset.images /= 255


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

for data in test_loader:
    flat = data.view(data.shape[0], -1)
    sorted = torch.argsort(flat, dim=1, descending=True)
    print(flat[:, sorted[:10]])

# %%
import matplotlib.pyplot as plt
import torch.nn as nn

class PrintLayer(nn.Module):
    def __init__(self, identifier: str):
        super(PrintLayer, self).__init__()
        self.has_printed = False
        self.identifier = identifier

    def forward(self, x):
        if not self.has_printed:
            print(f"PrintLayer\n\t{self.identifier}::Shape:", x.shape)
            self.has_printed = True
        return x


class Autoencoder(nn.Module):
    def __init__(
        self,
        image_height,
        image_width,
        hidden_feature_dim_1,
        hidden_feature_dim_2,
        hidden_feature_dim_3,
        latent_dim,
    ):
        super(Autoencoder, self).__init__()

        kernel_size = 3
        activation = nn.LeakyReLU()
        # dim: batch x 1 x 64 x 64
        self.encoder = nn.Sequential(
            PrintLayer(identifier="input"),
            nn.Conv2d(
                1, hidden_feature_dim_1, kernel_size=kernel_size, stride=1, padding=1
            ),
            activation,
            # dim: 64x64
            nn.MaxPool2d(kernel_size=2, stride=2),
            # dim: 32x32


            nn.Conv2d(
                hidden_feature_dim_1,
                hidden_feature_dim_2,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            ),
            activation,
            # dim: 32x32
            nn.MaxPool2d(kernel_size=2, stride=2),
            # dim: 16x16

            nn.Conv2d(
                hidden_feature_dim_2,
                hidden_feature_dim_3,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            ),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            # dim: 8x8

            # nn.Flatten(),
            # nn.Linear(
            #     8 * 8 * hidden_feature_dim_3, latent_dim
            # ),  # TODO: don't hardcode

            PrintLayer(identifier="latent")
        )

        self.decoder = nn.Sequential(
            # nn.Linear(latent_dim, 8 * 8 * hidden_feature_dim_3),
            # nn.Unflatten(dim=1, unflattened_size=(hidden_feature_dim_3, 8, 8)),

            nn.ConvTranspose2d(
                hidden_feature_dim_3,
                hidden_feature_dim_2,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            activation,

            nn.ConvTranspose2d(
                hidden_feature_dim_2,
                hidden_feature_dim_1,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            activation,

            nn.ConvTranspose2d(
                hidden_feature_dim_1,
                1,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),
            PrintLayer(identifier="output"),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def forward_testing(self, x):

        x = self.encoder(x)
        latent_space = x.clone()

        x = self.decoder(x)
        return latent_space, x


def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for input in train_loader:
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
        input: torch.Tensor = input.permute(0, 3, 1, 2)
        output = model(input)

        loss = criterion(input, output)
        total_loss += loss.item()

    mean_loss = total_loss / len(data_loader)
    return mean_loss


from tqdm import tqdm
import torch.optim as optim

model = Autoencoder(
    image_height=IMAGE_SIZE,
    image_width=IMAGE_SIZE,
    hidden_feature_dim_1=4,
    hidden_feature_dim_2=4,
    hidden_feature_dim_3=4,
    latent_dim=int(8 * 8 * 8 / 2),
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

initial_train_loss = test(model, train_loader, criterion)
print(f"{initial_train_loss=}")
initial_test_loss = test(model, test_loader, criterion)
print(f"{initial_test_loss=}")

num_epochs = 20
for i, epoch in tqdm(enumerate(range(num_epochs))):
    train_loss = train(model, train_loader, criterion, optimizer)
    test_loss = test(model, test_loader, criterion)

    print(f"Epoch {i+1}: Train loss {train_loss}, Test Loss {test_loss}")

# %% plot

for data in test_loader:
    img = data[torch.randint(low=0, high=31, size=(1,)).item(), :, :, :]
    print(img.shape)
    plt.imshow(img)
    plt.show()

    latent_space, output_img = model.forward_testing(img.permute(2, 0, 1).unsqueeze(0))
    # latent_img = latent_space.detach().permute(1, 2, 0)
    # latent_img_normed = (latent_img - latent_img.min()) / (
    #     latent_img.max() - latent_img.min()
    # )
    # plt.imshow(latent_img_normed[:, :, 0])
    # plt.imshow(latent_img_normed[:, :, 1])
    # plt.imshow(latent_img_normed[:, :, 2])

    output_img = output_img.detach().squeeze(0).permute(1, 2, 0)
    # loss = criterion(img, output_img)
    # print(loss.item())
    print(output_img.shape)
    plt.imshow(output_img)

    # print(img.max())
    # print(output_img.max())
    # print(img - output_img)

    # plt.imshow(img - output_img)

    break
