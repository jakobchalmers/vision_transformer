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
data_loader: dt.Sequence = (
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


data_size = 500
dataset: SequenceDataset = SequenceDataset(data_loader, data_size)

FILE_NAME = "particle_dataset.pth"
torch.save(dataset, FILE_NAME)

# %%

try:
    sequence_dataset = dataset
except:
    print("Loading...")
    sequence_dataset: SequenceDataset = torch.load(FILE_NAME)


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

data_loader = DataLoader(
    image_dataset,
    batch_size=32,
    shuffle=True,
)

# %%
import matplotlib.pyplot as plt
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(
        self, image_height, image_width, hidden_feature_dim, latent_feature_dim
    ):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_feature_dim, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                hidden_feature_dim,
                latent_feature_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_feature_dim,
                hidden_feature_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            # nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_feature_dim,
                1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for input in train_loader:
        input: torch.Tensor = input.permute(0, 3, 1, 2)
        output = model(input)

        loss = criterion(input, output)
        optimizer.zero_grad()
        loss.backward()

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


# %%
from tqdm import tqdm
import torch.optim as optim

model = Autoencoder(
    image_height=IMAGE_SIZE,
    image_width=IMAGE_SIZE,
    hidden_feature_dim=16,
    latent_feature_dim=8,
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 5
for i, epoch in tqdm(enumerate(range(num_epochs))):
    train_loss = train(model, data_loader, criterion, optimizer)
    test_loss = test(model, data_loader, criterion)

    print(f"Epoch {i}: Train loss {train_loss}, Test Loss {test_loss}")
