# %%

import deeptrack as dt
import numpy as np

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
data_loader = sequential_images >> dt.FlipUD() >> dt.FlipDiagonal() >> dt.FlipLR()

# dataset.plot()


# %% Generate batch
import torch
import torch.nn as nn

batch_size = 3


# batch = data_loader.batch(batch_size=batch_size)
# batch = torch.tensor(
#     batch
# )  # (num_frames, batch_size, image_width, image_height, color=1)
# batch = torch.tensor(batch, dtype=torch.float).swapaxes(
#     1, 0
# )  # (batch_size, num_frames, image_width, image_height, color=1)

# %%
import matplotlib.pyplot as plt

# print(batch.shape)

# plt.imshow(batch[0, 7, :, :, 0])


class Autoencoder(nn.Module):
    def __init__(
        self, image_height, image_width, hidden_feature_dim, latent_feature_dim
    ):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                hidden_feature_dim,
                latent_feature_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_feature_dim, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(model, num_batches, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    batch_count = 0
    for input in train_loader:
        output = model(input)

        loss = criterion(input, output)
        optimizer.zero_grad()
        loss.backward()

        total_loss += loss.item()

        if batch_count >= num_batches:
            break
        batch_count += 1

    mean_loss = total_loss / num_batches
    return mean_loss

def test(model, num_batches, data_loader, criterion):
    model.train()
    total_loss = 0
    batch_count = 0
    for input in data_loader:
        output = model(input)

        loss = criterion(input, output)
        total_loss += loss.item()

        if batch_count >= num_batches:
            break
        batch_count += 1

    mean_loss = total_loss / num_batches
    return mean_loss


# %%
from tqdm import tqdm
import torch.optim as optim

model = Autoencoder(image_height=IMAGE_SIZE, image_width=IMAGE_SIZE, hidden_feature_dim=16, latent_feature_dim=8)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 4
num_train_batches = 100
num_test_batches = num_train_batches // 2
for i, epoch in tqdm(enumerate(range(num_epochs))):
    train_loss = train(model, num_train_batches, data_loader, criterion, optimizer)
    test_loss = test(model, num_test_batches, data_loader= criterion)

    print(f"Epoch {i}: Train loss {train_loss}, Test Loss {test_loss}")



