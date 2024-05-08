# %%
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from data_generation import SequenceDataset, ImageDataset
import numpy as np

# %% Data Loading

# FILE_NAME = "data/particle_dataset_500.pth"
# TEST_FILE_NAME = "data/particle_test_dataset_100.pth"

FILE_NAME = "data/particle_dataset_4000.pth"
TEST_FILE_NAME = "data/particle_test_dataset_1000.pth"

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

# %% Functions
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

# %% Modules

class Patchify(nn.Module):
    def __init__(self, patch_size: int):
        super(Patchify, self).__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = self.unfold(x)  # -> (batch_size, channels*batch_size**2, num_patches)

        # x = x.view(batch_size, -1,channels*self.patch_size**2) # -> (batch_size, num_patches, patch_size**2)
        x = x.permute(0, 2, 1)
        return x


####
# testing
# img = torch.rand(1,1,8,8)
# patcher = Patchify(2)
# patches = patcher(img)
# print(patches.shape)

# fig, ax = plt.subplots(1, 2)

# ax[0].imshow(img[0,0].detach().numpy(), vmin=0, vmax=1)
# ax[1].imshow(patches[0,0,0].detach().numpy(), vmin=0, vmax=1)
# plt.show()
####


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


class ClassToken(nn.Module):
    def __init__(self, dim_embedding: int):
        super(ClassToken, self).__init__()
        self.dim_embedding = dim_embedding
        self.class_token = nn.Parameter(torch.randn(1, 1, dim_embedding))

    def forward(self, x):
        batch_size, num_patches, patch_dim = x.shape
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim_embedding: int, num_patches: int):
        super(PositionalEncoding, self).__init__()
        self.dim_embedding = dim_embedding
        self.num_patches = num_patches
        # dropout ??
        positional_encoding = torch.zeros(
            num_patches, dim_embedding
        )  # -> (num_patches, dim_embedding)
        position = torch.arange(0, num_patches).unsqueeze(1)  # -> (num_patches, 1)
        indices = torch.arange(0, dim_embedding, 2)  # -> (dim_embedding//2)
        factor = torch.exp(
            indices * -(np.log(10000.0) / dim_embedding)
        )  # -> (dim_embedding//2)
        positional_encoding[:, 0::2] = torch.sin(
            position * factor
        )  # -> (num_patches, dim_embedding//2)
        positional_encoding[:, 1::2] = (
            torch.cos(position * factor)
            if dim_embedding % 2 == 0
            else torch.cos(position * factor)[:, :-1]
        )
        self.positional_encoding = positional_encoding.unsqueeze(
            0
        )  # -> (1, num_patches, dim_embedding)

    def forward(self, x):
        return x + self.positional_encoding

class ClassTokenGrabber(nn.Module):
    def __init__(self, num_patches: int, dim_embedding: int):
        super(ClassTokenGrabber, self).__init__()
        self.num_patches = num_patches
        self.dim_embedding = dim_embedding

    def forward(self, x):
        return x[:, -1, :]  # -> (batch_size, dim_embedding)


class VisionTransformerAutoencoder(nn.Module):
    def __init__(self, dim_embedding: int):
        super(VisionTransformerAutoencoder, self).__init__()
        patch_size = 32
        # patch_size = 16
        image_size = 64
        num_patches = (image_size // patch_size) ** 2
        linear_input_dim = patch_size * patch_size * 1

        self.embedding = nn.Sequential(
            PrintLayer(identifier="input"),
            Patchify(patch_size=patch_size),
            PrintLayer(identifier="patched"),
            nn.Linear(linear_input_dim, dim_embedding),
            PrintLayer(identifier="linear"),
            ClassToken(dim_embedding=dim_embedding),
            PrintLayer(identifier="stacked"),
            PositionalEncoding(dim_embedding=dim_embedding, num_patches=num_patches+1),
            PrintLayer(identifier="positional"),
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_embedding,
                nhead=2,
                # nhead=1,
                dim_feedforward=2048,
                dropout=0.1,
                activation="gelu",
                norm_first=True,
            ),
            num_layers=4,
        )


        self.class_token_grabber = ClassTokenGrabber(num_patches=num_patches, dim_embedding=dim_embedding)

        activation = nn.SiLU()
        latent_dim = dim_embedding
        hidden_feature_dim_1 = 16
        hidden_feature_dim_2 = 32
        hidden_feature_dim_3 = 64
        kernel_size = 3
        self.decoder = nn.Sequential(
            PrintLayer(identifier="latent"),
            nn.Linear(latent_dim, 8 * 8 * hidden_feature_dim_3),
            activation,
            nn.Unflatten(dim=1, unflattened_size=(hidden_feature_dim_3, 8, 8)),
            

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
            
            PrintLayer(identifier="output"),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.class_token_grabber(x)
        x = self.decoder(x)
        return x

# %% Plot untrained reconstruction ###############################################

model = VisionTransformerAutoencoder(dim_embedding=4)

for batch in train_loader:
    input = batch.permute(0, 3, 1, 2)
    output = model(input)
    print(output.shape)
    plt.imshow(output[0, 0].detach().numpy())
    break

# %% Setup for training ###########################################################

model = VisionTransformerAutoencoder(dim_embedding=4)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# print(model)

initial_train_loss = test(model, train_loader, criterion)
print(f"{initial_train_loss=}")
initial_test_loss = test(model, test_loader, criterion)
print(f"{initial_test_loss=}")

last_epoch_number = 0

# %% Train #########################################################################
num_epochs = 20
for i, epoch in tqdm(enumerate(range(num_epochs))):
    train_loss = train(model, train_loader, criterion, optimizer)
    test_loss = test(model, test_loader, criterion)
    
    print(f"Epoch {i+1+last_epoch_number}: Train loss {train_loss}, Test Loss {test_loss}")
last_epoch_number += i+1

# %% Plot #########################################################################
for data in test_loader:
    img = data[torch.randint(low=0, high=31, size=(1,)).item(), :, :, :]
    print(img.shape)
    output_img = model.forward(img.permute(2, 0, 1).unsqueeze(0))
    output_img = output_img.detach().squeeze(0).permute(1, 2, 0)
    figure = plt.figure()
    subplot1 = figure.add_subplot(1, 2, 1)
    subplot1.imshow(img)
    subplot1.set_title('Original Image')
    
    subplot2 = figure.add_subplot(1, 2, 2)
    subplot2.imshow(output_img)
    subplot2.set_title('Output Image')
    
    plt.show()

    break