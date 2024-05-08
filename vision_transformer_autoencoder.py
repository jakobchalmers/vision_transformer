# %%
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from data_generation import SequenceDataset, ImageDataset
import numpy as np

# %% Modules
# class Patcher(nn.Module):
#     def __init__(self, patch_size: int, image_size: int, in_channels: int):
#         super(Patcher, self).__init__()
#         self.patch_size = patch_size
#         self.image_size = image_size
#         self.in_channels = in_channels

#         self.patch_dim = patch_size * patch_size * in_channels
#         self.num_patches = (image_size // patch_size) ** 2

#         self.conv = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=self.patch_dim,
#             kernel_size=patch_size,
#             stride=patch_size,
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
#         x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, self.patch_dim, self.patch_size, self.patch_size)
#         return x

# class Patcher(nn.Module):
#     def __init__(self):
#         super(Patcher, self).__init__()

#     def forward(self, x):
#         x = x.unfold(2, 16, 16).unfold(3, 16, 16)
#         x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, 1, 16, 16)
#         return x


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
        print(class_token.shape)
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
        indicies = torch.arange(0, dim_embedding, 2)  # -> (dim_embedding//2)
        factor = torch.exp(
            indicies * -(np.log(10000.0) / dim_embedding)
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
        return x+ self.positional_encoding


class VisionTransformerAutoencoder(nn.Module):
    def __init__(self, dim_embedding: int):
        super(VisionTransformerAutoencoder, self).__init__()
        patch_size = 32
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

        # self.encoder = nn.Sequential(
        # )

        # self.decoder = nn.Sequential(
        # )

    def forward(self, x):
        x = self.embedding(x)
        # x = self.encoder(x)
        # x = self.decoder(x)
        return x


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

# %% test on data

model = VisionTransformerAutoencoder(dim_embedding=3)

for batch in train_loader:
    input = batch.permute(0, 3, 1, 2)
    output = model(input)
    print(output.shape)

    break
