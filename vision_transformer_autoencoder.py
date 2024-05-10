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


class Patchify(nn.Module):
    def __init__(self, patch_size: int):
        super(Patchify, self).__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.unfold(x)  # -> (batch_size, channels*batch_size**2, num_patches)

        # x = x.view(batch_size, -1,channels*self.patch_size**2) # -> (batch_size, num_patches, patch_size**2)
        x = x.permute(0, 2, 1)
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
    def __init__(self, dim_embedding: int, num_patches: int, device):
        super(PositionalEncoding, self).__init__()
        self.dim_embedding = dim_embedding
        self.num_patches = num_patches
        # dropout ??
        positional_encoding = torch.zeros(
            num_patches, dim_embedding
        ).to(device)  # -> (num_patches, dim_embedding)
        position = torch.arange(0, num_patches).unsqueeze(1).to(device)  # -> (num_patches, 1)
        indices = torch.arange(0, dim_embedding, 2).to(device)  # -> (dim_embedding//2)
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


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation = nn.GELU(),
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.activation = activation

        self.norm1 = nn.LayerNorm(dim_embedding)
        # self.linear_query = nn.Linear(dim_embedding, dim_embedding)
        # self.linear_key = nn.Linear(dim_embedding, dim_embedding)
        # self.linear_value = nn.Linear(dim_embedding, dim_embedding)
        self.multihead_attention = nn.MultiheadAttention(
            dim_embedding, nhead, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim_embedding)
        self.mlp = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            self.activation,
            nn.Linear(dim_feedforward, dim_embedding),
        )

    def forward(self, x):
        x_1 = x
        x_2 = self.norm1(x)
        x_2 = self.multihead_attention(x_2, x_2, x_2)[
            0
        ]  # 0 is the output, 1 is the attention weights -> (batch_size, num_patches, dim_embedding)

        # x_2 = self.multihead_attention(
        #     self.linear_query(x_2), self.linear_key(x_2), self.linear_value(x_2)
        # )[0] # 0 is the output, 1 is the attention weights -> (batch_size, num_patches, dim_embedding)

        x_3 = x_1 + x_2
        x_4 = self.norm2(x_3)
        x_4 = self.mlp(x_4)
        x = x_3 + x_4
        return x

    def attention_gate_output(self, x):
        x_1 = x
        x_2 = self.norm1(x)

        # x_2 = self.multihead_attention(
        #     self.linear_query(x_2), self.linear_key(x_2), self.linear_value(x_2)
        # )[
        #     0
        # ]  # 0 is the output, 1 is the attention weights -> (batch_size, num_patches, dim_embedding)

        x_2 = self.multihead_attention(x_2, x_2, x_2)[
            0
        ]  # 0 is the output, 1 is the attention weights -> (batch_size, num_patches, dim_embedding)

        return x_2


class VisionTransformerAutoencoder(nn.Module):
    def __init__(self, dim_embedding: int, device):
        super(VisionTransformerAutoencoder, self).__init__()
        self.device = device
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
            PositionalEncoding(
                dim_embedding=dim_embedding, num_patches=num_patches + 1, device=device
            ),
            PrintLayer(identifier="positional"),
        )

        # self.encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=dim_embedding,
        #         nhead=2,
        #         # nhead=1,
        #         dim_feedforward=2048,
        #         dropout=0.1,
        #         activation="gelu",
        #         norm_first=True,
        #     ),
        #     num_layers=8,
        # )

        num_transformer_encoder_layers = 4
        self.encoder = nn.Sequential(
            *(
                TransformerEncoderLayer(
                    dim_embedding=dim_embedding,
                    nhead=2,
                    dim_feedforward=2048,
                    dropout=0.1,
                    activation=nn.GELU(),
                )
                for _ in range(num_transformer_encoder_layers)
            )
        )

        self.class_token_grabber = ClassTokenGrabber(
            num_patches=num_patches, dim_embedding=dim_embedding
        )

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

        # self.decoder = ConvolutionalDecoder(
        #     latent_image_size=16,
        #     latent_dim=dim_embedding,
        #     hidden_feature_dim_1=16,
        #     hidden_feature_dim_2=32,
        #     hidden_feature_dim_3=64,
        # )

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.class_token_grabber(x)
        x = self.decoder(x)
        return x

    def attention_gate_output(self, x):
        x = self.embedding(x)
        number_of_layers = len(self.encoder)
        for i, layer in enumerate(self.encoder):
            if i == number_of_layers - 1:
                x = layer.attention_gate_output(x)
                break
            x = layer(x)
        return x


# %% Setup for training ###########################################################

gpu_model = VisionTransformerAutoencoder(dim_embedding=4, device=device).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(gpu_model.parameters(), lr=0.01)

initial_train_loss = test(gpu_model, train_loader, criterion, device)
print(f"{initial_train_loss=}")
initial_test_loss = test(gpu_model, test_loader, criterion, device)
print(f"{initial_test_loss=}")

last_epoch_number = 0

# %% Train #########################################################################
num_epochs = 30
for i, epoch in tqdm(enumerate(range(num_epochs))):
    train_loss = train(gpu_model, train_loader, criterion, optimizer, device)
    test_loss = test(gpu_model, test_loader, criterion, device)

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

# %% Attention gate output ########################################################

for data in test_loader:
    img = data[torch.randint(low=0, high=31, size=(1,)).item(), :, :, :]
    print(img.shape)
    
    attention_gate_output = gpu_model.attention_gate_output(img.permute(2, 0, 1).unsqueeze(0).to(device)).detach().squeeze(0).cpu()
    print(attention_gate_output.shape)
    out_img_mean = attention_gate_output.mean(dim=1)[:-1].reshape(2, 2)
    out_img_ch1 = attention_gate_output[:, 0][:-1].reshape(2, 2)
    out_img_ch2 = attention_gate_output[:, 1][:-1].reshape(2, 2)
    out_img_ch3 = attention_gate_output[:, 2][:-1].reshape(2, 2)
    out_img_ch4 = attention_gate_output[:, 3][:-1].reshape(2, 2)

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

    figure = plt.figure()
    subplot1 = figure.add_subplot(2, 2, 1)
    subplot1.imshow(out_img_ch1, vmin=min_val, vmax=max_val)
    subplot1.set_title("Channel 1")

    subplot2 = figure.add_subplot(2, 2, 2)
    subplot2.imshow(out_img_ch2, vmin=min_val, vmax=max_val)
    subplot2.set_title("Channel 2")

    subplot3 = figure.add_subplot(2, 2, 3)
    subplot3.imshow(out_img_ch3, vmin=min_val, vmax=max_val)
    subplot3.set_title("Channel 3")

    subplot4 = figure.add_subplot(2, 2, 4)
    subplot4.imshow(out_img_ch4, vmin=min_val, vmax=max_val)
    subplot4.set_title("Channel 4")
    figure.tight_layout()

    plt.show()

    break
