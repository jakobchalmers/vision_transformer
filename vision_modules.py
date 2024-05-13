import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision
from torchvision.models import VisionTransformer
from tqdm import tqdm
from data_generation import SequenceDataset, ImageDataset
import numpy as np
import sys
from modules import PrintLayer, ConvolutionalDecoder

class Patchify(nn.Module):
    def __init__(self, patch_size: int):
        super(Patchify, self).__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.unfold(x)  # -> (batch_size, channels*batch_size**2, num_patches)
        x = x.permute(0, 2, 1)
        return x


class ClassToken(nn.Module):
    def __init__(self, dim_embedding: int):
        super(ClassToken, self).__init__()
        self.dim_embedding = dim_embedding
        initial_class_token = torch.randn(1, 1, dim_embedding, requires_grad=True)
        self.class_token = nn.Parameter(initial_class_token, requires_grad=True)
        print(self.class_token)

    def forward(self, x):
        batch_size, num_patches, patch_dim = x.shape
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        assert torch.all(class_token == x[:, 0, :])
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim_embedding: int, num_patches: int, device):
        super(PositionalEncoding, self).__init__()
        self.dim_embedding = dim_embedding
        self.num_patches = num_patches
        # dropout ??
        positional_encoding = torch.zeros(num_patches, dim_embedding).to(
            device
        )  # -> (num_patches, dim_embedding)
        position = (
            torch.arange(0, num_patches).unsqueeze(1).to(device)
        )  # -> (num_patches, 1)
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
        # Get class token (first element)
        return x[:, -1, :]  # -> (batch_size, dim_embedding)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation=nn.GELU(),
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.activation = activation

        self.norm1 = nn.LayerNorm(dim_embedding)
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
        x_1 = x.clone()
        x_2 = self.norm1(x)
        x_2 = self.multihead_attention(x_2, x_2, x_2)[
            0
        ]  # 0 is the output, 1 is the attention weights -> (batch_size, num_patches, dim_embedding)

        x_3 = x_1 + x_2
        x_4 = self.norm2(x_3)
        x_4 = self.mlp(x_4)
        x = x_3 + x_4
        return x

    def attention_gate_output(self, x):
        x_1 = x
        x_2 = self.norm1(x)

        x_2 = self.multihead_attention(x_2, x_2, x_2)[
            0
        ]  # 0 is the output, 1 is the attention weights -> (batch_size, num_patches, dim_embedding)

        return x_2


class VisionTransformerAutoencoder(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        latent_dim: int,
        patch_size: int,
        number_of_heads,
        number_of_transformer_layers,
        device,
    ):
        super(VisionTransformerAutoencoder, self).__init__()
        self.device = device
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
        #         nhead=number_of_heads,
        #         dim_feedforward=256, # 2048
        #         dropout=0.1,
        #         activation="gelu",
        #         norm_first=True,
        #     ),
        #     num_layers=2,
        # )

        
        self.transformer_encoder_layers = nn.Sequential(
            *(
                TransformerEncoderLayer(
                    dim_embedding=dim_embedding,
                    nhead=number_of_heads,
                    dim_feedforward=2048,
                    dropout=0.1,
                    activation=nn.GELU(),
                )
                for _ in range(number_of_transformer_layers)
            )
        )

        self.class_token_grabber = ClassTokenGrabber(
            num_patches=num_patches, dim_embedding=dim_embedding
        )
        # self.linear_embedding2latent = nn.Linear(dim_embedding, latent_dim)
        
        self.encoder = nn.Sequential(
            PrintLayer(identifier="input"),
            self.embedding,
            PrintLayer(identifier="embedding"),
            self.transformer_encoder_layers,
            PrintLayer(identifier="transformer"),
            self.class_token_grabber,
            PrintLayer(identifier="latent"),
        )

        self.decoder = ConvolutionalDecoder(
            latent_image_size=8,
            latent_dim=latent_dim,
            hidden_feature_dim_1=16,
            hidden_feature_dim_2=32,
            hidden_feature_dim_3=64,
            activation=nn.SiLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def attention_gate_output(self, x):
        x = self.embedding(x)
        number_of_layers = len(self.transformer_encoder_layers)
        for i, layer in enumerate(self.transformer_encoder_layers):
            if i == number_of_layers - 1:
                x = layer.attention_gate_output(x)
                break
            x = layer(x)
        return x
    


