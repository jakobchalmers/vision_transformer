import torch.nn as nn
import torch


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


class ConvolutionalEncoder(nn.Module):
    def __init__(
        self,
        latent_dim=4,
        hidden_feature_dim_1=16,
        hidden_feature_dim_2=32,
        hidden_feature_dim_3=64,
        activation=nn.ReLU(),
        kernel_size=3,
    ):
        super(ConvolutionalEncoder, self).__init__()
        latent_image_size = 16

        self.encoder = nn.Sequential(
            nn.Conv2d(
                1,
                hidden_feature_dim_1,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            ),
            activation,

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                hidden_feature_dim_1,
                hidden_feature_dim_2,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            ),
            activation,

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                hidden_feature_dim_2,
                hidden_feature_dim_3,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            ),
            activation,

            nn.AvgPool2d(kernel_size=latent_image_size, stride=1),
            nn.Flatten(),
            nn.Linear(hidden_feature_dim_3, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class ConvolutionalDecoder(nn.Module):
    def __init__(
        self,
        latent_image_size=16,
        latent_dim=4,
        hidden_feature_dim_1=16,
        hidden_feature_dim_2=32,
        hidden_feature_dim_3=64,
        activation=nn.ReLU(),
        kernel_size=3,
    ):
        super(ConvolutionalDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, (latent_image_size**2) * hidden_feature_dim_3),
            activation,
            nn.Unflatten(
                dim=1,
                unflattened_size=(
                    hidden_feature_dim_3,
                    latent_image_size,
                    latent_image_size,
                ),
            ),

            nn.Conv2d(
                hidden_feature_dim_3,
                hidden_feature_dim_3,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            ),
            activation,

            nn.ConvTranspose2d(
                hidden_feature_dim_3,
                hidden_feature_dim_2,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                output_padding=1,
            ),

            nn.Conv2d(
                hidden_feature_dim_2,
                hidden_feature_dim_2,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
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
            
            nn.Conv2d(
                hidden_feature_dim_1,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        return self.decoder(x)
