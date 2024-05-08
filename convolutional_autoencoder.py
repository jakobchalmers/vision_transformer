# %%
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from data_generation import SequenceDataset, ImageDataset


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
        hidden_feature_dim_1,
        hidden_feature_dim_2,
        hidden_feature_dim_3,
        latent_dim,
    ):
        super(Autoencoder, self).__init__()

        kernel_size = 3
        # activation = nn.LeakyReLU()
        activation = nn.SiLU()
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

            nn.AvgPool2d(kernel_size=8),
            nn.Flatten(start_dim=1),
            PrintLayer(identifier="after flatten"),

            nn.Linear(hidden_feature_dim_3, latent_dim),
            activation,
            PrintLayer(identifier="latent"),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * hidden_feature_dim_3),
            # nn.ConvTranspose1d(latent_dim, hidden_feature_dim_3, kernel_size=2),
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
            # activation,

            # nn.Conv2d(
            #    in_channels=4,
            #    out_channels=1,
            #    kernel_size=3,
            #    stride=1,
            #    padding=1,
            # ),
            # nn.Sigmoid(),
            
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

# %%

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

# %%


from tqdm import tqdm
import torch.optim as optim

model = Autoencoder(
    hidden_feature_dim_1=16,
    hidden_feature_dim_2=32,
    hidden_feature_dim_3=64,
    latent_dim=3,
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

    latent_space, output_img = model.forward_testing(img.permute(2, 0, 1).unsqueeze(0))
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

# %%
rec_iter = 1000
for data in test_loader:
    img = data[torch.randint(low=0, high=31, size=(1,)).item(), :, :, :]

    input = img.permute(2, 0, 1).unsqueeze(0)

    for i in range(rec_iter):
        latent_space, output_img = model.forward_testing(input)
        output_img = output_img.detach().squeeze(0).permute(1, 2, 0)
        input = output_img.permute(2, 0, 1).unsqueeze(0)
    
    figure = plt.figure()
    subplot1 = figure.add_subplot(1, 2, 1)
    subplot1.imshow(img)
    subplot1.set_title('Original Image')

    subplot2 = figure.add_subplot(1, 2, 2)
    subplot2.imshow(output_img)
    subplot2.set_title(f'Output Image after {i+1} iterations')

    plt.show()
    break
        
