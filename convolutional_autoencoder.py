# %%
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from data_generation import SequenceDataset, ImageDataset
import sys

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
print("Done")

# %%

try:
    del sys.modules['modules']
    from modules import PrintLayer, ConvolutionalDecoder, ConvolutionalEncoder
except KeyError:
    from modules import PrintLayer, ConvolutionalDecoder, ConvolutionalEncoder


class ConvolutionalAutoencoder(nn.Module):
    def __init__(
        self,
        hidden_feature_dim_1,
        hidden_feature_dim_2,
        hidden_feature_dim_3,
        latent_dim,
    ):
        super(ConvolutionalAutoencoder, self).__init__()

        kernel_size = 3
        activation = nn.SiLU()
        # dim: batch x 1 x 64 x 64

        self.encoder = ConvolutionalEncoder(
            latent_dim=latent_dim,
            hidden_feature_dim_1=hidden_feature_dim_1,
            hidden_feature_dim_2=hidden_feature_dim_2,
            hidden_feature_dim_3=hidden_feature_dim_3,
            activation=activation,
            kernel_size=kernel_size,
        )
        self.decoder = ConvolutionalDecoder(
            latent_dim=latent_dim,
            hidden_feature_dim_1=hidden_feature_dim_1,
            hidden_feature_dim_2=hidden_feature_dim_2,
            hidden_feature_dim_3=hidden_feature_dim_3,
            activation=activation,
            kernel_size=kernel_size,
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


# %% Training Setup
from tqdm import tqdm
import torch.optim as optim

model = ConvolutionalAutoencoder(
    hidden_feature_dim_1=16,
    hidden_feature_dim_2=32,
    hidden_feature_dim_3=64,
    latent_dim=4,
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

initial_train_loss = test(model, train_loader, criterion)
print(f"{initial_train_loss=}")
initial_test_loss = test(model, test_loader, criterion)
print(f"{initial_test_loss=}")

# %% Training
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
        
