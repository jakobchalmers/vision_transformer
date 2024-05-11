# %%
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from data_generation import SequenceDataset, ImageDataset
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device=}")

# %%

FILE_NAME = "data/particle_dataset_500.pth"
TEST_FILE_NAME = "data/particle_test_dataset_100.pth"

print("Loading...")
sequence_dataset: SequenceDataset = torch.load(FILE_NAME)
sequence_test_dataset: SequenceDataset = torch.load(TEST_FILE_NAME)

sequence_train_loader = DataLoader(
    sequence_dataset,
    batch_size=32,
    shuffle=True,
)

sequence_test_loader = DataLoader(
    sequence_test_dataset,
    batch_size=32,
    shuffle=False,
)


image_dataset = ImageDataset(sequence_dataset)
image_test_dataset = ImageDataset(sequence_test_dataset)

image_train_loader = DataLoader(
    image_dataset,
    batch_size=32,
    shuffle=True,
)

image_test_loader = DataLoader(
    image_test_dataset,
    batch_size=32,
    shuffle=False,
)
print("Done")

# %%

try:
    del sys.modules["modules"]
    from modules import PrintLayer, ConvolutionalDecoder, ConvolutionalEncoder, ConvolutionalAutoencoder
except KeyError:
    from modules import PrintLayer, ConvolutionalDecoder, ConvolutionalEncoder, ConvolutionalAutoencoder

class Time2Vector(nn.Module):
    def __init__(self, seq_len, latent_dim):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        d = 2

        self.linear_weights = nn.Parameter(torch.rand(seq_len)) # requires grad by default?
        self.linear_biases = nn.Parameter(torch.rand(seq_len))
        self.periodic_weights = nn.Parameter(torch.rand(seq_len))
        self.periodic_biases = nn.Parameter(torch.rand(seq_len))

    def forward(self, x):
        x_embedding = x.clone()
        x_embedding = x_embedding.mean(dim=-1)

        time_linear = (self.linear_weights * x_embedding + self.linear_biases).unsqueeze(-1)
        time_periodic = torch.sin(self.periodic_weights * x_embedding + self.periodic_biases).unsqueeze(-1)

        x = torch.cat([x, time_linear, time_periodic], dim=-1)
        return x




class TransformerPredictor(nn.Module):
    def __init__(self, latent_dim, seq_len):
        super(TransformerPredictor, self).__init__()
        time_encoding_dim = 2
        self.time2vector = Time2Vector(seq_len=seq_len, latent_dim=latent_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim+time_encoding_dim,
                nhead=3,
                dim_feedforward=256,
                dropout=0.1,
                activation="relu",
            ),
            num_layers=4,
        )
    
    def forward(self, x):
        x = self.time2vector(x)
        x = self.transformer(x)
        return x[:,-1,:]
    
# train or load autoencoder

# autoencoder = ConvolutionalAutoencoder(
#     hidden_feature_dim_1=16,
#     hidden_feature_dim_2=32,
#     hidden_feature_dim_3=64,
#     latent_dim=4,
# ).to(device)
autoencoder: ConvolutionalAutoencoder = torch.load("models/convolutional_autoencoder.pth")

model = TransformerPredictor(latent_dim=4, seq_len=9).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# %% Train
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x in sequence_train_loader:
        batch_size, seq_len, _, _, _ = x.shape
        x = x.to(device)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        with torch.no_grad():
            x = autoencoder.encode(x.permute(0, 3, 1, 2))
        x = x.unflatten(0, (batch_size, seq_len))
        x_train = x[:,:-1,:]
        y_train = x[:,-1,:]


        out = model(x_train)[:,:-2]

        loss = criterion(out, y_train)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss / len(train_loader)
        

epochs = 10
for epoch in range(epochs):
    train_loss = train(model, sequence_train_loader, criterion, optimizer)
    print(f"Epoch {epoch} Loss: {train_loss}")

# %% Test
x = next(iter(sequence_train_loader))
orginal = x.clone()
batch_size, seq_len, _, _, _ = x.shape
x = x.to(device)
x = torch.flatten(x, start_dim=0, end_dim=1)
with torch.no_grad():
    x = autoencoder.encode(x.permute(0, 3, 1, 2))
x = x.unflatten(0, (batch_size, seq_len))
x_test = x[:,:-1,:]
y_test = x[:,-1,:]

out = model(x_test)[:,:-2]
    

print(out.shape, y_test.shape)
out = autoencoder.decoder(out)
y_test = autoencoder.decoder(y_test)

num_rows = 4
num_frames = 5
fig, ax = plt.subplots(num_rows, num_frames+1, figsize=(2*(num_frames+1), 2*num_rows))
# Remove vertical space between subplots
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i in range(num_rows):
    for j in range(num_frames):
        ax[i,j].imshow(orginal[i,j-num_frames].squeeze())
        ax[i,j].axis('off') 
        if i == 0:
            ax[i,j].set_title(f"{10-num_frames+j+1}th")

    ax[i,num_frames].imshow(out[i].detach().cpu().permute(1, 2, 0))
    ax[i,num_frames].axis('off') 
    if i == 0:
        ax[i,num_frames].set_title("Predicted 10th")




# plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.show()

# %% Test autoencoder

latent = torch.rand(1,4, device=device)
out = autoencoder.decoder(latent)
plt.imshow(out[0].detach().cpu().permute(1, 2, 0))
plt.show()
