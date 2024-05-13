import deeptrack as dt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

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

class ImageDataset(Dataset):
    def __init__(self, sequence_dataset):
        sequences = sequence_dataset.sequences
        self.images: torch.Tensor = sequences.view(-1, *sequences.shape[2:])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = self.images[index].float()
        return image

def main():
    IMAGE_SIZE = 64
    sequence_length = 10  # Number of frames per sequence
    MIN_SIZE = 0.5e-6
    MAX_SIZE = 1.5e-6
    MAX_VEL = 10  # Maximum velocity. The higher the trickier!
    MAX_PARTICLES = 2  # Max number of particles in each sequence. The higher the trickier!

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
        # optics(particle ** (lambda: 1 + np.random.randint(MAX_PARTICLES))),
        # optics(particle ^ (lambda: 1 + np.random.randint(MAX_PARTICLES))),
        optics(particle ^ (lambda: MAX_PARTICLES)),
        sequence_length=sequence_length,
    )
    train_loader: dt.Sequence = (
        sequential_images >> dt.FlipUD() >> dt.FlipDiagonal() >> dt.FlipLR()
    )

    # Generate and save

    data_size = 2000
    test_data_size = data_size // 10
    FILE_NAME = f"data/consistent_intensity_multiple_particle_dataset_{data_size}.pth"
    TEST_FILE_NAME = f"data/consistent_intensity_multiple_particle_test_dataset_{test_data_size}.pth"

    print(FILE_NAME, TEST_FILE_NAME)
    input("Sure?")

    # Train data
    dataset: SequenceDataset = SequenceDataset(train_loader, data_size)
    torch.save(dataset, FILE_NAME)

    # Test data
    test_dataset: SequenceDataset = SequenceDataset(train_loader, test_data_size)
    torch.save(test_dataset, TEST_FILE_NAME)


if __name__ == "__main__":
    main()