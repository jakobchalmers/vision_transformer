## Vision Transformer for video frame prediction

In this project, we predict the next frame of a particle in a box. 
Frames of dimension 64 x 64 are compressed using an autoencoder to a latent vector of length 4.
A video of a moving particle is encoded to latent vectors, and a vision transformer is trained to predict the next latent vector in each time step.
The predicted latent vectors are then upsampled to 64 x 64 frames using a convolutional decoder, which is part of the autoencoder.
