# face-maker

**face-maker** is a deep learning project for generating and reconstructing face images using invertible neural networks and flow-based models. It is designed for research and experimentation with generative models, particularly those that can map between image and latent spaces in a reversible manner.

## Features

- **Flow-based generative modeling:** Learn invertible mappings between images and latent representations.
- **Image reconstruction:** Reconstruct input images from their latent encodings.
- **Random sampling:** Generate new face images by sampling from the latent space.
- **Configurable training:** Easily adjust hyperparameters, loss weights, and training schedules.
- **Visualization:** Save and compare generated, reconstructed, and perturbed images during training and validation.

## Project Structure
face-maker/ ├── train.py # Training and validation routines ├── models.py # Model architectures (e.g., invertible networks) ├── loss_functions.py # Custom loss functions (e.g., Huber, flow loss) ├── data/ # Dataset and data loading utilities ├── samples/ # Generated and reconstructed image outputs ├── requirements.txt # Python dependencies └── README.md # Project documentation


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/face-maker.git
   cd face-maker

2. **Install dependencies:**
   pip install -r requirements.txt