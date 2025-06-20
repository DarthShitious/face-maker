import torch
import yaml
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import os
from torch.optim import AdamW
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random

from train import Trainer
from models import CNNRealNVPFlow
from data import CelebALoader




if __name__ == "__main__":

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Results directory
    results_dir = f"results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Load configuration from YAML file
    config = yaml.safe_load(open("config.yaml", "r"))

    # Save a copy of the configuration
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    data_train_full = CelebALoader(root="data", split="train", image_size=config["IMAGE_SIZE"], download=True)
    data_val_full = CelebALoader(root="data", split="valid", image_size=config["IMAGE_SIZE"], download=True)

    # Instantiate model
    model = CNNRealNVPFlow(input_shape=(3, config["IMAGE_SIZE"], config["IMAGE_SIZE"])).to(device)

    print(f"Model architecture: {model}")
    print(f"Model parameters:   {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Load model weights
    if config["PRETRAINED_MODEL_PATH"] is not None:
        if os.path.exists(config["PRETRAINED_MODEL_PATH"]):
            print(f"Loading pretrained weights from {config['PRETRAINED_MODEL_PATH']}")
            model.load_state_dict(torch.load(config["PRETRAINED_MODEL_PATH"], map_location=device), strict=False)

    optimizer = AdamW(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])

    # Instantiate scheduler
    scheduler = None

    # Instantiate trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sample_dir=results_dir,
        config=config,
        device=device
    )

    # Train
    train_losses = []
    val_losses = []
    best_loss = np.inf
    best_epoch = 0
    for epoch in tqdm(range(config["EPOCHS"])):
        print(f"Epoch {epoch + 1}/{config['EPOCHS']}")

        # Create data loaders
        subset_indices = random.sample(range(len(data_train_full)), config["TRAIN_SIZE"])
        data_train = Subset(data_train_full, subset_indices)

        data_val = Subset(
            data_val_full,
            range(min(config["TEST_SIZE"], len(data_val_full)))
        )

        dataloader_train = DataLoader(data_train, batch_size=config["BATCH_SIZE"], shuffle=True)
        dataloader_val = DataLoader(data_val, batch_size=config["BATCH_SIZE"], shuffle=False)

        print(f"Number of images in training set: {len(data_train)}")
        print(f"Number of images in validation set: {len(data_val)}")

        # Sample datasets
        if epoch == 0:
            trainer.sample_grid(data_loader=dataloader_train, filename="sample_grid_train.png")
            trainer.sample_grid(data_loader=dataloader_val, filename="sample_grid_val.png")

        # Train for one epoch
        train_loss = trainer.train_epoch(data_loader=dataloader_train)
        print(f"Training loss: {train_loss:.4f}")

        # Validate for one epoch
        val_loss = trainer.validate_epoch(data_loader=dataloader_val)
        print(f"Validation loss: {val_loss:.4f}")

        if val_loss < best_loss:
            print(f"New best loss!")
            torch.save(model.state_dict(), f"{results_dir}/best_model.pth")
            best_epoch = epoch

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % config["SAVE_EVERY"] == 0:
            torch.save(model.state_dict(), f"{results_dir}/current_model.pth")
            print(f"Model saved at epoch {epoch + 1}")

            # Plot training and validation losses
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss', color='blue')
            plt.plot(val_losses, label='Validation Loss', color='orange')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.ylim(0, 1)
            plt.title('Training and Validation Losses')
            plt.legend()
            plt.grid("both")
            plt.savefig(f"{results_dir}/loss_plot.png")
            plt.close()

            # Save validation of reverse
            trainer.validate_reverse(epoch)

            # Save validation of recon
            trainer.validate_recon(
                data_loader=dataloader_val,
                epoch=epoch
            )


    print()
