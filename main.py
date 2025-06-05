import torch
import yaml
from torch.utils.data import DataLoader
from datetime import datetime
import os
from torch.optim import AdamW
import matplotlib.pyplot as plt

from data import ImageLoader
from train import Trainer
from models import Autoencoder, NormalizingFlow
from loss_functions import mse_loss, mae_loss, kl_divergence, huber_loss, covariance_loss, flow_loss





if __name__ == "__main__":

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Results directory
    results_dir = f"results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Load configuration from YAML file
    config = yaml.safe_load(open("config.yaml", "r"))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    train_image_dir, val_image_dir = config["IMAGE_DIRS"]

    # Scrub image_dir and its subdirectories for images
    image_paths_train = []
    image_paths_val = []

    for root, _, files in os.walk(train_image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths_train.append(os.path.join(root, file))

    for root, _, files in os.walk(val_image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths_val.append(os.path.join(root, file))

    print(f"Found {len(image_paths_train)} images in for training.")
    print(f"Found {len(image_paths_val)} images in for validation.")    

    # Create training and validation datasets
    data_train = ImageLoader(image_paths=image_paths_train, image_size=config["IMAGE_SIZE"])
    data_val = ImageLoader(image_paths=image_paths_val, image_size=config["IMAGE_SIZE"])
    
    # Create model, optimizer, loss function, and scheduler
    

    model = NormalizingFlow(
        input_shape=(3, config["IMAGE_SIZE"], config["IMAGE_SIZE"])
    ).to(device)
    print(f"Model architecture: {model}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Load model weights
    if config["PRETRAINED_MODEL_PATH"] is not None:
        if os.path.exists(config["PRETRAINED_MODEL_PATH"]):
            print(f"Loading pretrained weights from {config['PRETRAINED_MODEL_PATH']}")
            model.load_state_dict(torch.load(config["PRETRAINED_MODEL_PATH"], map_location=device), strict=False)

    optimizer = AdamW(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])

    scheduler = None

    # loss_func = lambda output, target, mu, logvar, beta, z :huber_loss(output, target) + beta * kl_divergence(mu, logvar) + 1.0 * covariance_loss(z)
    loss_func = flow_loss

    # Load trainer
    trainer = Trainer(
        model=model,
        criterion=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        sample_dir=results_dir,
        config=config,
        device=device
    )

    train_losses = []
    val_losses = []
    for epoch in range(config["EPOCHS"]):
        print(f"Epoch {epoch + 1}/{config['EPOCHS']}")
        
        # Train for one epoch
        train_loss = trainer.train_epoch(
            data_loader=DataLoader(data_train, batch_size=config["BATCH_SIZE"], shuffle=True),
            epoch=epoch,
            save_sample=True if epoch == 0 else False,
        )
        print(f"Training loss: {train_loss:.4f}")

        # Validate for one epoch
        val_loss = trainer.validate_epoch(
            data_loader=DataLoader(data_val, batch_size=config["BATCH_SIZE"], shuffle=False),
            epoch=epoch,
            save_sample=True if epoch == 0 else False,
        )
        print(f"Validation loss: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % config["SAVE_EVERY"] == 0:
            torch.save(model.state_dict(), f"{results_dir}/model_epoch.pth")
            print(f"Model saved at epoch {epoch + 1}")
    
            # Plot training and validation losses
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss', color='blue')
            plt.plot(val_losses, label='Validation Loss', color='orange')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Losses')
            plt.legend()
            plt.grid("both")
            plt.savefig(f"{results_dir}/loss_plot.png")
            plt.close()

            # Save validation of reverse
            trainer.validate_reverse(epoch)
            

    print()
