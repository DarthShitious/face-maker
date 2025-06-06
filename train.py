import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from loss_functions import huber_loss, flow_loss_func
from rich.console import Console
from rich.table import Table
from rich.live import Live

# Trainer class for training a model
class Trainer(torch.nn.Module):
    def __init__(
            self,
            model,
            optimizer,
            config,
            sample_dir=None,
            scheduler=None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):

        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.scheduler = scheduler
        self.sample_dir = sample_dir

    def sample_grid(self, data_loader, filename="sample_grid.png"):
        # Sample images
        images = next(iter(data_loader))

        # Save smaple grid
        grid = make_grid(images, nrow=8, normalize=True)
        save_image(grid, f"{self.sample_dir}/{filename}", normalize=True)

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0.0
        ################# Remove after debugging ###############
        from itertools import islice
        for images in tqdm(islice(data_loader, 100)):
        ########################################################
        # for images in tqdm(data_loader):
            # Load images into tensors
            images = images.to(self.device)

            # Clear gradients
            self.optimizer.zero_grad()

            # Infer on images to retrieve latent space and log of jacobian determinant (ljd)
            z, ljd = self.model(images)

            # Reverse infer on z to reconstuct images
            images_recon = self.model(z, reverse=True)

            # Conjure up random point in latent space
            z_rand = torch.randn_like(z)

            # Perturb that point just a bit...
            z_randpert = z_rand + torch.randn_like(z) * 0.05

            # Calculate loss
            flow_loss = flow_loss_func(z, ljd)
            recon_loss = huber_loss(images, images_recon)
            smooth_loss = huber_loss(self.model(z_rand, reverse=True), self.model(z_randpert, reverse=True))
            loss = flow_loss + 0.1 * recon_loss + 0.1 * smooth_loss

            # Not-so fancy progress reporting
            print("_" * 80)
            print(f"Training Flow Loss:   {flow_loss:.6f}")
            print(f"Training Recon Loss:  {recon_loss:.6f}")
            print(f"Training Smooth_loss: {smooth_loss:.6f}")
            print(f"Training Total Loss:  {loss:.6f}")
            print("_" * 80)

            # Calculate gradients
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            # Backpropagate gradients
            self.optimizer.step()

            # Advance scheduler if necessary
            if self.scheduler is not None:
                self.scheduler.step()

            # Add to total loss
            total_loss += loss.item()

        return total_loss / len(data_loader)

    def validate_epoch(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images in tqdm(data_loader):
                # Load images into tensors
                images = images.to(self.device)

                # Infer on images to retrieve latent space and log of jacobian determinant (ljd)
                z, ljd = self.model(images)

                # Reverse infer on z to reconstuct images
                images_recon = self.model(z, reverse=True)

                # Conjure up random point in latent space
                z_rand = torch.randn_like(z)

                # Perturb that point just a bit...
                z_randpert = z_rand + torch.randn_like(z) * 0.05

                # Calculate loss
                flow_loss_val = flow_loss_func(z, ljd)
                recon_loss_val = huber_loss(images, images_recon)
                smooth_loss_val = huber_loss(self.model(z_rand, reverse=True), self.model(z_randpert, reverse=True))
                loss_val = flow_loss_val + 0.1 * recon_loss_val + 0.1 * smooth_loss_val

                # Not-so fancy progress reporting
                print("-" * 80)
                print(f"Validation Flow Loss:   {flow_loss_val:.6f}")
                print(f"Validation Recon Loss:  {recon_loss_val:.6f}")
                print(f"Validation Smooth Loss: {smooth_loss_val:.6f}")
                print(f"Validation Total Loss:  {loss_val:.6f}")
                print("_" * 80)

                # Add to total loss
                total_loss += loss_val.item()

        return total_loss / 1000 #len(data_loader) <-------------------------------------------------

    def validate_reverse(self, epoch: int):
        self.model.eval()

        with torch.no_grad():
            z = torch.randn(64, 3, self.config["IMAGE_SIZE"], self.config["IMAGE_SIZE"]).to(self.device)
            images = self.model(z, reverse=True) * 0.5
            print(f"Maximum image value: {images.max()}")
            print(f"Minimum image value: {images.min()}")
            print(f"Mean image value:    {images.mean()}")
            print(f"STDev image value:   {images.std()}")

        grid = make_grid(images, nrow=8, normalize=True)
        save_image(grid, f"{self.sample_dir}/latent_sample_val_{epoch+1:08d}.png", normalize=True)

        grid = make_grid(torch.sigmoid(images), nrow=8, normalize=True)
        save_image(grid, f"{self.sample_dir}/latent_sample_val_sigmoid_{epoch+1:08d}.png", normalize=True)

        torch.clamp(images, 0, 1)
        grid = make_grid(images, nrow=8, normalize=False)
        save_image(grid, f"{self.sample_dir}/latent_sample_val_clamped_{epoch+1:08d}.png", normalize=False)

    def validate_recon(self, data_loader, epoch: int):
        self.model.eval()

        # Get one batch of images
        images = next(iter(data_loader))
        images = images[:8].to(self.device)  # First 8 images, move to correct device

        # Forward: image → z
        z, _ = self.model(images)  # Forward pass (get z, ignore log_jac_det)

        # Reverse: z → recon_image
        recons = self.model(z, reverse=True)

        # Reverse w/ noise: z + noise -> recon_image
        recons_noise = self.model(z + torch.randn_like(z) * 0.05, reverse=True)

        # Stack original and reconstructions for comparison
        comps = torch.cat([images, recons, recons_noise], dim=0)  # cat, not concatenate, for torch

        # Save grid with normalization
        grid = make_grid(comps, nrow=8, normalize=True)
        save_image(grid, f"{self.sample_dir}/recon_sample_val_{epoch+1:08d}.png", normalize=True)

        # Save grid with clamping
        comps_clamped = torch.clamp(comps, 0, 1)
        grid_clamped = make_grid(comps_clamped, nrow=8, normalize=False)
        save_image(grid_clamped, f"{self.sample_dir}/recon_sample_val_clamped_{epoch+1:08d}.png", normalize=False)

        # Save grid with sigmoid
        comps_sigmoid = torch.sigmoid(comps)
        grid_sigmoid = make_grid(comps_sigmoid, nrow=8, normalize=False)
        save_image(grid_sigmoid, f"{self.sample_dir}/recon_sample_val_sigmoid_{epoch+1:08d}.png", normalize=False)



