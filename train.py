import torch
from torchvision.utils import save_image, make_grid

class BetaScheduler:
    def __init__(self, beta_min, beta_max, rate):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.rate = rate
        self.beta_t = beta_min

    def __call__(self, epoch):
        beta = self.beta_t * self.rate
        beta = max(self.beta_min, min(beta, self.beta_max))
        self.beta_t = beta
        # print(f"beta: {beta:.4f}")
        return beta


# Trainer class for training a model
class Trainer(torch.nn.Module):
    def __init__(
            self,
            model,
            criterion,
            optimizer,
            config,
            sample_dir=None,
            scheduler=None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):
        
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.scheduler = scheduler
        self.sample_dir = sample_dir

        self.beta = BetaScheduler(beta_min=0.1, beta_max=0.1, rate=1.0)



    def train_epoch(self, data_loader, epoch, save_sample=False):
        self.model.train()
        total_loss = 0.0
        for images, labels in data_loader:
            
            # Save smaple grid
            if save_sample and self.sample_dir is not None:
                grid = make_grid(images, nrow=8, normalize=True)
                save_image(grid, f"{self.sample_dir}/input_sample_train.png", normalize=True)

            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            z, ljd = self.model(images)

            loss = self.criterion(z, ljd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def validate_epoch(self, data_loader, epoch, save_sample=False,):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in data_loader:

                # Save smaple grid
                if save_sample and self.sample_dir is not None:
                    grid = make_grid(images, nrow=8, normalize=True)
                    save_image(grid, f"{self.sample_dir}/input_sample_val.png", normalize=True)

                images, labels = images.to(self.device), labels.to(self.device)
                z, ljd = self.model(images)

                loss = self.criterion(z, ljd)
                total_loss += loss.item()
        return total_loss / len(data_loader)
    
    # def validation_sample(self, data_loader, epoch: int):
    #     self.model.eval()
    #     with torch.no_grad():
    #         images, _ = next(iter(data_loader))
    #         images = images.to(self.device)
    #         outputs = self.model(images)
    #         grid = make_grid(torch.concatenate([images, outputs], dim=-1), nrow=2, normalize=True)
    #         save_image(grid, f"{self.sample_dir}/output_sample_val_{epoch:08d}.png", normalize=True)

    def validate_reverse(self, epoch: int):
        self.model.eval()

        with torch.no_grad():
            z = torch.randn(64, 3, self.config["IMAGE_SIZE"], self.config["IMAGE_SIZE"]).to(self.device)
            images = self.model(z, reverse=True)
            print(f"Maximum image value: {images.max()}")
            print(f"Minimum image value: {images.min()}")
            print(f"Mean image value:    {images.mean()}")
            print(f"STDev image value:   {images.std()}")

        torch.clamp(images, 0, 1)
        grid = make_grid(images, nrow=8, normalize=True)
        save_image(grid, f"{self.sample_dir}/latent_sample_val_{epoch+1:08d}.png", normalize=False)
