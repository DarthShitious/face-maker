import torch
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image

class SquareCropTransform(torch.nn.Module):
    def __init__(self, size):
        super(SquareCropTransform, self).__init__()
        self.size = size
        
    def __call__(self, image):
        h, w = image.shape[1:3]
        if h > w:
            diff = (h - w) // 2
            image = image[:, diff:diff + w, :]
        elif w > h:
            diff = (w - h) // 2
            image = image[:, :, diff:diff + h]
        return torchvision.transforms.functional.resize(image, self.size)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, image_paths, image_size):
        self.image_paths = image_paths
        self.square_crop_transform = SquareCropTransform(size=(image_size, image_size))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        image = self.square_crop_transform(image)
        # image = image + torch.randn_like(image)*0.01
        # image = torch.clamp(image, 0, 1)
        return image, image
    
    def load_image(self, image_path):
        image = read_image(image_path).float()
        if image.shape[0] == 1:  # If grayscale, convert to RGB
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:  # If RGBA, convert to RGB
            image = image[:3, :, :]
        return image / 255.0  # Normalize to [0, 1] range
    