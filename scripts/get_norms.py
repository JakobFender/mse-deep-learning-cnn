import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.utils.constants import TRAIN_SET_FOLDER

if __name__ == "__main__":
    dataset = datasets.ImageFolder(TRAIN_SET_FOLDER, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    n = 0

    for imgs, _ in tqdm(loader, desc="Calculating mean and std"):
        # imgs: (B, C, H, W)
        b = imgs.size(0)
        imgs = imgs.view(b, 3, -1)  # (B, C, H * W)
        mean += imgs.mean(dim=[0, 2]) * b
        std += imgs.std(dim=[0, 2]) * b
        n += b

    mean /= n
    std /= n

    print(f"Mean: {mean}")
    print(f"Std:  {std}")
