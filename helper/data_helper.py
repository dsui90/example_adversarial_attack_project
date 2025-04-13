from pathlib import Path
from typing import Union
from configs.constants import AVAILABLE_MODELS, BATCH_SIZE, IMG_INPUT_SIZE, NORMALIZATION_RGB_MEAN, NORMALIZATION_RGB_STD, NUM_WORKERS
import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

def get_cifar10_dataloaders(
    data_save_dir:       Union[Path, str]  = 'data',
    val_fraction:        float     =  0.2,
    data_transforms:     dict      = {'train': None, 'val': None, 'test': None}, 
    batch_sizes:         dict[int] = {'train': BATCH_SIZE, 'val': BATCH_SIZE, 'test': BATCH_SIZE},
    shuffle_train_val:   bool      = True,
    shuffle_seed:        int       = 42,
    loader_num_workers:  int       = NUM_WORKERS,
    loader_pin_memory:   bool      = False,
)-> dict[str, torch.utils.data.DataLoader]:
    
    """
    Load CIFAR-10 dataset and create data loaders for training, validation, and testing.
    
    Args:
        data_save_dir (str): Directory to save the dataset.
        val_fraction (float): Fraction of training data to use for validation.
        shuffle_seed (int): Seed for shuffling the dataset.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory for data loading.
        data_transforms (dict): Dictionary containing transformations for train, val, and test datasets.
        batch_sizes (dict): Dictionary containing batch sizes for train, val, and test datasets.
        
    Returns:
        dict: Dictionary containing data loaders for train, val, and test datasets.
    """
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_save_dir, train=True,
        download=True, transform=data_transforms['train'],
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root=data_save_dir, train=True,
        download=True, transform=data_transforms['val'],
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_save_dir, train=False,
        download=True, transform=data_transforms['test'],
    )

    train_val_num_samples = len(train_dataset)
    train_val_indices = list(range(train_val_num_samples))
    split_idx = int(np.floor(val_fraction * train_val_num_samples))

    if shuffle_train_val:
        np.random.seed(shuffle_seed)
        np.random.shuffle(train_val_indices)

    train_indices = train_val_indices[split_idx:]
    val_indices = train_val_indices[:split_idx]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_sizes['train'], sampler=train_sampler,
        num_workers=loader_num_workers, pin_memory=loader_pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_sizes['val'], sampler=val_sampler,
        num_workers=loader_num_workers, pin_memory=loader_pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_sizes['test'],
        num_workers=loader_num_workers, pin_memory=loader_pin_memory,
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }

def get_data_transforms(model_name: str):
    """
    Get data transforms for the specified model.

    Args:
        model_name (str): Name of the model.

    Returns:
        dict: Dictionary containing data transforms for train, val, and test datasets.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model name: {model_name}")

    return {
        'train': 
            torchvision.transforms.Compose([
            torchvision.transforms.Resize(IMG_INPUT_SIZE[model_name]),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(360),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(NORMALIZATION_RGB_MEAN, NORMALIZATION_RGB_STD)
        ]),
        'val': 
            torchvision.transforms.Compose([
            torchvision.transforms.Resize(IMG_INPUT_SIZE[model_name]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(NORMALIZATION_RGB_MEAN, NORMALIZATION_RGB_STD)
        ]),
        'test': 
            torchvision.transforms.Compose([
            torchvision.transforms.Resize(IMG_INPUT_SIZE[model_name]),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize(NORMALIZATION_RGB_MEAN, NORMALIZATION_RGB_STD)
        ]),
    }


def store_cifar10_data_in_folder(model_name, folder_path = 'local'):
    # get test dataloader

    data_transforms = get_data_transforms(model_name)
    dataloaders     = get_cifar10_dataloaders(data_transforms=data_transforms)
    dataloader_test = dataloaders['test']

    cifar10_test_data_path = Path(folder_path) / f'cifar10_test_images_{IMG_INPUT_SIZE[model_name]}/'

    if not cifar10_test_data_path.exists() or not any(cifar10_test_data_path.iterdir()):
        print(f"Storing test data into: {cifar10_test_data_path}")
        store_data_in_folder(cifar10_test_data_path, dataloader_test)


def store_data_in_folder(
    destination_folder: Union[Path, str], 
    dataloader: torch.utils.data.DataLoader,
    suffix: str = '',
) -> None:
    """
    Store test data in a specified folder.

    Args:
        destination_folder (Union[Path, str]): Path to the folder where test data will be stored.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test data.
    """
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)

    for i, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        for j in range(inputs.size(0)):
            img = unnormalize_img_tensor(inputs[j]).squeeze(0)
            img = img.permute(1, 2, 0) # Change from (C, H, W) to (H, W, C)
            img = img.numpy()
            img = (img * 255).astype(np.uint8)
            img_path = destination_folder / f"img_{i * dataloader.batch_size + j}_label_{labels[j]}{suffix}.png"
            cv2.imwrite(str(img_path), img)


def unnormalize_img_tensor(
    image: torch.Tensor, 
    mean:  list = NORMALIZATION_RGB_MEAN,
    std:   list = NORMALIZATION_RGB_STD,
) -> torch.Tensor:
    """
    Unnormalize an image tensor using the provided mean and std.

    Args:
        image (torch.Tensor): The input image tensor.
        mean (list): The mean values for each channel.
        std (list): The standard deviation values for each channel.

    Returns:
        torch.Tensor: The unnormalized image tensor.
    """
    # Convert mean and std to tensors
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    
    # Unnormalize the image
    return image * std + mean


class ImageFolderWithLabels(Dataset):
    """
    Custom Dataset to load images and infer labels from filenames.

    Args:
        folder_path (str): Path to the folder containing images.
        transform (callable, optional): A function/transform to apply to the images.
    """
    def __init__(self, folder_path: str, transform=None):
        self.folder_path = Path(folder_path)
        self.image_paths = list(self.folder_path.glob("*.png"))  # Adjust extension if needed
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Infer label from filename
        label = int(image_path.stem.split("_label_")[-1])

        if self.transform:
            image = self.transform(image)

        return image, label

def dataloader_from_folder_with_labels(folder_path, batch_size, num_workers):
    """
    Create a DataLoader from a folder containing images with labels in filenames.

    Args:
        folder_path (str): Path to the folder containing images.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of workers for the DataLoader.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the images in the folder.
    """
    transform = transforms.Compose([
        transforms.Resize(IMG_INPUT_SIZE['resnet50']),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_RGB_MEAN, NORMALIZATION_RGB_STD)
    ])

    dataset = ImageFolderWithLabels(folder_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader