
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Union
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm

from configs.constants import LR, LR_GAMMA, LR_STEP_SIZE, MOMENTUM, NUM_EPOCHS
from helper.data_helper import get_cifar10_dataloaders, get_data_transforms
from helper.misc import get_device
from models.pretrained_models import get_pretrained_network_for_CIFAR10

def train_model_on_cifar10(
    model_name: str = 'resnet50',
    experiment_suffix: str = '',
    cache_path: Union[Path, str] = 'local/cache/',
    load_from_cache: bool = True,
):

    save_path = Path(cache_path)/f'best_model_{model_name}{experiment_suffix}.pth'
    if save_path.exists() and load_from_cache:
        print(f"Loading model from {save_path}")
        model = get_pretrained_network_for_CIFAR10(model_name, freeze_hidden_layers=False)
        model = load_weights(model, save_path)
        return model
    
    device = get_device()
    data_transforms = get_data_transforms(model_name)

    model       = get_pretrained_network_for_CIFAR10(model_name, freeze_hidden_layers=False).to(device)
    dataloaders = get_cifar10_dataloaders(data_transforms=data_transforms)
    optimizer   = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    model       = train_model(
        model                = model,
        loss_fn              = torch.nn.CrossEntropyLoss(),
        optimizer            = optimizer,
        scheduler            = lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA),
        dataloaders          = dataloaders,
        num_epochs           = NUM_EPOCHS,
        best_model_selection = 'val_loss',
        device               = device
    )

    cache_path.mkdir(parents=True, exist_ok=True)
    save_model(model, save_path)
    print(f"Model saved to {save_path}")
    return model

def train_model(
    model,
    loss_fn,
    optimizer,
    scheduler,
    dataloaders,
    num_epochs=25,
    best_model_selection='val_loss',
    device='cpu',
):
    """
    Train a PyTorch model and save the best model based on validation performance.

    Args:
        model (torch.nn.Module): The model to train.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        num_epochs (int): Number of epochs to train.
        best_model_selection (str): Criterion for selecting the best model ('val_loss' or 'val_acc').
        device (str): Device to train on ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The trained model with the best weights loaded.
    """
    start_time    = time.time()
    best_val_acc  = 0.0
    best_val_loss = float('inf')
    dataset_sizes = {x: len(dataloaders[x].sampler.indices) for x in ['train', 'val']}

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = Path(tempdir) / 'best_model_params.pt'
        torch.save(model.state_dict(), best_model_params_path)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            # Training
            model.train()
            loss_train              = 0.0
            num_correct_preds_train = 0
            for inputs, labels in tqdm(dataloaders['train']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)
                    # Backward pass and optimization (only in training phase)
                    loss.backward()
                    optimizer.step()
                # Statistics
                loss_train += loss.item() * inputs.size(0)
                num_correct_preds_train += torch.sum(preds == labels.data)
            scheduler.step()
            epoch_loss_train = loss_train / dataset_sizes['train']
            epoch_acc_train  = num_correct_preds_train.double() / dataset_sizes['train']
            print(f'TRAINING Loss: {epoch_loss_train:.4f} Acc: {epoch_acc_train:.4f}')
            
            # Validation
            model.eval()
            loss_val              = 0.0
            num_correct_preds_val = 0

            for inputs, labels in tqdm(dataloaders['val']):
                inputs   = inputs.to(device)
                labels   = labels.to(device)
                outputs  = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss     = loss_fn(outputs, labels)
                loss_val += loss.item() * inputs.size(0)
                num_correct_preds_val += torch.sum(preds == labels.data)

            epoch_loss_val = loss_val / dataset_sizes['val']
            epoch_acc_val  = num_correct_preds_val.double() / dataset_sizes['val']
            print(f'VALIDATION Loss: {epoch_loss_val:.4f} Acc: {epoch_acc_val:.4f}')
            
            # Save the best model
            if best_model_selection == 'val_loss' and epoch_loss_val < best_val_loss:
                best_val_loss = epoch_loss_val
                torch.save(model.state_dict(), best_model_params_path)
            elif best_model_selection == 'val_acc' and epoch_acc_val > best_val_acc:
                best_val_acc = epoch_acc_val
                torch.save(model.state_dict(), best_model_params_path)

        # Training complete
        time_elapsed = time.time() - start_time
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        if best_model_selection == 'val_loss':
            print(f'Best Validation Loss: {best_val_loss:.4f}')
        else:
            print(f'Best Validation Accuracy: {best_val_acc:.4f}')

        # Load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def save_model(model, path):
    """
    Save the model to the specified path.

    Args:
        model (torch.nn.Module): The model to save.
        path (str or Path): Path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_weights(model, path, device='cpu'):
    """
    Load the model weights from the specified path.

    Args:
        model (torch.nn.Module): The model to load weights into.
        path (str or Path): Path to the saved weights.
        device (str): Device to map the weights to ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model weights loaded from {path}")
    return model