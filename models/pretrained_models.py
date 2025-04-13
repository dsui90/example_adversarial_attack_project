
import torch
import torchvision

from configs.constants import NUM_CLASSES_CIFAR10


def get_pretrained_network(
    model_name: str,
) -> torch.nn.Module:
    """
    Load a pre-trained model without modifications.

    Args:
        model_name (str): Name of the model ('mobilenetv2' or 'resnet50').

    Returns:
        nn.Module: The selected model.
    """
    if model_name == 'mobilenetv2':
        model = torchvision.models.mobilenet_v2(
            weights = 'IMAGENET1K_V1',
        )
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(
            weights = 'IMAGENET1K_V1',
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

def get_pretrained_network_for_CIFAR10(
    model_name:           str, 
    freeze_hidden_layers: bool = False,
) -> torch.nn.Module:
    """
    Load a pre-trained model and modify it for CIFAR-10 classification.

    Args:
        model_name (str): Name of the model ('mobilenetv2' or 'resnet50').
        num_classes (int): Number of output classes (default is 10 for CIFAR-10).
        freeze_layers (bool): If True, freeze the layers of the model (default is True).

    Returns:
        nn.Module: The modified model.
    """
    model = get_pretrained_network(model_name)
    if freeze_hidden_layers:
        for param in model.parameters():
            param.requires_grad = False
            
    # Modify the final fully connected layer to match the number of classes
    if model_name == 'mobilenetv2':
        model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES_CIFAR10)
        for param in model.classifier[1].parameters():
            param.requires_grad = True
    elif model_name == 'resnet50':
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES_CIFAR10)
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model