
import torch
from adversarial_attack.pgd import pgd
from models.pretrained_models import get_mobilenetv2_cifar10_model, get_mobilenetv2_model, get_resnet50_cifar10_model, get_resnet50_model


def blackbox_pgd(img, surr_model_name='mobilenetv2'):
    if surr_model_name == 'mobilenetv2':
        surr_model = get_mobilenetv2_model()
    elif surr_model_name == 'resnet50':
        surr_model = get_resnet50_model()
    elif surr_model_name == 'resnet50_cifar10':
        surr_model = get_resnet50_cifar10_model(suffix='_blackbox_surrogate')
    elif surr_model_name == 'mobilenetv2_cifar10_model':
        surr_model = get_mobilenetv2_cifar10_model(suffix='_blackbox_surrogate')
    else:
        raise ValueError(f"Unknown surrogate model name: {surr_model_name}")
    
    return pgd(img, surr_model, torch.tensor([8]), torch.tensor([0]), 0.1, 0.1, 100)