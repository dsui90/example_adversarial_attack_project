
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

from adversarial_attack.base import WhiteBoxBaseClass

class PGD(WhiteBoxBaseClass):
    """
    This class implements the Projected Gradient Descent (PGD) attack for generating adversarial examples.
    PGD is an iterative method that perturbs the input data in the direction of the gradient of the loss with respect to the input.
    """
    
    def generate(self, source_img, gt_label=None, target_label=None, eps_step=0.01, eps=0.1, num_iterations=40):
        """
        Generate adversarial examples using PGD.
        
        Args:
            source_img: Input data.
            gt_label: Ground truth labels.
            target_label: Target labels (optional).
            eps_step: Step size for the attack.
            eps: Maximum perturbation allowed.
            num_iterations: Number of iterations for the attack.
        
        Returns:
            Adversarial examples.
        """
        
        # If input image is numpy array, convert it to PyTorch tensor
        if isinstance(source_img, np.ndarray):
            source_img = torch.from_numpy(source_img).float()
        
        # Ensure the input image is a PyTorch tensor
        if not isinstance(source_img, torch.Tensor):
            raise TypeError("Input image must be a PyTorch tensor.")
        
        perturbed_image = source_img.detach().clone()
    
        for _ in tqdm(range(num_iterations)):
            
            perturbed_image.requires_grad = True
            self.model.zero_grad()
            output = self.model(perturbed_image) 
            if target_label is None:
                loss = - torch.nn.CrossEntropyLoss()(output, gt_label)
            else:         
                loss = torch.nn.CrossEntropyLoss()(output, target_label)
            loss.backward()                  
            grad = perturbed_image.grad.detach()
            grad = grad.sign()
            perturbed_image = perturbed_image + eps_step * grad

        # Projection
        perturbed_image = source_img + torch.clamp(perturbed_image - source_img, min=-eps, max=eps)
        perturbed_image = perturbed_image.detach()
        perturbed_image = torch.clamp(perturbed_image, min=0, max=1)
        return perturbed_image