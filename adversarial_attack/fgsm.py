
import numpy as np
import torch

from adversarial_attack.base import WhiteBoxBaseClass

class FGSM(WhiteBoxBaseClass):
    """
    This class implements the Fast Gradient Sign Method (FGSM) for generating adversarial examples.
    FGSM is a method to create adversarial examples by perturbing the input data in the direction of the gradient of the loss with respect to the input.
    """
    def __init__(self, model, epsilon=0.1):
        super(FGSM, self).__init__(model)
        self.epsilon = epsilon

    def generate(self, source_img, gt_label, target_label=None, eps=0.1, device='cpu'):
        """
        Generate adversarial examples using FGSM.

        Args:
            source_img: Input data.
            gt_label: Ground truth labels.
            target_label: Target labels (optional).
            **kwargs: Additional parameters for the attack.

        Returns:
            Adversarial examples.
        """
        # If input image is numpy array, convert it to PyTorch tensor
        if isinstance(source_img, np.ndarray):
            source_img = torch.from_numpy(source_img).float()
        
        # Ensure the input image is a PyTorch tensor
        if not isinstance(source_img, torch.Tensor):
            raise TypeError("Input image must be a PyTorch tensor.")
        
        self.model.to(device)
        source_img = source_img.to(device)
        gt_label = gt_label.to(device)
        if target_label is not None:
            target_label = target_label.to(device)
            
        self.model.eval()
        source_img.requires_grad = True
        output = self.model(source_img)
        
        if target_label is None:
            loss = - torch.nn.CrossEntropyLoss()(output, gt_label)
        else:
            loss = torch.nn.CrossEntropyLoss()(output, target_label)
        
        self.model.zero_grad()
        loss.backward()
        data_grad = source_img.grad.data
        
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = source_img - eps * data_grad.sign()
        
        # Clip the perturbed image to ensure pixel values are valid
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image