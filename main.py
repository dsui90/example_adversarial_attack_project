import sys
import numpy as np
from configs.constants import IMG_INPUT_SIZE
import cv2
import torch.nn.functional as F

import torch

import numpy as np
import torchvision
from PIL import Image
from pathlib import Path

from helper.data_helper import get_cifar10_dataloaders, store_cifar10_data_in_folder, store_data_in_folder
from helper.misc import get_device, print_cuda_info
from helper.training_helper import train_model_on_cifar10
from models.pretrained_models import get_pretrained_network


print_cuda_info()
device = get_device()

# get test dataloader
model_name = 'resnet50'
store_cifar10_data_in_folder(model_name=model_name, folder_path='data/')


#load model
model_target      = train_model_on_cifar10(model_name='resnet50', experiment_suffix='_target')
model_surrogate_0 = get_pretrained_network('mobilenetv2')
model_surrogate_1 = train_model_on_cifar10(model_name='mobilenetv2', experiment_suffix='_surrogate')
model_surrogate_2 = train_model_on_cifar10(model_name='resnet50', experiment_suffix='_surrogate')

sys.exit(42)
model_target.eval()  # Set the model to evaluation mode

if False:
    # get test dataloader
    data_transforms = get_cifar10_data_transforms_resnet50()
    dataloaders     = get_cifar10_dataloaders(data_transforms=data_transforms)
    dataloader_test = dataloaders['test']

    cifar10_test_data_path = Path('local/cifar10_test_images/')

    if not cifar10_test_data_path.exists() or not any(cifar10_test_data_path.iterdir()):
        print(f"Storing test data into: {cifar10_test_data_path}")
        store_data_in_folder(cifar10_test_data_path, dataloader_test)

normalizer = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# load image
image_path = Path('local/cifar10_test_images/img_1_label_8.png')
img = Image.open(image_path)
img = img.convert('RGB')
img = img.resize((224, 224))
img = transforms.ToTensor()(img)
img = img.unsqueeze(0)  # Add batch dimension
model.eval()


#perturbed_image, data_grad = fgsm(img, model, torch.tensor([0]), 0.1)



def save_perturbed_img(perturbed_image, img, suffix=''):

    diff_img = perturbed_image - img


    # show before and after
    img_np = img.squeeze(0).permute(1, 2, 0).detach().numpy()
    perturbed_image_np = perturbed_image.squeeze(0).permute(1, 2, 0).detach().numpy()
    diff_img_np = diff_img.squeeze(0).permute(1, 2, 0).detach().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    perturbed_image_np = (perturbed_image_np * 255).astype(np.uint8)
    diff_img_np = (diff_img_np * 255).astype(np.uint8)
    cv2.imwrite(f'00_original_image{suffix}.png', img_np)
    cv2.imwrite(f'00_perturbed_image{suffix}.png', perturbed_image_np)
    cv2.imwrite(f'00_difference_image{suffix}.png', diff_img_np)
    print("Images saved as 'original_image.png', 'perturbed_image.png', and 'difference_image.png'")

    print("---")
    print(torch.argmax(model(img)))
    print(torch.argmax(model(perturbed_image)))
    print("---")

perturbed_image = pgd(img, model, torch.tensor([8]), torch.tensor([0]), 0.01, 0.01, 100)
save_perturbed_img(perturbed_image, img, suffix='')  # Save the perturbed image

perturbed_image = blackbox_pgd(img, surr_model_name='mobilenetv2_cifar10_model')
save_perturbed_img(perturbed_image, img, suffix='_mobnetv2')  # Save the perturbed image