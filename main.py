from helper.data_helper import  get_cifar10_dataloaders, get_data_transforms, store_cifar10_data_in_folder, store_data_in_folder
from helper.evaluation_helper import evaluate_model
from helper.misc import get_device, print_cuda_info
from helper.training_helper import train_model_on_cifar10
from models.pretrained_models import get_pretrained_network


print_cuda_info()
device = get_device()


model_name = 'resnet50'

#load models
model_target      = train_model_on_cifar10(model_name='resnet50', experiment_suffix='_from_other')
model_surrogate_0 = get_pretrained_network('mobilenetv2')
model_surrogate_1 = train_model_on_cifar10(model_name='mobilenetv2', experiment_suffix='_blackbox_surrogate')
model_surrogate_2 = train_model_on_cifar10(model_name='resnet50', experiment_suffix='_blackbox_surrogate')

model_target.eval()  # Set the model to evaluation mode

data_transforms = get_data_transforms('resnet50')
dataloaders     = get_cifar10_dataloaders(data_transforms=data_transforms)
dataloader_test = dataloaders['test']

evaluate_model(
    model_target,
    dataloader_test,
    attack_method='fgsm',
    eps_range=[0.01, 0.1, 0.01],
    device=device,
    output_folder='local/evaluation_results/',
    experiment_suffix='_no_surrogate'
)

evaluate_model(
    model_target,
    dataloader_test,
    model_surrogate=model_surrogate_0,
    attack_method='fgsm',
    eps_range=[0.01, 0.1, 0.01],
    device=device,
    output_folder='local/evaluation_results/',
    experiment_suffix='_mobilenetv2_vanilla_surrogate'
)

evaluate_model(
    model_target,
    dataloader_test,
    model_surrogate=model_surrogate_1,
    attack_method='fgsm',
    eps_range=[0.01, 0.1, 0.01],
    device=device,
    output_folder='local/evaluation_results/',
    experiment_suffix='_mobilenetv2_retrained_surrogate'
)

evaluate_model(
    model_target,
    dataloader_test,
    model_surrogate=model_surrogate_2,
    attack_method='fgsm',
    eps_range=[0.01, 0.1, 0.01],
    device=device,
    output_folder='local/evaluation_results/',
    experiment_suffix='_resnet50_retrained_surrogate'
)

evaluate_model(
    model_target,
    dataloader_test,
    attack_method='pgd',
    eps_range=[0.01, 0.1, 0.01],
    device=device,
    output_folder='local/evaluation_results/',
    experiment_suffix='_no_surrogate'
)

evaluate_model(
    model_target,
    dataloader_test,
    model_surrogate=model_surrogate_0,
    attack_method='pgd',
    eps_range=[0.01, 0.1, 0.01],
    device=device,
    output_folder='local/evaluation_results/',
    experiment_suffix='_mobilenetv2_vanilla_surrogate'
)

evaluate_model(
    model_target,
    dataloader_test,
    model_surrogate=model_surrogate_1,
    attack_method='pgd',
    eps_range=[0.01, 0.1, 0.01],
    device=device,
    output_folder='local/evaluation_results/',
    experiment_suffix='_mobilenetv2_retrained_surrogate'
)

evaluate_model(
    model_target,
    dataloader_test,
    model_surrogate=model_surrogate_2,
    attack_method='pgd',
    eps_range=[0.01, 0.1, 0.01],
    device=device,
    output_folder='local/evaluation_results/',
    experiment_suffix='_resnet50_retrained_surrogate'
)
