import yaml  # Import PyYAML for loading the configuration file
from helper.data_helper import get_cifar10_dataloaders, get_data_transforms
from helper.evaluation_helper import evaluate_model
from helper.misc import get_device, print_cuda_info
from helper.training_helper import train_model_on_cifar10
from models.pretrained_models import get_pretrained_network


def load_config(config_path):
    """
    Load the configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main(config):
    '''
    Main function to run the evaluation process.
    Args:
        config (dict): Configuration dictionary containing all parameters.
    '''
    # Print CUDA information and get the device
    print_cuda_info()
    device = get_device()

    # Load models
    model_target = train_model_on_cifar10(
        model_name=config['model_target_name'], 
        experiment_suffix=config['experiment_suffix_target']
    )
    
    model_surrogates = {}
    for name, suffix in config['model_surrogates'].items():
        if name is None:
            model_surrogates['no_surrogate'] = None
            continue
        if suffix:
            model_surrogates[name] = train_model_on_cifar10(model_name=name, experiment_suffix=suffix)
        else:
            model_surrogates[name] = train_model_on_cifar10(model_name=name)

    model_target.eval()  # Set the target model to evaluation mode

    # Load data
    data_transforms = get_data_transforms(config['model_target_name'])
    dataloaders = get_cifar10_dataloaders(data_transforms=data_transforms)
    dataloader_test = dataloaders['test']

    # Evaluate the model with different configurations
    for attack_method in config['attack_methods']:
        for surrogate_name, surrogate_model in model_surrogates.items():
            evaluate_model(
                model=model_target,
                dataloader_test=dataloader_test,
                model_surrogate=surrogate_model,
                attack_method=attack_method,
                eps_range=config['eps_range'],
                device=device,
                output_folder=config['output_folder'],
                experiment_suffix=f"_{surrogate_name}_{attack_method}"
            )


if __name__ == "__main__":
    '''
    Main entry point of the script.
    Loads the configuration from a YAML file and runs the main function.
    '''
    # Load configuration from YAML file
    config_path = "configs/config.yaml"
    config = load_config(config_path)

    main(config)
