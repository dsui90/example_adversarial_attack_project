from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import os  # Add this import for checking file existence

from adversarial_attack.fgsm import FGSM
from adversarial_attack.pgd import PGD

def evaluate_model(
    model, 
    dataloader_test, 
    attack_method='fgsm',
    model_surrogate = None,
    eps_range=[0.0, 1.0, 0.1], 
    device='cpu',
    output_folder='local/evaluation_results/',
    experiment_suffix='',
):
    """
    Evaluate the model on the test set.
    Use the specified attack method to generate adversarial examples and evaluate the model's performance.
    We use the attack success rate as the evaluation metric.
    """

    if model_surrogate is None:
        model_surrogate = model
    if attack_method == 'fgsm':
        attacker = FGSM(model_surrogate)
    elif attack_method == 'pgd':
        attacker = PGD(model_surrogate)
    else:
        raise ValueError(f"Unknown attack method: {attack_method}")
    
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    for eps in np.arange(*eps_range):
        print(f"Evaluating with eps: {eps}")
        num_successful_attacks_total = 0
        for i, (inputs, labels) in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Generate adversarial examples
            perturbed_inputs = attacker.generate(
                inputs, 
                gt_label=labels, 
                target_label=None,
                eps=eps,
                device=device,
            )
            outputs = model(inputs)
            output_labels = torch.argmax(outputs, dim=1)
            
            outputs_perturbed = model(perturbed_inputs)
            outputs_perturbed_labels = torch.argmax(outputs_perturbed, dim=1)
            num_successful_attacks = (output_labels != outputs_perturbed_labels).sum().item()
            num_successful_attacks_total += num_successful_attacks
        print(f"Successful attacks for eps {eps}: {num_successful_attacks_total}")
        print(f"Successful attack rate for eps {eps}: {num_successful_attacks_total / len(dataloader_test.dataset)}")
        print(f"Total images: {len(dataloader_test.dataset)}")
        
        # Add data to CSV file
        csv_file_path = output_folder_path / f"evaluation_results_{attack_method}{experiment_suffix}.csv"
        file_exists = os.path.exists(csv_file_path)

        with open(csv_file_path, 'a') as f:
            # Write header if the file is empty or doesn't exist
            if not file_exists or os.stat(csv_file_path).st_size == 0:
                f.write("epsilon,num_successful_attacks,attack_success_rate\n")
            f.write(f"{eps},{num_successful_attacks_total},{num_successful_attacks_total / len(dataloader_test.dataset)}\n")

