# example_adversarial_attack_project
Exemplary implemenation of the adversarial attacks
- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)

# Setting up the Environment
This project runs in a conda environment.
To set it up, please set up conda on your distribution
and run the following terminal commands:

```bash
conda create -n attack_env python=3.11
conda activate attack_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python tqdm
```
# What does main.py do?
Once everything is set up,
you can run 
```
python main.py
```
to run the experiments.

# Current experiments
Currenty, `FGSM` and `PGD` are used to attack a `Resnet50` model,
which was pretrained on `ImageNet` and has been fine tuned 
on `CIFAR-10`.
The attacks modify any incoming test image from the `CIFAR-10` test split.
Any successful deviation from the original prediction is considered a successful attack.
Using all test samples, a attack success rate is calculated for any deviation boundary
in the range of [0.1, 0.9] with a step size of 0.01.

Furthermore, the attacks are conducted in the intended WhiteBox manner,
where the target model is known, but also in a BlackBox manner,
where a surrogate model is used to generate the adversaries,
in the hope that these will be transferable to the original (unknown) target model.

For this, 3 scenarios are covered:
- We don't know the target model architecture, nor its application domain.
The surrogate model here is a `MobileNet_v2`, pretrained on `ImageNet` "only" without any fine tuning.
- We don't know the target model architecture, but its application domain, that is `CIFAR-10`.
The surrogate model here is a `MobileNet_v2`, pretrained on `ImageNet` and fine tuned on `CIFAR-10`.
- We know the target model architecture (but don't have access to the trained weights), 
and we know the application domain.
The surrogate model here is a `Resnet50`, pretrained on `ImageNet` and fine tuned on `CIFAR-10` (with all weights unfrozen).

The attack success rates are all exported to a `.csv` file.

# Folder structure
The project contains the following folders:
- `adversarial_attack` which contains the implemented attack methods. Currently, this is `FGSM` and `PGD`.
- `configs` which contains anything related to assumed constants and parameter configurations.
- `data` which will be automatically created locally in the first run and which will store the CIFAR-10 related data.
- `helper` which contains any helper functions to get the code running.
- `local` which contains any cache or local files, which shall not be pushed to the repository.
- `models` which contains code regarding the used models.


