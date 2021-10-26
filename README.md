HOW TO REPRODUCE EXPERIMENTS
(Note that training scripts were run on via a SLURM scheduler)

# ResNet50

> sbatch slurm/sbatch_combinact_new {DATASET} {EPOCHS} {RESNET_TYPE} {SEED}

DATASET can be any of:
- MNIST
- cifar10
- cifar100

RESNET_TYPE refers to the width of the ResNet with a 1d actfun such as ReLU. Can be any of:
- 0.5
- 1
- 2
- 4
Higher order activation functions will adjust accordingly to have the same number of parameters.
