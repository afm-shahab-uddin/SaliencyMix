#!/bin/bash

# Train ResNet18 on CIFAR10
CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar10 --model resnet18 --beta 1.0 --salmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1

# Train ResNet50 on CIFAR10
CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar10 --model resnet50 --beta 1.0 --salmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1

# Train ResNet101 on CIFAR10
CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar10 --model resnet101 --beta 1.0 --salmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1

# Train WideResNet on CIFAR10
CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar10 --model wideresnet --beta 1.0 --salmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1




# Train ResNet18 on CIFAR100
CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar100 --model resnet18 --beta 1.0 --salmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1

# Train ResNet50 on CIFAR100
CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar100 --model resnet50 --beta 1.0 --salmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1

# Train ResNet101 on CIFAR100
CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar100 --model resnet101 --beta 1.0 --salmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1

# Train WideResNet on CIFAR100
CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar100 --model wideresnet --beta 1.0 --salmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1
