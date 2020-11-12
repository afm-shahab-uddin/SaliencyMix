# SaliencyMix
SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization

This implementation is based on 
https://github.com/uoguelph-mlrg/Cutout

### Dependencies  
[PyTorch v0.4.0](http://pytorch.org/)  
[tqdm](https://pypi.python.org/pypi/tqdm)
[OpenCV-Python](https://pypi.org/project/opencv-python/)


### CIFAR 10
To train ResNet18 on CIFAR10 with SaliencyMix and traditional data augmentation:    
`CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar10 --model resnet18 --beta 1.0 --cutmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1`

To train ResNet50 on CIFAR10 with SaliencyMix and traditional data augmentation:
`CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar10 --model resnet50 --beta 1.0 --cutmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1`

To train WideResNet on CIFAR10 with SaliencyMix and traditional data augmentation:
`CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar10 --model wideresnet --beta 1.0 --cutmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1`



### CIFAR 100
To train ResNet18 on CIFAR100 with SaliencyMix and traditional data augmentation:    
`CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar100 --model resnet18 --beta 1.0 --cutmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1`

To train ResNet50 on CIFAR100 with SaliencyMix and traditional data augmentation:
`CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar100 --model resnet50 --beta 1.0 --cutmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1`

To train WideResNet on CIFAR100 with SaliencyMix and traditional data augmentation:
`CUDA_VISIBLE_DEVICES=0,1 python3 saliencymix.py --dataset cifar100 --model wideresnet --beta 1.0 --cutmix_prob 0.5 --batch_size 128 --data_augmentation --learning_rate 0.1`
