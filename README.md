# SaliencyMix
SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization


CIFAR training and testing code is based on 
- [Cutout](https://github.com/uoguelph-mlrg/Cutout)

The ImageNet is based on
- [Cutmix-PyTorch](https://github.com/clovaai/CutMix-PyTorch)


### Requirements  
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)
- NumPy
- OpenCV-contrib-python (4.2.0.32)


### CIFAR
Please use "SaliencyMix_CIFAR" directory

#### CIFAR 10
-To train ResNet18 on CIFAR10 with SaliencyMix and traditional data augmentation:    
```
CUDA_VISIBLE_DEVICES=0,1 python saliencymix.py \
--dataset cifar10 \
--model resnet18 \
--beta 1.0 \
--salmix_prob 0.5 \
--batch_size 128 \
--data_augmentation \
--learning_rate 0.1
```

-To train ResNet50 on CIFAR10 with SaliencyMix and traditional data augmentation:    
```
CUDA_VISIBLE_DEVICES=0,1 python saliencymix.py \
--dataset cifar10 \
--model resnet50 \
--beta 1.0 \
--salmix_prob 0.5 \
--batch_size 128 \
--data_augmentation \
--learning_rate 0.1
```

-To train WideResNet on CIFAR10 with SaliencyMix and traditional data augmentation:    
```
CUDA_VISIBLE_DEVICES=0,1 python saliencymix.py \
--dataset cifar10 \
--model wideresnet \
--beta 1.0 \
--salmix_prob 0.5 \
--batch_size 128 \
--data_augmentation \
--learning_rate 0.1
```


#### CIFAR 100
-To train ResNet18 on CIFAR100 with SaliencyMix and traditional data augmentation:    
```
CUDA_VISIBLE_DEVICES=0,1 python saliencymix.py \
--dataset cifar100 \
--model resnet18 \
--beta 1.0 \
--salmix_prob 0.5 \
--batch_size 128 \
--data_augmentation \
--learning_rate 0.1
```

-To train ResNet50 on CIFAR100 with SaliencyMix and traditional data augmentation:    
```CUDA_VISIBLE_DEVICES=0,1 python saliencymix.py \
--dataset cifar100 \
--model resnet50 \
--beta 1.0 \
--salmix_prob 0.5 \
--batch_size 128 \
--data_augmentation \
--learning_rate 0.1
```

-To train WideResNet on CIFAR100 with SaliencyMix and traditional data augmentation:    
```
CUDA_VISIBLE_DEVICES=0,1 python saliencymix.py \
--dataset cifar100 \
--model wideresnet \
--beta 1.0 \
--salmix_prob 0.5 \
--batch_size 128 \
--data_augmentation \
--learning_rate 0.1
```


### ImageNet
-Please use "SaliencyMix-ImageNet" directory

### Train Examples
- ImageNet with 4 NVIDIA GeForce RTX 2080 Ti GPUs 
```
python train.py \
--net_type resnet \
--dataset imagenet \
--batch_size 256 \
--lr 0.1 \
--depth 50 \
--epochs 300 \
--expname ResNet50 \
-j 40 \
--beta 1.0 \
--salmix_prob 1.0 \
--no-verbose
```

### Test Examples using ImageNet Pretrained models

- Trained models can be downloaded from [here](https://drive.google.com/drive/folders/1vnJHtgzcBInuPZVkwQxQ5A5SE_i_-EON?usp=sharing)

- ResNet-50
```
python test.py \
--net_type resnet \
--dataset imagenet \
--batch_size 64 \
--depth 50 \
--pretrained /runs/ResNet50_SaliencyMix_21.26/model_best.pth.tar
```
- ResNet-101
```
python test.py \
--net_type resnet \
--dataset imagenet \
--batch_size 64 \
--depth 101 \
--pretrained /runs/ResNet101_SaliencyMix_20.09/model_best.pth.tar
```
