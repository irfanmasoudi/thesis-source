# Thesis Repository
## “DILLEMA: Metamorphic Testing for Deep Learning using Diffusion and Large Language Models”
A framework to generate new test cases to uncover the faulty behavior of Image-based Deep Learning application (e.g. Autonomous driving).
In the result, DILLEMA detects 24.8 % pixel-based misclassification of Semantic Segmentation task on Autonomous driving SHIFT dataset. Moreover, it also founds 53.3 % misclassification images in state-of-the-art neural networks for Classification task with ImageNet1K dataset.

## Folder Structure
1. ```SHIFT``` contains notebooks and executable Python for training the model ```DeepLabV3_ResNet50``` and testing processes (DILLEMA augmentation) for Semantic Segmentation task with SHIFT dataset (synthetic dataset for autonomous driving). The model builds on PyTorch Lightning. It contains data exploration and data visualization for the result (e.g. confusion matrix).
2. ```Imagenet``` contains notebooks and executable for training and testing processes for the state-of-the-art (ResNet18, ResNet50, ResNet152) of pre-trained model for ```ImageNet1K```.
