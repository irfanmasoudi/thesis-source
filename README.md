# Thesis Repository
### Author: Muhammad Irfan Mas'udi
### Supervisor: Prof. Luciano Baresi, Co-Supervisor: Davide Hu
### Master in Computer Science and Engineering @POLIMI
## “DILLEMA: Metamorphic Testing for Deep Learning using Diffusion and Large Language Models”

This repository related to training and testing process of the state-of-the-art architectures. 
Here is [DILLEMA repository](https://github.com/irfanmasoudi/DILLEMA)

DILLEMA is a framework to generate new test cases to uncover the faulty behavior of Image-based Deep Learning application (e.g. Autonomous driving).
In the result, DILLEMA detects 24.8 % pixel-based misclassification of Semantic Segmentation task on Autonomous driving on SHIFT dataset. Moreover, it also found 53.3 % misclassification images in state-of-the-art neural networks for Classification task with ImageNet1K dataset.

## Folder Structure
1. ```SHIFT``` contains notebooks and executable Python for training the model ```DeepLabV3_ResNet50``` and testing processes (DILLEMA augmentation) for Semantic Segmentation task with SHIFT dataset (synthetic dataset for autonomous driving). The model builds on PyTorch Lightning. It contains data exploration and data visualization for the result (e.g. confusion matrix).
<img width="1440" alt="SHIFT Dataset" src="https://github.com/irfanmasoudi/thesis-source/assets/6355974/bcd036bf-f9bd-4ce5-a305-810b7b4aeeb7">



2. ```Imagenet``` contains notebooks and executable for training and testing processes for the state-of-the-art (ResNet18, ResNet50, ResNet152) of pre-trained model for ```ImageNet1K``` which is built on top of PyTorch. This folder also contains data-exploration (Pandas), data-visualization (matplotlib, seaborn).
<img width="1440" alt="image" src="https://github.com/irfanmasoudi/thesis-source/assets/6355974/53290b65-0e14-4d39-9caf-35e58aa45d1a">

## Result
1. Semantic Segmentation on SHIFT Dataset


