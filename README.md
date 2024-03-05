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
<br /><br />

2. ```Imagenet``` contains notebooks and executable for training and testing processes for the state-of-the-art (ResNet18, ResNet50, ResNet152) of pre-trained model for ```ImageNet1K``` which is built on top of PyTorch. This folder also contains data-exploration (Pandas), data-visualization (matplotlib, seaborn).
<img width="1440" alt="image" src="https://github.com/irfanmasoudi/thesis-source/assets/6355974/53290b65-0e14-4d39-9caf-35e58aa45d1a">
<br /><br />

## Result: Comparison between original and DILLEMA Augmented 
1. SHIFT Dataset on Semantic Segmentation task
<img width="1043" alt="Screenshot 2023-12-13 at 20 10 12 copy" src="https://github.com/irfanmasoudi/thesis-source/assets/6355974/6a56afe5-1e39-4ce8-93c5-9e3550dbf0dc">

<br /><br />
<img width="1438" alt="image" src="https://github.com/irfanmasoudi/thesis-source/assets/6355974/95f0248b-499e-428e-b7a8-36e9cfbfc1d1"> <br /><br />

3. ImageNet1K Dataset for Classification task
<img width="978" alt="Screenshot 2023-12-13 at 17 58 08 copy" src="https://github.com/irfanmasoudi/thesis-source/assets/6355974/025be22f-98be-463b-b083-95d502e5aabe">
<br />
Because the confusion matrix is vast, here we attach several spreadsheet results:

[ResNet18 Original](https://docs.google.com/spreadsheets/d/1YZ-g10NW4lC5UMA4rVZrvNnC5LDT7zdA2HM-qZFAous/edit?usp=sharing)
[ResNet18 DILLEMA](https://docs.google.com/spreadsheets/d/12RDB1xP20rOzTWpPN5XLp96DRPsqJdh2qxU-Dn5EXAs/edit?usp=sharing)
[ResNet50 Original](https://docs.google.com/spreadsheets/d/1go6huKzcYEAwSpHpMyqMCBHURA0ON29H/edit?usp=sharing&ouid=107614737529128863932&rtpof=true&sd=true)
[ResNet50 DILLEMA](https://docs.google.com/spreadsheets/d/1GEhledp1DPOKBU1eZjNV-g7rwaKiTiPn/edit?usp=sharing&ouid=107614737529128863932&rtpof=true&sd=true)
[ResNet152 Original](https://docs.google.com/spreadsheets/d/1sO8gISG1VgVcK3AQh09VKIbm3tmNAPo_/edit?usp=sharing&ouid=107614737529128863932&rtpof=true&sd=true)
[ResNet152 DILLEMA](https://docs.google.com/spreadsheets/d/14VRt01-Cb0QqK6RWIt7tcXi9YFr90zNv/edit?usp=sharing&ouid=107614737529128863932&rtpof=true&sd=true)



