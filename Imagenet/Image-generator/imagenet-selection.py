from tqdm import tqdm
import pathlib
import torch
from torchvision.datasets import ImageNet

dataset = ImageNet(root="/home/jovyan/work/dataset/ILSVRC2012")
selected_image_folder = pathlib.Path("lim25-imagenet-image/")
counterfactuals_folder = pathlib.Path("lim25-imagenet-counterfactuals/")

count = 0
last_dir = " "
file_cp = 'cp_selection.txt'
with open(file_cp, 'r') as f:
    check_point = f.read()

t_next = True
for i, elem in tqdm(enumerate(dataset)):
    img, y = elem
    img_name = dataset.imgs[i][0].split('/')
    img_file_name = img_name[-1]
    img_dir = img_name[-2]
    
    if img_dir == 'n02110063':
        t_next = False
    
    if (img_dir != last_dir) and (t_next == False):
        with open(file_cp, 'w') as f:
            f.write(img_dir)
        print(img_dir, img_file_name, count)
        selected_image_file = selected_image_folder.joinpath(img_dir).joinpath(img_file_name)
        counterfactual_file = counterfactuals_folder.joinpath(img_dir).joinpath(img_file_name.replace('JPEG', 'txt'))
        if count > 5 :
            count = 0
            break
            # last_dir = img_dir
            
        else :
            if selected_image_file.exists():
                # count += 1
                continue
            
            else :
                selected_image_file.parent.mkdir(parents=True, exist_ok=True)
                img.save(selected_image_file)
                count += 1