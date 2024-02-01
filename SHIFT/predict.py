from typing import Tuple, Any
import torchvision.transforms as transforms
import lightning as pl
import torch
import torchmetrics
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import FileBackend
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T
import segmentation_models_pytorch as smp
# from utils.path import DATA_DIR
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

class SemanticSegmentation(pl.LightningModule):

    def __init__(self, encoder_name: str, input_shape: [int, int, int] = (3, 800, 1200), learning_rate: float = 2e-4,
                 num_classes: int = 23, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.in_channels, self.in_height, self.in_width = input_shape[-3:]
        self.num_classes = num_classes
        self.example_input_array = torch.zeros((1, self.in_channels, self.in_height, self.in_width))
        self.loss = torch.nn.CrossEntropyLoss()
        self.model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=self.in_channels,
            classes=self.num_classes,
        )
        self.total_error = 0
        self.total_true = 0
        self.total_instance = 0
        self.df1 = pd.DataFrame()
        self.overall_cm = np.zeros((23, 23), dtype=np.int64)

    def forward(self, x: Tensor):
        return self.model(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        img, seg = batch
        out = self.model(img)
        loss = self.loss(out, seg)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        img, seg = batch
        out = self.model(img)
        loss = self.loss(out, seg)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        img, seg = batch
        out = self.model(img)
        loss = self.loss(out, seg)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0):
        img, seg = batch
        pred = self(img)
        label = seg.cpu().numpy().flatten()
        output = (torch.max(torch.exp(pred), 1)[1]).cpu().numpy().flatten()
        cm = confusion_matrix(label, output, labels=np.arange(23))
        self.overall_cm += cm
        self.total_instance += len(label)
        # labels = label.flatten()
        # result = output.flatten()
        # with open("preds.png", "ab") as f:
        #     f.write(b"\n")
        #     np.savetxt(f, result, fmt='%i', delimiter=',')
        # with open("labels.png", "ab") as f:
        #     f.write(b"\n")
        #     np.savetxt(f, labels, fmt='%i', delimiter=',')
        # labels = label.flatten()
        # self.df1 = pd.concat([self.df1, pd.DataFrame([result])], ignore_index=True)
        self.df1 = pd.DataFrame(self.overall_cm)
        self.df1.to_excel("original/confmat_original.xlsx")
        error = np.sum(output != label)
        true = np.sum(output == label)
        self.total_error += error
        self.total_true += true
        with open('original/total_error.txt', 'w+') as f:
            f.write(str(self.total_error))
        with open('original/total_true.txt', 'w+') as f:
            f.write(str(self.total_true))
        with open('original/total_instance.txt', 'w+') as f:
            f.write(str(self.total_instance))
        # np.savetxt('total_error.txt', self.total_error)
        # np.savetxt('total_true.txt', self.total_true)
        # np.savetxt('total_instance.txt', self.instance)
        # self.y_pred.append(result)
        
    # def on_predict_epoch_end(self):
    #     np.savetxt('predict1.txt', self.y_pred, fmt='%i', delimiter=',')

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]
    
if __name__ == '__main__':
    model = SemanticSegmentation("resnet18")
    
    ds = SHIFTDataset(
    data_root="../../dataset/shift",
    split="train",
    keys_to_load=[
        Keys.images,
        Keys.intrinsics,
        Keys.boxes2d,
        Keys.boxes2d_classes,
        Keys.boxes2d_track_ids,
        Keys.segmentation_masks,
    ],
    views_to_load=["front"],
    framerate="images",
    shift_type="continuous/1x",
    backend=FileBackend(),  # also supports HDF5Backend(), FileBackend()
    verbose=True,
    )


    class SHIFTDatasetWrapper:
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            # Mixing image at index idx and one at a random index
            img = self.dataset[idx]['front']['images'][0]
            res = self.transform(img) / 255
            seg = self.dataset[idx]['front']['segmentation_masks']
            res_seg = self.transform(seg).flatten(start_dim=0, end_dim=1)
            return res, res_seg
        
        
    preprocess = transforms.Compose([
        transforms.Resize((800,1200))
    ])
    
    dataset = SHIFTDatasetWrapper(ds, transform=preprocess)
    torch.set_float32_matmul_precision('medium')
    
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=32
    )
    
    trainer = Trainer(
        accelerator="gpu",
        devices=[0]
    )
    
    trainer.predict(model, ckpt_path="base_epoch=061_val_loss=0.078.ckpt", dataloaders=test_loader)