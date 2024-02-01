from typing import Tuple, Any

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
from utils.path import DATA_DIR


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
        img, _ = batch
        pred = self(img)
        return pred

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]


if __name__ == '__main__':
    model = SemanticSegmentation("resnet18", learning_rate=2e-3, input_shape=(10, 3, 800, 1200))

    ds = SHIFTDataset(
        data_root=DATA_DIR.joinpath("shift"),
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
        def __init__(
                self,
                dataset,
                transform=T.ToTensor(),
        ) -> None:
            self.dataset = dataset
            self.transform = transform
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            # Mixing image at index idx and one at a random index
            img = self.dataset[idx]['front']['images'][0] / 255
            seg = self.dataset[idx]['front']['segmentation_masks'].flatten(start_dim=0, end_dim=1)
            return img, seg


    dataset = SHIFTDatasetWrapper(ds)
    torch.set_float32_matmul_precision('medium')

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=12,
        shuffle=True,
        prefetch_factor=2,
        num_workers=32
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=12,
        shuffle=False,
        num_workers=32
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./',  # path to save the checkpoints
        filename='base_{epoch:03d}_{val_loss:.3f}',
        # filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=20,
        mode='min',
    )

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=200,
        callbacks=[
            checkpoint_callback,
        ],
        devices=[0],
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

