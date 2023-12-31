import os
import random
import warnings

warnings.filterwarnings("ignore")
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, rlimit[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

torch.set_float32_matmul_precision("high")

from typing import Optional
from lightning_fabric.utilities.seed import seed_everything
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger

from monai.networks.layers import MedianFilter, BilateralFilter 
from monai.networks.layers.factories import Norm
from monai.networks.nets import Unet

from torchmetrics.image import TotalVariation
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from generative.losses import PerceptualLoss

from argparse import ArgumentParser

from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    look_at_view_transform,
)

from pytorch_histogram_matching import Histogram_Matching

from datamodule import UnpairedDataModule
from dvr.renderer import DirectVolumeFrontToBackRenderer
from nerv2.renderer import NeRVFrontToBackInverseRenderer

backbones = {
    "efficientnet-b0": (16, 24, 40, 112, 320),
    "efficientnet-b1": (16, 24, 40, 112, 320),
    "efficientnet-b2": (16, 24, 48, 120, 352),
    "efficientnet-b3": (24, 32, 48, 136, 384),
    "efficientnet-b4": (24, 32, 56, 160, 448),
    "efficientnet-b5": (24, 40, 64, 176, 512),
    "efficientnet-b6": (32, 40, 72, 200, 576),
    "efficientnet-b7": (32, 48, 80, 224, 640),
    "efficientnet-b8": (32, 56, 88, 248, 704),
    "efficientnet-l2": (72, 104, 176, 480, 1376),
}


def make_cameras_dea(
    dist: torch.Tensor, 
    elev: torch.Tensor, 
    azim: torch.Tensor, 
    fov: int = 10, 
    znear: int = 18.0, 
    zfar: int = 22.0, 
    is_orthogonal: bool = False
):
    assert dist.device == elev.device == azim.device
    _device = dist.device
    R, T = look_at_view_transform(dist=dist.float(), elev=elev.float() * 90, azim=azim.float() * 180)
    if is_orthogonal:
        return FoVOrthographicCameras(R=R, T=T, znear=znear, zfar=zfar).to(_device)
    return FoVPerspectiveCameras(R=R, T=T, fov=fov, znear=znear, zfar=zfar).to(_device)

class HistogramMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss(reduction="mean")
        self.hmatch = Histogram_Matching(differentiable=True)
        
    def forward(self, ref, dst):
        rst = self.hmatch(dst, ref)
        return self.l1loss(dst, rst)        

class MappingEmbedding(nn.Module):
    def __init__(self, num_bins=256):
        super(MappingEmbedding, self).__init__()
        self.num_bins = num_bins
        self.embedding = nn.Embedding(self.num_bins, 1)
        
        # Initialize the embedding layer weights to ones
        nn.init.constant_(self.embedding.weight, 1.0)

    def forward(self, input_tensor):
        # Ensure input values are in the range (0, 1)
        input_tensor = torch.clamp(input_tensor, 0, 1)

        # Calculate the bin index for each element in the input tensor
        bin_indices = (input_tensor * (self.num_bins - 1)).floor().long()

        # Map the input tensor to the range (0, 1) using the learnable one-hot encoding
        one_hot = self.embedding(bin_indices.view(-1))
        one_hot = one_hot.view(*bin_indices.size(), -1)

        # Map the input tensor to the range (0, 1) using the bin centers
        bin_centers = (bin_indices.float() + 0.5) / self.num_bins
        mapped_tensor = (one_hot * bin_centers.unsqueeze(-1)).sum(dim=-1).reshape(input_tensor.shape)

        return mapped_tensor

class DXRLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.lr = hparams.lr

        self.ckpt = hparams.ckpt
        self.strict = hparams.strict
        self.img_shape = hparams.img_shape
        self.vol_shape = hparams.vol_shape
        self.fov_depth = hparams.fov_depth
        self.alpha = hparams.alpha
        self.gamma = hparams.gamma
        self.delta = hparams.delta
        self.theta = hparams.theta
        self.omega = hparams.omega
        self.lamda = hparams.lamda
        self.timesteps = hparams.timesteps

        self.logsdir = hparams.logsdir
        self.sh = hparams.sh
        self.pe = hparams.pe

        self.n_pts_per_ray = hparams.n_pts_per_ray
        self.weight_decay = hparams.weight_decay
        self.batch_size = hparams.batch_size
        self.devices = hparams.devices
        self.backbone = hparams.backbone

        self.save_hyperparameters()

        self.fwd_renderer = DirectVolumeFrontToBackRenderer(
            image_width=self.img_shape, 
            image_height=self.img_shape, 
            n_pts_per_ray=self.n_pts_per_ray, 
            min_depth=4.0, 
            max_depth=8.0, 
            ndc_extent=2.0,
        )

        self.inv_renderer = NeRVFrontToBackInverseRenderer(
            in_channels=1, 
            out_channels=self.sh ** 2 if self.sh > 0 else 1, 
            vol_shape=self.vol_shape, 
            img_shape=self.img_shape, 
            fov_depth=self.fov_depth, 
            sh=self.sh, 
            pe=self.pe, 
            backbone=self.backbone, 
            fwd_renderer=self.fwd_renderer,
        )
        
        self.inv_alpharer = MappingEmbedding(num_bins=65536)

        if self.ckpt:
            print("Loading checkpoint...")
            checkpoint = torch.load(self.ckpt, map_location=torch.device("cpu"))["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=False)

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.l1loss = nn.L1Loss(reduction="mean")
        self.hmloss = HistogramMatchingLoss()
        self.piloss = PerceptualLoss(
            spatial_dims=2, 
            network_type="radimagenet_resnet50", 
            is_fake_3d=False, 
            pretrained=True
        )
        self.tval = TotalVariation()
        self.psnr = PeakSignalNoiseRatio(data_range=(0, 1))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(0, 1))
        self.psnr_outputs = []
        self.ssim_outputs = []
    
    def forward_screen(self, image3d, cameras, opacity=None, norm_type="standardized", scale=0.1):
        screen = self.fwd_renderer(image3d, cameras, opacity, norm_type, scale) 
        return screen

    def forward_volume(self, image2d, cameras, timesteps=None):
        return self.inv_renderer(image2d, cameras)        
    
    def forward_opaque(self, image3d):
        return self.inv_alpharer(image3d)
        
    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = "evaluation"):
        image3d = batch["image3d"]
        image2d = batch["image2d"]
        _device = batch["image3d"].device
        batchsz = image2d.shape[0]

        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_random = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_random = torch.rand_like(dist_random) - 0.5
        azim_random = torch.rand_like(dist_random) * 2 - 1  # [0 1) to [-1 1)
        view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=18, znear=4, zfar=8)

        dist_hidden = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_hidden = torch.zeros(self.batch_size, device=_device)
        azim_hidden = torch.zeros(self.batch_size, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=18, znear=4, zfar=8)
        
        # From XR -> image3d -> opacity -> XR
        figure_xr_hidden = image2d
        volume_ct = image3d
        opaque_ct = self.forward_opaque(volume_ct)
        figure_ct = self.forward_screen(torch.cat([volume_ct, volume_ct]),
                                        join_cameras_as_batch([view_hidden, view_random]),
                                        torch.cat([opaque_ct, opaque_ct]), scale=0.1)
        
        figure_ct_hidden, figure_ct_random = torch.split(figure_ct, batchsz)
        figure_dx = torch.cat([figure_xr_hidden, figure_ct_hidden, figure_ct_random])
        
        volume_dx_second, \
        middle_dx_second = self.forward_volume(figure_dx, join_cameras_as_batch([view_hidden, view_hidden, view_random]))
        opaque_dx_second = self.forward_opaque(volume_dx_second)
        
        opaque_xr_hidden_second, opaque_ct_hidden_second, opaque_ct_random_second = torch.split(opaque_dx_second, batchsz)
        volume_xr_hidden_second, volume_ct_hidden_second, volume_ct_random_second = torch.split(volume_dx_second, batchsz)
        middle_xr_hidden_second, middle_ct_hidden_second, middle_ct_random_second = torch.split(middle_dx_second, batchsz)
        
        figure_dx_second = self.forward_screen(torch.cat([volume_xr_hidden_second,
                                                          volume_xr_hidden_second]), 
                                               join_cameras_as_batch([view_hidden, 
                                                                      view_random]),
                                               torch.cat([opaque_xr_hidden_second, 
                                                          opaque_xr_hidden_second]), 
                                               scale=0.1)
        figure_xr_hidden_second_hidden, \
        figure_xr_hidden_second_random = torch.split(figure_dx_second, batchsz)
        
        im2d_loss = self.l1loss(figure_xr_hidden, figure_xr_hidden_second_hidden) 
                      
        im3d_loss = self.l1loss(volume_ct, volume_ct_hidden_second) + self.l1loss(volume_ct, middle_ct_hidden_second) \
                  + self.l1loss(volume_ct, volume_ct_random_second) + self.l1loss(volume_ct, middle_ct_random_second) \
                #   + self.l1loss(opaque_ct, torch.ones_like(opaque_ct)) \
                #   + self.l1loss(opaque_dx_second, torch.ones_like(opaque_dx_second))
                  
        hist_loss = self.hmloss(figure_xr_hidden, figure_ct_hidden) \
                  + self.hmloss(figure_ct_random, figure_xr_hidden_second_random) 
                  
        perc_loss = self.piloss(figure_xr_hidden, figure_ct_hidden) \
                  + self.piloss(figure_ct_random, figure_xr_hidden_second_random) 
                                    
        # Visualization step
        if batch_idx == 0:
            zeros = torch.zeros_like(image2d)
            viz2d = torch.cat([
                torch.cat([
                    figure_xr_hidden, 
                    volume_xr_hidden_second[..., self.vol_shape // 2, :], 
                    opaque_xr_hidden_second[..., self.vol_shape // 2, :], 
                    figure_xr_hidden_second_hidden,
                    figure_xr_hidden_second_random,
                    zeros,
                ], dim=-2,).transpose(2, 3),
                torch.cat([
                    volume_ct[..., self.vol_shape // 2, :], 
                    opaque_ct[..., self.vol_shape // 2, :], 
                    figure_ct_hidden,
                    figure_ct_random,
                    volume_ct_hidden_second[..., self.vol_shape // 2, :], 
                    volume_ct_random_second[..., self.vol_shape // 2, :], 
                ], dim=-2,).transpose(2, 3),
            ], dim=-2,)
            tensorboard = self.logger.experiment
            grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0).clamp(0.0, 1.0)  
            tensorboard.add_image(f"{stage}_df_samples", grid2d, self.current_epoch * self.batch_size + batch_idx,)
                      
        # Log the final losses
        self.log(f"{stage}_im2d_loss", im2d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size,)
        self.log(f"{stage}_im3d_loss", im3d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size,)
        self.log(f"{stage}_hist_loss", hist_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size,)
        self.log(f"{stage}_perc_loss", perc_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size,)
        
        loss = self.alpha * im3d_loss + self.gamma * im2d_loss + self.theta * hist_loss 
        if stage=='train':
            loss += self.lamda * perc_loss
        return loss
    
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self._common_step(batch, batch_idx, optimizer_idx, stage="train")
        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, optimizer_idx=-1, stage="validation")
        self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        image3d = batch["image3d"]
        image2d = batch["image2d"]
        _device = batch["image3d"].device
        batchsz = image2d.shape[0]

        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_hidden = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_hidden = torch.zeros(self.batch_size, device=_device)
        azim_hidden = torch.zeros(self.batch_size, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=18, znear=4, zfar=8)

        # Construct the samples in 2D
        # figure_ct_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)
        # figure_xr_hidden = image2d

        # secondstruct the Encoder-Decoder
        # volume_ct_hidden = self.forward_volume(image2d=figure_ct_hidden)
        
        psnr = self.psnr(volume_ct_hidden, image3d)
        ssim = self.ssim(volume_ct_hidden, image3d)
        self.psnr_outputs.append(psnr)
        self.ssim_outputs.append(ssim)
        
    def on_train_epoch_end(self):
        loss = torch.stack(self.train_step_outputs).mean()
        self.log(f"train_loss_epoch", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True,)
        self.train_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log(f"validation_loss_epoch", loss, on_step=False, prog_bar=True, logger=True, sync_dist=True,)
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):
        print(f"PSNR :{torch.stack(self.psnr_outputs).mean()}")
        print(f"SSIM :{torch.stack(self.ssim_outputs).mean()}")
        self.psnr_outputs.clear()
        self.ssim_outputs.clear()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.inv_renderer.parameters()},
                {'params': self.inv_alpharer.parameters()}, # Add 
            ],
            lr=self.lr,
            betas=(0.5, 0.999)
            # self.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--conda_env", type=str, default="Unet")
    parser.add_argument("--devices", default=None)

    # Model arguments
    parser.add_argument("--sh", type=int, default=0, help="degree of spherical harmonic (2, 3)")
    parser.add_argument("--pe", type=int, default=0, help="positional encoding (0 - 8)")
    parser.add_argument("--n_pts_per_ray", type=int, default=400, help="Sampling points per ray")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--vol_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--fov_depth", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--epochs", type=int, default=301, help="number of epochs")
    parser.add_argument("--train_samples", type=int, default=1000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=400, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")
    parser.add_argument("--timesteps", type=int, default=180, help="timesteps for diffusion")
    parser.add_argument("--amp", action="store_true", help="train with mixed precision or not")
    parser.add_argument("--test", action="store_true", help="train with mixed precision or not")
    parser.add_argument("--gan", action="store_true", help="train with gan xray ct random")
    parser.add_argument("--strict", action="store_true", help="checkpoint loading")

    parser.add_argument("--alpha", type=float, default=1.0, help="vol loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="img loss")
    parser.add_argument("--delta", type=float, default=1.0, help="vgg loss")
    parser.add_argument("--theta", type=float, default=1.0, help="cam loss")
    parser.add_argument("--omega", type=float, default=1.0, help="cam cond")
    parser.add_argument("--lamda", type=float, default=0.1, help="perc loss")

    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logsdir", type=str, default="logs", help="logging directory")
    parser.add_argument("--datadir", type=str, default="data", help="data directory")
    parser.add_argument("--strategy", type=str, default="ddp_find_unused_parameters_true", help="training strategy")
    parser.add_argument("--backbone", type=str, default="vgg-19", help="Backbone for network")
    parser.add_argument("--prediction_type", type=str, default="sample", help="prediction_type for network",)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # parser = Trainer.add_argparse_args(parser)

    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.logsdir}",
        # filename='epoch={epoch}-validation_loss={validation_loss_epoch:.2f}',
        monitor="validation_loss_epoch",
        auto_insert_metric_name=True,
        save_top_k=-1,
        save_last=True,
        every_n_epochs=10,
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=f"{hparams.logsdir}", log_graph=True)
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    callbacks = [
        lr_callback,
        checkpoint_callback,
    ]
    if hparams.strategy != "gan":
        callbacks.append(swa_callback)
    # Init model with callbacks
    trainer = Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=callbacks,
        accumulate_grad_batches=4,
        strategy=hparams.strategy,  # "auto", #"ddp_find_unused_parameters_true",
        precision=16 if hparams.amp else 32,
        profiler="advanced",
    )

    # Create data module
    train_image3d_folders = [
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/NSCLC/processed/train/images"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0",),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1",),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2",),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3",),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4",),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/train/images"),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/val/images"),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),
    ]

    train_label3d_folders = []

    train_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/train/images/"),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),
    ]

    train_label2d_folders = []

    val_image3d_folders = [
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/NSCLC/processed/train/images"),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0",),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1",),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2",),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3",),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4",),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/train/images"),
        # os.path.join(hparams.datadir, "ChestXRLungSegmentation/MELA2022/raw/val/images"),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/TCIA/CT-Covid-19-2020/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/TCIA/CT-Covid-19-2021/images'),
    ]

    val_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        os.path.join(hparams.datadir, "ChestXRLungSegmentation/VinDr/v1/processed/test/images/"),
    ]

    test_image3d_folders = val_image3d_folders
    test_image2d_folders = val_image2d_folders

    datamodule = UnpairedDataModule(
        train_image3d_folders=train_image3d_folders,
        train_image2d_folders=train_image2d_folders,
        val_image3d_folders=val_image3d_folders,
        val_image2d_folders=val_image2d_folders,
        test_image3d_folders=test_image3d_folders,
        test_image2d_folders=test_image2d_folders,
        train_samples=hparams.train_samples,
        val_samples=hparams.val_samples,
        test_samples=hparams.test_samples,
        batch_size=hparams.batch_size,
        img_shape=hparams.img_shape,
        vol_shape=hparams.vol_shape,
    )
    datamodule.setup()

    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = DXRLightningModule(hparams=hparams)
    
    if hparams.test:
        trainer.test(
            model,
            dataloaders=datamodule.test_dataloader(),
            ckpt_path=hparams.ckpt
        )

    else:
        trainer.fit(
            model,
            # compiled_model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            # datamodule=datamodule,
            ckpt_path=hparams.ckpt if hparams.ckpt is not None and hparams.strict else None,  # "some/path/to/my_checkpoint.ckpt"
        )

    # serve
