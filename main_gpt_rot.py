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

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from generative.losses import PerceptualLoss

from argparse import ArgumentParser

from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    look_at_view_transform,
)

from datamodule import UnpairedDataModule
from dvr.renderer import DirectVolumeFrontToBackRenderer

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
    "efficientnet-b9": (32, 64, 96, 256, 800),
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

        self.unet2d_model = DiffusionModelUNet(
            spatial_dims=2, 
            in_channels=2, 
            out_channels=1, 
            num_channels=backbones[self.backbone], 
            attention_levels=[False, False, False, True, True], 
            norm_num_groups=16, 
            num_res_blocks=2, 
            with_conditioning=True, 
            cross_attention_dim=24, # Condition with straight/hidden view  # flatR | flatT
        )

        self.ddpmsch = DDPMScheduler(
            num_train_timesteps=self.timesteps, 
            schedule="scaled_linear_beta", 
            prediction_type=hparams.prediction_type, 
            beta_start=0.0005, 
            beta_end=0.0195,
        )
        self.ddimsch = DDIMScheduler(
            num_train_timesteps=self.timesteps, 
            schedule="scaled_linear_beta", 
            prediction_type=hparams.prediction_type, 
            beta_start=0.0005, 
            beta_end=0.0195, 
            clip_sample=True,
        )
        self.ddimsch.set_timesteps(num_inference_steps=100)
        self.inferer = DiffusionInferer(scheduler=self.ddimsch)

        if self.ckpt:
            checkpoint = torch.load(self.ckpt, map_location=torch.device("cpu"))["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=self.strict)

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.l1loss = nn.L1Loss(reduction="mean")
            
        self.psnr = PeakSignalNoiseRatio(data_range=(0, 1))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(0, 1))
        self.psnr_outputs = []
        self.ssim_outputs = []
    
    def forward_screen(self, image3d, cameras):
        screen = self.fwd_renderer(image3d * 0.5 + 0.5, cameras) * 2.0 - 1.0
        return screen
    
    def forward_timing(self, srcimg=None, tgtimg=None, srccam=None, tgtcam=None, timesteps=None):
        _device = tgtimg.device
        assert (srcimg is not None) and (srccam is not None) and (tgtcam is not None)
        B = tgtimg.shape[0]
        if timesteps is None:
            timesteps = torch.zeros((B,), device=_device).long()
            
        context = torch.cat([
            srccam.R.reshape(B, 1, -1), srccam.T.reshape(B, 1, -1),
            tgtcam.R.reshape(B, 1, -1), tgtcam.T.reshape(B, 1, -1),
        ], dim=-1,)
        combine = torch.cat([srcimg, tgtimg], dim=1)
        results = self.unet2d_model(x=combine, context=context, timesteps=timesteps,)
        return results        

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = "evaluation"):
        image3d = batch["image3d"] * 2.0 - 1.0
        image2d = batch["image2d"] * 2.0 - 1.0
        _device = batch["image3d"].device
        batchsz = image2d.shape[0]
        
        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_0 = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_0 = 0.0 * torch.ones(self.batch_size, device=_device)
        azim_0 = 0.0 * torch.ones(self.batch_size, device=_device)
        view_0 = make_cameras_dea(dist_0, elev_0, azim_0, fov=18, znear=4, zfar=8)
        
        dist_1 = 6.0 * torch.ones(self.batch_size, device=_device)
        elev_1 = 1.0 * torch.rand(self.batch_size, device=_device) - 0.5
        azim_1 = 1.0 * torch.rand(self.batch_size, device=_device) * 2 - 1  # [0 1) to [-1 1)
        view_1 = make_cameras_dea(dist_1, elev_1, azim_1, fov=18, znear=4, zfar=8)

        # dist_2 = 6.0 * torch.ones(self.batch_size, device=_device)
        # elev_2 = 1.0 * torch.rand(self.batch_size, device=_device) - 0.5
        # azim_2 = 1.0 * torch.rand(self.batch_size, device=_device) * 2 - 1  # [0 1) to [-1 1)
        # view_2 = make_cameras_dea(dist_2, elev_2, azim_2, fov=18, znear=4, zfar=8)

        # dist_3 = 6.0 * torch.ones(self.batch_size, device=_device)
        # elev_3 = 1.0 * torch.rand(self.batch_size, device=_device) - 0.5
        # azim_3 = 1.0 * torch.rand(self.batch_size, device=_device) * 2 - 1  # [0 1) to [-1 1)
        # view_3 = make_cameras_dea(dist_3, elev_3, azim_3, fov=18, znear=4, zfar=8)

        
        # Diffusion step: 2 kinds of blending
        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (batchsz,), device=_device).long()  # 3 views

        # Construct the samples in 2D
        figure_xr_0 = image2d
        # figure_xr_1 = image2d
        figure_ct_0 = self.forward_screen(image3d=image3d, cameras=view_0)
        figure_ct_1 = self.forward_screen(image3d=image3d, cameras=view_1)

        figure_xr_latent_0 = torch.randn(image2d.shape, device=_device)
        # figure_xr_latent_1 = torch.randn(image2d.shape, device=_device)
        figure_ct_latent_0 = torch.randn(image2d.shape, device=_device)
        figure_ct_latent_1 = torch.randn(image2d.shape, device=_device)
        
        figure_xr_interp_0 = self.ddpmsch.add_noise(original_samples=figure_xr_0, noise=figure_xr_latent_0, timesteps=timesteps)
        # figure_xr_interp_1 = self.ddpmsch.add_noise(original_samples=figure_xr_1, noise=figure_xr_latent_1, timesteps=timesteps)
        figure_ct_interp_0 = self.ddpmsch.add_noise(original_samples=figure_ct_0, noise=figure_ct_latent_0, timesteps=timesteps)
        figure_ct_interp_1 = self.ddpmsch.add_noise(original_samples=figure_ct_1, noise=figure_ct_latent_1, timesteps=timesteps)

        # Run the backward diffusion (denoising + reproject)
        figure_dx_output = self.forward_timing(
            srcimg=torch.cat([figure_xr_0, 
                              figure_ct_0, 
                              figure_ct_0, 
                              figure_ct_1, 
                              figure_ct_1]), 
            tgtimg=torch.cat([figure_xr_interp_0, 
                              figure_ct_interp_0, 
                              figure_ct_interp_1, 
                              figure_ct_interp_0, 
                              figure_ct_interp_1]), 
            srccam=join_cameras_as_batch([view_0, view_0, view_0, view_1, view_1]),
            tgtcam=join_cameras_as_batch([view_0, view_0, view_1, view_0, view_1]),
            timesteps=timesteps.repeat(5),
        )
        (
            figure_xr_output_00, 
            figure_ct_output_00, 
            figure_ct_output_01,
            figure_ct_output_10, 
            figure_ct_output_11,
        ) = torch.split(figure_dx_output, batchsz)

        if self.ddpmsch.prediction_type == "sample":
            figure_xr_target_0 = figure_xr_0
            figure_ct_target_0 = figure_ct_0 # swap
            figure_ct_target_1 = figure_ct_1
        elif self.ddpmsch.prediction_type == "epsilon":
            figure_xr_target_0 = figure_xr_latent_0
            figure_ct_target_0 = figure_ct_latent_0
            figure_ct_target_1 = figure_ct_latent_1
        elif self.ddpmsch.prediction_type == "v_prediction":
            figure_xr_target_0 = self.ddpmsch.get_velocity(figure_xr_0, figure_xr_latent_0, timesteps)
            figure_ct_target_0 = self.ddpmsch.get_velocity(figure_ct_0, figure_ct_latent_0, timesteps)
            figure_ct_target_1 = self.ddpmsch.get_velocity(figure_ct_1, figure_ct_latent_1, timesteps)
        
        im2d_loss = self.l1loss(figure_xr_output_00, figure_xr_target_0) \
                  + self.l1loss(figure_ct_output_00, figure_ct_target_0) \
                  + self.l1loss(figure_ct_output_01, figure_ct_target_1) \
                  + self.l1loss(figure_ct_output_10, figure_ct_target_0) \
                  + self.l1loss(figure_ct_output_11, figure_ct_target_1) \
     
        # Log the final losses
        self.log(f"{stage}_im2d_loss", im2d_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size,)
        
        # Visualization step
        if batch_idx == 0:
            # Sampling step for X-ray
            with torch.no_grad():
                scheduler = self.ddimsch
                figure_xr_1 = torch.randn_like(image2d)
                pbar = iter(scheduler.timesteps)
                for t in pbar:
                    # 1. predict noise model_output
                    model_output = self.forward_timing(
                        # torch.cat([figure_xr_1, figure_xr_hidden], dim=1), 
                        # timesteps=torch.Tensor((t,)).to(_device), context=pose_sample
                        srcimg=figure_xr_0, 
                        tgtimg=figure_xr_1, 
                        srccam=view_0,
                        tgtcam=view_1,
                        timesteps=torch.Tensor((t,)).to(_device)
                    )

                    # 2. compute previous image: x_t -> x_t-1
                    figure_xr_1, _ = scheduler.step(model_output, t, figure_xr_1)


            zeros = torch.zeros_like(image2d)
            viz2d = torch.cat([
                torch.cat([
                    figure_xr_0, 
                    figure_xr_1,
                    figure_ct_interp_0,
                    figure_ct_interp_1,
                    figure_ct_target_0,
                    figure_ct_target_1,
                ], dim=-2,).transpose(2, 3),
                torch.cat([
                    figure_ct_0, 
                    figure_ct_1,
                    figure_ct_output_00,
                    figure_ct_output_01,
                    figure_ct_output_10,
                    figure_ct_output_11,
                ], dim=-2,).transpose(2, 3),
            ], dim=-2,)
            tensorboard = self.logger.experiment
            grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0).clamp(-1.0, 1.0) * 0.5 + 0.5
            tensorboard.add_image(f"{stage}_df_samples", grid2d, self.current_epoch * self.batch_size + batch_idx)

        loss = self.gamma * im2d_loss
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
        pass
        
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
                # {"params": self.inv_renderer.parameters()},
                {'params': self.unet2d_model.parameters()}, # Add diffusion model, remove lpips model
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
