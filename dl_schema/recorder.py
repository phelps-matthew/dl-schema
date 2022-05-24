"""
Logging utility for training runs

https://github.com/microsoft/qlib/blob/d7d19feb4ebb0c4318ac3bfda32a34c56e28a6a0/qlib/workflow/recorder.py#L368
"""
import math
import logging
import json
from pathlib import Path
import numpy as np
import mlflow
import torchvision
import torch
import torch.nn.functional as F
import cv2
from matplotlib import pyplot as plt
from spec21.keypoint_regression.models.pnp import Pnp
import networkx as nx

from spec21.utils import l2, FriendlyEncoder, load_yaml
from spec21.triangulation.triangulate import project_3d_to_2d
from spec21.recorder_base import RecorderBase, AsyncCaller
from spec21.speedplus import SpeedPlus
from spec21.keypoint_regression.plot_vf import vf_overlay_batch, vf_overlay_single, vf_heatmap

logger = logging.getLogger(__name__)


class Recorder(RecorderBase):
    """artifact logger for spec"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.pnp = Pnp()
        self.speed = SpeedPlus()

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_iter(self, inputs, step=None, batch=True, split="train"):
        """log batch artifacts
        Args:
            inputs: dictionary of torch inputs from run_epoch
            step: iteration step
            batch: batch iteration or epoch iteration
            split: train, test, or infer split
        """
        iter_name = "batch" if batch else "epoch"
        iter_cfg = getattr(self.cfg.log, iter_name)
        # suffix for artifact filenames
        suffix = f"_{split}_{iter_name}"

        if not getattr(iter_cfg, f"{split}"):
            return

        # float
        loss = inputs.get("loss", None)
        # float
        coord_loss = inputs.get("coord_loss", None)
        # float
        div_loss = inputs.get("div_loss", None)
        # float
        vf_loss = inputs.get("vf_loss", None)
        # float
        coord_loss = inputs.get("coord_loss", None)
        # float
        rot_loss = inputs.get("rot_loss", None)
        # float
        trans_loss = inputs.get("trans_loss", None)
        # float
        rot_err = inputs.get("rot_err", None)
        # float
        trans_err = inputs.get("trans_err", None)
        # float
        score = inputs.get("score", None)
        # (N, 1, H, W), torch.float32
        x_img = inputs.get("x_img", None)
        # (N, 12, 2, H/down, W/down), torch.float32
        kvf = inputs.get("kvf", None)
        # (N, 12, 2, H/down, W/down), torch.float32
        kvf_pred = inputs.get("kvf_pred", None)
        # float
        lr = inputs.get("lr", None)
        # dict (torch.float32). bbox = (N, 1, 4) normalized [0, 1]
        annot = inputs.get("annot", None)
        # (N, 4) list
        q_pred = inputs.get("q_pred", None)
        # (N, 3) list
        trans_pred = inputs.get("trans_pred", None)

        # log lr if exists
        if lr is not None:
            self.client.log_metric(self.run_id, "lr_" + iter_name, lr, step=step)

        if loss is not None:
            self.client.log_metric(self.run_id, "loss" + suffix, loss, step=step)

        if iter_cfg.coord_loss and coord_loss is not None:
            self.client.log_metric(
                self.run_id, "coord_loss" + suffix, coord_loss, step=step
            )

        if iter_cfg.divergence and div_loss is not None:
            self.client.log_metric(
                self.run_id, "div_loss" + suffix, div_loss, step=step
            )

        if iter_cfg.vf_loss and vf_loss is not None:
            self.client.log_metric(
                self.run_id, "vf_loss" + suffix, vf_loss, step=step
            )

        if rot_loss is not None:
            self.client.log_metric(
                self.run_id, "rot_loss" + suffix, rot_loss, step=step
            )

        if trans_loss is not None:
            self.client.log_metric(
                self.run_id, "trans_loss" + suffix, trans_loss, step=step
            )

        if iter_cfg.rot_err and rot_err is not None:
            name = "rot_err"
            self.client.log_metric(self.run_id, name + suffix, rot_err, step=step)

        if iter_cfg.trans_err and trans_err is not None:
            name = "trans_err"
            self.client.log_metric(self.run_id, name + suffix, trans_err, step=step)

        if iter_cfg.score and score is not None:
            name = "score"
            self.client.log_metric(self.run_id, name + suffix, score, step=step)

        if iter_cfg.annot and annot is not None:
            self._save_annot(annot, f"annot" + suffix + ".json")

        # log batch image grid
        if getattr(iter_cfg, f"{split}_imgs", False):
            self._save_batch_imgs(
                x_img,
                kvf,
                kvf_pred,
                iter_cfg,
                prefix=split,
                suffix=iter_name,
            )

        # log indivial images (inference batch only)
        if getattr(iter_cfg, f"{split}_single_imgs", False):
            self._save_single_imgs(
                x=x_img,
                kvf_pred=kvf_pred,
                annot=annot,
                r=trans_pred,
                q=q_pred,
                iter_cfg=iter_cfg,
                prefix=split,
                suffix=iter_name,
            )

    def _save_annot(self, annot, filename, batch=True):
        annot = annot.copy()
        # serialize to str using FriendlyEncoder
        annot_dict = json.dumps(annot, cls=FriendlyEncoder, indent=4)
        # deserialize back to dictionary
        annot_dict = json.loads(annot_dict)
        self.client.log_dict(self.run_id, annot_dict, "annotations/" + filename)

    def _save_batch_imgs(
        self,
        x,
        kvf,
        kvf_pred,
        iter_cfg,
        prefix="train",
        suffix="",
    ):
        """form batch image grid from inputs and log images"""
        n_rows = math.ceil(math.sqrt(self.cfg.bs))  # actually n_cols

        # log images. these are (N, C, H, W) torch.float32
        if x is not None:
            grid_x = torchvision.utils.make_grid(
                x, normalize=True, nrow=n_rows, pad_value=1.0, padding=2
            ).permute(1, 2, 0)
            self.client.log_image(
                self.run_id, grid_x.numpy(), f"{prefix}_x_{suffix}.jpg"
            )
        if iter_cfg.kvf_overlay and kvf is not None and x is not None:
            kvf_imgs = self.kvf_overlay(x, kvf, n=iter_cfg.kvf_samples)
            for i, img in enumerate(kvf_imgs):
                self.client.log_image(
                    self.run_id, img, f"{prefix}_kvf_{suffix}_{i:02d}.png")

        if iter_cfg.kvf_overlay and kvf_pred is not None and x is not None:
            kvf_imgs = self.kvf_overlay(x, kvf_pred, n=iter_cfg.kvf_samples)
            for i, img in enumerate(kvf_imgs):
                self.client.log_image(
                    self.run_id, img, f"{prefix}_kvf-pred_{suffix}_{i:02d}.png")

    def _save_single_imgs(
        self,
        x=None,
        kvf_pred=None,
        annot=None,
        q=None,
        r=None,
        iter_cfg=None,
        prefix="infer",
        suffix="",
    ):
        """save individual images as kvf overlays and wireframes"""

        hmap_pred = vf_heatmap(kvf_pred[:, :, 0, ...], kvf_pred[:, :, 1, ...])
        # torch.float32 of shape (N, 3, H, W)
        hmap_overlay = self._heatmap_overlay(hmap_pred, x)
        # upsample the heatmap overlay (N, 3, H * downsample, W * downsample)
        hmap_up = F.interpolate(
            hmap_overlay, x.shape[-2:], mode="bilinear", align_corners=False
        )

        # form plot panel
        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=((0.5 * 20, 0.5 * 7.5)),  # carefully adjusted
            gridspec_kw={
                "wspace": 0,
                "hspace": 0,
                "width_ratios": [2, 1, 1],
                "height_ratios": [1],
            },
        )

        # x is (N, 1, H, W) torch.float32
        if x is not None:
            batch_size = x.shape[0]
            for i in range(batch_size):
                filename = annot[i]["filename"]
                split = annot[i]["category"]
                if prefix == "infer":
                    if self.cfg.data.domain != split:
                        continue
                    speed_split = split
                else:
                    speed_split = "synthetic"
                idx, partition = self.speed.get_index_from_filename(
                    filename, speed_split
                )

                # form np arrays of x-input and normalized heatmap-pred overlay
                x_img = x[i].permute(1, 2, 0).numpy()
                hmap_img = hmap_up[i].permute(1, 2, 0).numpy()
                hmap_img = (hmap_img - hmap_img.min()) / (
                    hmap_img.max() - hmap_img.min()
                )

                # form wireframe axis
                wire_ax = self.speed.image_ax(idx, partition, ax=axes[0])
                wire_ax = self.speed.wireframe_ax(
                    i=None, partition=None, ax=wire_ax, q=q[i], r=r[i]
                )

                # plot the remaining image axes
                axes[1].imshow(x_img, cmap="gray")
                axes[2].imshow(hmap_img)

                # remove axes ticks and labels
                plt.setp(axes, xticks=[], yticks=[])
                # doesn't work perfectly.. matplotlib annoyance
                fig.tight_layout(pad=0, w_pad=0, h_pad=0)
                # render image to canvas
                fig.canvas.draw()

                # save matplotlib figure to np.ndarray
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                # reshape from (n) to (3, H, W)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # clear figure and axes for next image
                plt.cla()
                axes[0].clear()
                axes[1].clear()
                axes[2].clear()
                plt.close("all")

                # log image as artifact
                self.client.log_image(self.run_id, data, f"images/{split}_{filename}")

    def _heatmap_overlay(self, heatmap_pred, x):
        """overlay heatmap mask onto base image (x)
        Args:
            heatmap_pred: torch.float32 of shape (N, 12, H, W)
            x: torch.float32 of shape (N, 1, H * downsample, W * downsample)

        Returns:
            result: torch.float32 of shape (N, 3, H, W)
        """
        # blur = torchvision.transforms.GaussianBlur(kernel_size=(3, 3))
        # downsample input images (x), repeat channels to form rgb of shape (N, 3, H, W)
        x_down = F.interpolate(
            x, heatmap_pred.shape[2:], mode="bilinear", align_corners=False
        )
        x_rgb = x_down.repeat(1, 3, 1, 1)

        hmap = heatmap_pred.sum(axis=1, keepdims=True).div(self.cfg.data.num_keypoints)
        hmap = self.norm_batch(hmap)
        hmap = np.float32(hmap.squeeze(1))  # (N, H, W)
        # cv2 input (H, W), output (H, W, 3)
        cmap_fn = lambda x: cv2.applyColorMap(np.uint8(255 * x), cv2.COLORMAP_VIRIDIS)
        hmap_bgr = np.empty((*hmap.shape, 3), np.uint8)  # (N, H, W, 3)
        # apply colormap along batch axis
        # np.apply_along_axis doesn't appear to be any faster
        for i in range(hmap.shape[0]):
            hmap_bgr[i] = cmap_fn(hmap[i])

        # make compatible with x_rgb
        hmap_bgr = torch.from_numpy(hmap_bgr).permute(0, 3, 1, 2).float().div(255)
        hmap_rgb = hmap_bgr[:, [2, 1, 0], :, :]

        # result is unormalized torch.float32
        result = self.norm_batch(hmap_rgb) + self.norm_batch(x_rgb)
        return result

    def log_pose_stats(self, q_preds: list, trans_preds: list, categories: list):
        """plot translations inference statistics"""
        sun_trans = []
        lb_trans = []
        for trans, cat in zip(trans_preds, categories):
            if cat == "sunlamp":
                sun_trans.append(trans)
            else:
                lb_trans.append(trans)
        fig, axes = plt.subplots(
            nrows=2,
            ncols=3,
            figsize=((10, 8)),
        )
        axes[0, 0].hist([x[0] for x in sun_trans], bins=20)
        axes[0, 1].hist([x[1] for x in sun_trans], bins=20)
        axes[0, 2].hist([x[2] for x in sun_trans], bins=20)
        axes[1, 0].hist([x[0] for x in lb_trans], bins=20)
        axes[1, 1].hist([x[1] for x in lb_trans], bins=20)
        axes[1, 2].hist([x[2] for x in lb_trans], bins=20)

        axes[0, 0].set(xlabel="x [m]", ylabel="sunlamp")
        axes[0, 1].set(xlabel="y [m]", ylabel="sunlamp")
        axes[0, 2].set(xlabel="z [m]", ylabel="sunlamp")
        axes[1, 0].set(xlabel="x [m]", ylabel="lightbox")
        axes[1, 1].set(xlabel="y [m]", ylabel="lightbox")
        axes[1, 2].set(xlabel="z [m]", ylabel="lightbox")

        # fix overlapping labels
        plt.tight_layout()
        # render image to canvas
        fig.canvas.draw()

        # save matplotlib figure to np.ndarray
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # reshape from (n) to (3, H, W)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        self.client.log_image(self.run_id, data, f"translation_distribution.jpg")

    def kvf_overlay(self, x_img, kvf, n=1):
        """plot vector field overlayed on x_img"""
        x_img = self.norm_batch(x_img[:n])
        vfx = kvf[:n, :, 0, ...]
        vfy = kvf[:n, :, 1, ...]
        kvf_imgs = vf_overlay_batch(vfx, vfy, x_img)
        return kvf_imgs
