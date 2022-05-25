"""
Training executer - handles lr schedulers, optimizers, model saving/loading, 
datasets/generators, train steps, test steps, metrics, losses, etc
"""
import logging
import math
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
from tqdm import tqdm

from dl_schema.utils import configure_adamw

logger = logging.getLogger(__name__)


class Trainer:
    """train or evaluate a dataset over n epochs"""

    def __init__(self, model, cfg, train_dataset, test_dataset=None, recorder=None, verbose=True):
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.recorder = recorder
        self.test_only = self.train_dataset is None
        self.verbose = verbose
        self.curr_epoch = 0
        self.scheduler = None

        # set mlflow paths for model/optim saving
        if recorder is not None:
            self.ckpt_root = self.recorder.root / "checkpoints"
            (self.ckpt_root).mkdir(parents=True, exist_ok=True)
        else:
            self.ckpt_root = ""

        # set gpu device if available
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

        # set datloaders
        self.train_loader = self.create_dataloader(train=True)
        self.test_loader = self.create_dataloader(train=False)

        # configure optimizer
        if self.test_only:
            self.optimizer = None
            self.cfg.load_optimizer = False
        else:
            self.optimizer = configure_adamw(self.model, self.cfg)
            self.set_scheduler(steps=len(self.train_loader))

        # initialize best loss for ckpt saving
        self.best_epoch_loss = float("inf")

    def create_dataloader(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        if dataset is None:
            return None
        loader = DataLoader(
            dataset,
            shuffle=self.cfg.data.shuffle,
            pin_memory=True,
            batch_size=self.cfg.bs,
            num_workers=self.cfg.num_workers,
            drop_last=True,
        )
        return loader

    def set_scheduler(self, steps):
        """create lr scheduler; steps argument required for onecycle"""
        if self.cfg.lr_method.name == "onecycle":
            self.scheduler = self.cfg.lr_method(
                self.optimizer,
                self.cfg.lr,
                steps_per_epoch=steps,
                epochs=self.cfg.epochs - self.curr_epoch,
                div_factor=self.cfg.onecycle_div_factor,
                final_div_factor=self.cfg.onecycle_final_div_factor,
            )
        else:
            self.scheduler = self.cfg.lr_method(self.optimizer, lr_lambda=lambda epoch: 1)

    def save_model(self, path="last.pt", loss=None, as_artifact=True):
        """save model state dict, optim state dict, epoch and loss"""
        save_path = self.ckpt_root / path if as_artifact else path
        if self.verbose:
            logger.info(f"saving {save_path}")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": self.curr_epoch,
                "loss": loss,
            },
            save_path,
        )

    def load_model(self):
        """load model state dict, optim state dict, epoch and loss"""
        ckpt_path = Path(self.cfg.load_ckpt_pth).expanduser().resolve()
        ckpt = torch.load(ckpt_path)

        # load optimizer
        if self.cfg.load_optimizer:
            logger.info(f"loading optimizer from {ckpt_path}")
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_epoch_loss = ckpt["loss"]

        # only update scheduler and epoch counter if resuming
        if self.cfg.resume:
            logger.info(f"resuming from epoch: {ckpt['epoch']}")
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.curr_epoch = ckpt["epoch"] + 1

        # load parameters
        logger.info(f"loading model params from {ckpt_path}")
        self.model.load_state_dict(ckpt["model_state_dict"])

    def run_epoch(self, split: Literal["train", "test"] = "train"):
        """train or evalauate on a single epoch, returning mean epoch loss"""
        assert split in {"train", "test"}
        is_train = split == "train"
        epoch = self.curr_epoch
        self.model.train() if is_train else self.model.eval()
        loader = self.train_loader if is_train else self.test_loader

        # initialize running lists of quantities to be logged
        losses, metric1s = [], []

        # train/test loop
        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (x, y) in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            # forward the model, calculate loss
            with torch.set_grad_enabled(is_train):
                y_pred = self.model(x)
                loss = self.cfg.loss(y_pred, y)
                losses.append(loss.item())

            # get current learning rate before optim step
            curr_lr = self.optimizer.param_groups[0]["lr"]

            # backward step
            if is_train:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # calculate relevant metrics
            y_pred_digit = y_pred.argmax(dim=1, keepdim=True)
            metric1 = self.cfg.metric1(y_pred_digit, y)

            # append losses and metrics to running lists
            losses.append(loss.item())
            metric1s.append(metric1.item())

            # report progress bar
            pbar.set_description(
                f"({split}) epoch {epoch} iter {it}: {split} loss {loss.item():.6e} "
                + f"lr {curr_lr:.2e}"
            )

            # log batch quantities
            step = it + epoch * len(loader)
            if step % self.cfg.log.batch_freq == 0:
                suffix = f"_{split}_batch"
                if is_train:
                    self.recorder.log_metric("lr" + suffix, curr_lr, step)
                self.recorder.log_metric("loss" + suffix, loss.item(), step)
                self.recorder.log_metric(self.cfg.metric1.name + suffix, metric1.item(), step)
                if self.cfg.log.images:
                    self.recorder.log_image_grid(x.detach().cpu(), name=f"digits{suffix}")

                    # log grid of batch images
                    #n_rows = math.ceil(math.sqrt(self.cfg.bs))  # actually n_cols
                    #grid = torchvision.utils.make_grid(
                    #    x.cpu(), normalize=True, nrow=n_rows
                    #).permute(1, 2, 0)
                    #mlflow.log_image(grid.numpy(), f"digits{suffix}.png")

            # stop training early based on steps
            if self.cfg.steps is not None and step >= self.cfg.steps:
                break

        # log epoch end mean quantities
        loss_epoch = float(np.mean(losses))
        suffix = f"_{split}_epoch"
        mlflow.log_metric("loss" + suffix, loss_epoch, step=epoch)
        mlflow.log_metric(
            self.cfg.metric1.name + suffix, float(np.mean(metric1s)), step=epoch
        )
        return loss_epoch

    def train(self):
        """train or evaluate over a number of epochs, returning best_epoch_loss"""
        cfg = self.cfg
        for epoch in range(self.curr_epoch, cfg.epochs):
            self.curr_epoch = epoch

            # train if dataset provided
            if self.train_dataset is not None:
                self.run_epoch("train")

            # evaluate on test dataset
            if self.test_dataset is not None:
                test_epoch_loss = self.run_epoch("test")

                # update best loss, possibly save best model state
                if test_epoch_loss < self.best_epoch_loss:
                    self.best_epoch_loss = test_epoch_loss
                    mlflow.log_metric(
                        "loss_test-best_epoch", self.best_epoch_loss, epoch
                    )
                    if cfg.save_best:
                        self.save_model("best.pt", loss=self.best_epoch_loss)

                # save latest model
                if cfg.save_last:
                    self.save_model("last.pt", loss=test_epoch_loss)

        return self.best_epoch_loss
