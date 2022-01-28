"""
Sample training loop.
"""
import math
import logging
from tqdm import tqdm
from dataclasses import asdict
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from dl_schema.utils import flatten

import mlflow

from torch.utils.tensorboard import SummaryWriter
import torchvision

from ray import tune

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self, model, cfg, train_dataset, test_dataset=None, tune=False, verbose=True
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.cfg = cfg
        self.verbose = verbose
        self.tune = tune

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        # Configure optimizer
        # - DataParallel wrappers keep raw model object in .module attribute
        self.raw_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        self.optimizer = self.raw_model.configure_optimizers(self.cfg)

    def save_checkpoint(self):
        logger.info(f"saving {self.cfg.ckpt_path}")
        torch.save(self.raw_model.state_dict(), self.cfg.ckpt_path)

    def train_epoch(self, epoch):
        """ Returns: train_epoch_loss """
        model, cfg, optim = self.model, self.cfg, self.optimizer
        model.train()
        loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=cfg.bs,
            num_workers=cfg.num_workers,
        )
        pbar = tqdm(enumerate(loader), total=len(loader), disable=not (self.verbose))
        losses = []
        for it, (x, y) in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.set_grad_enabled(True):
                linear_units, loss = model(x, y)
                # collapse all losses if they are scattered on multiple gpus
                loss = loss.mean()
                losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            optim.step()

            # report progress
            pbar.set_description(
                f"(train) epoch {epoch+1} iter {it}: train loss {loss.item():.5f} "
                + f"lr {cfg.lr:.2e}"
            )

            # log train batch loss and train batch image grid
            mlflow.log_metric("train_batch_loss", loss.item(), it)
            grid = torchvision.utils.make_grid(x.cpu()).permute(1, 2, 0)
            mlflow.log_image(grid.numpy(), "latest_train_batch.png")

        train_epoch_loss = float(np.mean(losses))
        logger.info(f"train loss: {train_epoch_loss}")
        return train_epoch_loss

    def test_epoch(self, epoch):
        """ Returns: test_epoch_loss """
        model, cfg = self.model, self.cfg
        model.eval()
        # Form dataloader. Uses bs and workers from train cfg.
        loader = DataLoader(
            self.test_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=cfg.bs,
            num_workers=cfg.num_workers,
        )
        pbar = tqdm(enumerate(loader), total=len(loader), disable=not (self.verbose))
        losses = []
        for it, (x, y) in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.set_grad_enabled(False):
                linear_units, loss = model(x, y)
                # collapse all losses if they are scattered on multiple gpus
                loss = loss.mean()
                losses.append(loss.item())

            # report progress
            pbar.set_description(
                f"(test) epoch {epoch+1} iter {it}: test loss {loss.item():.5f}"
            )

            # log test batch loss and test batch image grid
            mlflow.log_metric("test_batch_loss", loss.item(), it)
            grid = torchvision.utils.make_grid(x.cpu()).permute(1, 2, 0)
            mlflow.log_image(grid.numpy(), "latest_test_batch.png")

        test_epoch_loss = float(np.mean(losses))
        logger.info(f"test loss: {test_epoch_loss}")
        return test_epoch_loss

    def train(self):
        """ Returns: best_epoch_loss """
        cfg = self.cfg
        best_epoch_loss = float("inf")
        for epoch in range(cfg.epochs):
            train_epoch_loss = self.train_epoch(epoch)
            mlflow.log_metric("train_epoch_loss", train_epoch_loss, epoch + 1)

            # Log best loss and support early stop checkpoints
            if self.test_dataset is not None:
                test_epoch_loss = self.test_epoch(epoch)
                mlflow.log_metric("test_epoch_loss", test_epoch_loss, epoch + 1)
                if self.tune:
                    tune.report(loss=test_epoch_loss, epoch=epoch + 1)
                if test_epoch_loss < best_epoch_loss:
                    best_epoch_loss = test_epoch_loss
                    mlflow.log_metric(
                        "best_test_epoch_loss", best_epoch_loss, epoch + 1
                    )
                    if cfg.early_stop:
                        mlflow.pytorch.log_model(self.model, "best_model")
        return best_epoch_loss
