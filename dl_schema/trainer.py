"""
Sample training loop.
"""
import logging
from pathlib import Path
from dataclasses import asdict
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from dl_schema.utils import flatten, configure_adamw
import torchvision

import mlflow


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
        self.curr_epoch = 0
        self.scheduler = None

        # Set mlflow paths for model/optim saving
        self.mlflow_root = Path(mlflow.get_artifact_uri()[7:])  # cut file:/ uri
        self.ckpt_root = self.mlflow_root / "checkpoints"
        (self.ckpt_root).mkdir(parents=True, exist_ok=True)

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        # Configure optimizer
        # - DataParallel wrappers keep raw model object in .module attribute
        self.raw_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        self.optimizer = configure_adamw(self.raw_model, self.cfg)

    def set_scheduler(self, steps):
        """pass dataloader length into lr scheduler"""
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            self.cfg.lr,
            steps_per_epoch=steps,
            epochs=self.cfg.epochs,
        )

    def save_model(self, path="last.pt", loss=None):
        """save model state dict, optim state dict, epoch and loss"""
        logger.info(f"saving {self.ckpt_root / path}")
        torch.save(
            {
                "model_state_dict": self.raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.curr_epoch,
                "loss": loss,
            },
            self.ckpt_root / path,
        )

    def load_model(self, path="last.pt"):
        logger.info(f"loading {path}")
        ckpt = torch.load(path)
        self.raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.curr_epoch = ckpt["epoch"] + 1

    def train_epoch(self):
        """Returns: train_epoch_loss"""
        model, cfg, epoch = self.model, self.cfg, self.curr_epoch
        model.train()
        loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=cfg.bs,
            num_workers=cfg.num_workers,
        )

        # Initialize lr scheduler
        if self.scheduler is None:
            self.set_scheduler(steps=len(loader))

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

            params = {k: self.optimizer.param_groups[0][k] for k in ["lr"]}

            model.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # report progress
            pbar.set_description(
                f"(train) epoch {epoch} iter {it}: train loss {loss.item():.5f} "
                + f"lr {params['lr']:.2e}"
            )

            # log things we like
            step = it + epoch * len(loader)
            mlflow.log_metrics(params, step)
            mlflow.log_metric("train_batch_loss", loss.item(), step)
            grid = torchvision.utils.make_grid(x.cpu()).permute(1, 2, 0)
            mlflow.log_image(grid.numpy(), "latest_train_batch.png")

        train_epoch_loss = float(np.mean(losses))
        logger.info(f"train loss: {train_epoch_loss}")
        return train_epoch_loss

    def test_epoch(self):
        """Returns: test_epoch_loss"""
        model, cfg, epoch = self.model, self.cfg, self.curr_epoch
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
                f"(test) epoch {epoch} iter {it}: test loss {loss.item():.5f}"
            )

            # log test batch loss and test batch image grid
            step = it + epoch * len(loader)
            mlflow.log_metric("test_batch_loss", loss.item(), step)
            grid = torchvision.utils.make_grid(x.cpu()).permute(1, 2, 0)
            mlflow.log_image(grid.numpy(), "latest_test_batch.png")

        test_epoch_loss = float(np.mean(losses))
        logger.info(f"test loss: {test_epoch_loss}")
        return test_epoch_loss

    def train(self):
        """Returns: best_epoch_loss"""
        cfg = self.cfg
        best_epoch_loss = float("inf")
        for epoch in range(self.curr_epoch, cfg.epochs + self.curr_epoch):
            self.curr_epoch = epoch
            train_epoch_loss = self.train_epoch()
            mlflow.log_metric("train_epoch_loss", train_epoch_loss, epoch + 1)

            # Evaluate on test dataset
            if self.test_dataset is not None:
                test_epoch_loss = self.test_epoch()
                mlflow.log_metric("test_epoch_loss", test_epoch_loss, epoch + 1)

                # Update best loss, save best model state
                if test_epoch_loss < best_epoch_loss:
                    best_epoch_loss = test_epoch_loss
                    mlflow.log_metric(
                        "best_test_epoch_loss", best_epoch_loss, epoch + 1
                    )
                    if cfg.save_best:
                        self.save_model("best.pt", loss=best_epoch_loss)

                # Save latest model
                if cfg.save_last:
                    self.save_model("last.pt", loss=test_epoch_loss)

                if self.tune:
                    tune.report(loss=test_epoch_loss, epoch=epoch + 1)
        return best_epoch_loss
