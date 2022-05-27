"""
Training executer - handles lr schedulers, optimizers, model saving/loading, 
datasets/generators, train steps, test steps, metrics, losses, etc
"""
import logging
from pathlib import Path

import numpy as np
from ray import tune
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dl_schema.utils import configure_adamw

logger = logging.getLogger(__name__)


class Trainer:
    """train over n steps and evaluate over val/test set"""

    def __init__(
        self, model, cfg, train_dataset, test_dataset=None, recorder=None, verbose=True
    ):
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.recorder = recorder
        self.test_only = self.train_dataset is None
        self.verbose = verbose
        self.curr_step = 0
        self.scheduler = None
        self.total_steps = self.cfg.train_steps
        self.tune = self.cfg.tune
        self.tune_linked = False

        # set mlflow paths for model/optim saving
        if recorder is not None:
            self.ckpt_root = self.recorder.root / "checkpoints"
            (self.ckpt_root).mkdir(parents=True, exist_ok=True)
        else:
            self.ckpt_root = Path("./")

        # set gpu device(s) if available
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        # set dataloaders
        self.train_loader = self.create_dataloader(train=True)
        self.test_loader = self.create_dataloader(train=False)

        # configure optimizer
        if self.test_only:
            self.optimizer = None
            self.cfg.load_optimizer = False
        else:
            self.optimizer = configure_adamw(self.model, self.cfg)
            self.set_scheduler()

        # initialize best loss for ckpt saving
        self.best_loss = float("inf")

    def create_dataloader(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        shuffle = self.cfg.data.shuffle if train else False
        if dataset is None:
            return None
        loader = DataLoader(
            dataset,
            shuffle=shuffle,
            pin_memory=True,
            batch_size=self.cfg.bs,
            num_workers=self.cfg.num_workers,
            drop_last=True,
        )
        return loader

    def set_scheduler(self):
        """create learning rate scheduler"""
        if self.cfg.lr_method.name == "onecycle":
            self.scheduler = self.cfg.lr_method(
                self.optimizer,
                self.cfg.lr,
                total_steps=self.total_steps + 1,
                div_factor=self.cfg.onecycle_div_factor,
                final_div_factor=self.cfg.onecycle_final_div_factor,
            )
        elif self.cfg.lr_method.name in [
            "linear_warmup_cosine_decay",
            "linear_warmup_linear_decay",
        ]:
            self.scheduler = self.cfg.lr_method(
                self.optimizer,
                num_warmup_steps=self.cfg.warmup_steps,
                num_training_steps=self.total_steps + 1,
            )
        else:
            self.scheduler = self.cfg.lr_method(
                self.optimizer, lr_lambda=lambda step: 1
            )

    def save_model(self, path="last.pt", loss=None, as_artifact=True):
        """save model state dict, optim state dict, step and loss"""
        save_path = self.ckpt_root / path if as_artifact else path
        if self.verbose:
            logger.info(f"saving {save_path}")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "step": self.curr_step,
                "loss": loss,
            },
            save_path,
        )

    def load_model(self):
        """load model state dict, optim state dict, step and loss"""
        ckpt_path = Path(self.cfg.load_ckpt_pth).expanduser().resolve()
        ckpt = torch.load(ckpt_path)

        # load optimizer
        if self.cfg.load_optimizer:
            logger.info(f"loading optimizer from {ckpt_path}")
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_loss = ckpt["loss"]

        # if resuming, current and total steps must be set to match scheduler
        if self.cfg.resume:
            logger.info(f"resuming from step: {ckpt['step']}")
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.curr_step = ckpt["step"] + 1
            self.total_steps = ckpt["scheduler_state_dict"]["total_steps"] - 1

        # load parameters
        logger.info(f"loading model params from {ckpt_path}")
        self.model.load_state_dict(ckpt["model_state_dict"])

    def train_step(self, x, y):
        """single train step (weight update)"""
        self.model.train()
        # forward model while tracking gradients, compute loss
        with torch.enable_grad():
            y_pred = self.model(x)
            loss = self.cfg.loss(y_pred, y)
        # get current learning rate before optim step
        lr = self.optimizer.param_groups[0]["lr"]
        # backward step
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return y_pred, loss, lr

    def evaluate(self):
        """evaluation routine, iterating over val/test set"""
        self.model.eval()
        losses, metric1s = [], []
        pbar = tqdm(self.test_loader)
        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            # forward model without tracking gradients, compute loss
            with torch.no_grad():
                y_pred = self.model(x)
                loss = self.cfg.loss(y_pred, y)

            # compute relevant metrics
            y_pred_digit = y_pred.argmax(dim=1, keepdim=True)
            metric1 = self.cfg.metric1(y_pred_digit, y)

            # append losses and metrics to running lists
            losses.append(loss.item())
            metric1s.append(metric1.item())

            pbar.set_description(f"EVALUATION loss {np.mean(losses):.6e}")

        # log val/test quantities (losses, metrics, batch of images)
        mean_loss = float(np.mean(losses))
        mean_metric1 = float(np.mean(metric1s))
        self.recorder.log_metric("loss_test", mean_loss, self.curr_step)
        self.recorder.log_metric(
            self.cfg.metric1.name + "_test", mean_metric1, self.curr_step
        )
        self.recorder.log_image_grid(x.detach().cpu(), name="digits_test")

        # model checkpointing
        if self.curr_step % self.cfg.log.save_freq == 0:
            # update best loss, possibly save best model state
            if mean_loss < self.best_loss:
                if self.cfg.log.save_best:
                    self.save_model("best.pt", loss=self.best_loss)
            # save latest model
            if self.cfg.log.save_last:
                self.save_model("last.pt", loss=mean_loss)

        # ray tune
        if self.tune:
            # tune automatically creates the checkpoint dir
            # we must report eval loss back to tune as below
            with tune.checkpoint_dir(step=self.curr_step) as checkpoint_dir:
                path = Path(checkpoint_dir) / "last.pt"
                self.save_model(path, loss=mean_loss)
                tune.report(loss=mean_loss, step=self.curr_step)
                # link mlflow articats/tune to tune run directory if unlinked
                if not self.tune_linked:
                    tune_root = Path(checkpoint_dir).parent
                    (self.recorder.root / "tune").symlink_to(tune_root)
                    self.tune_linked = True

    def run(self):
        """iterate over train set and evaluate on val/test set"""
        data_iter = iter(self.train_loader)

        # initialize running lists of quantities to be logged
        losses, metric1s = [], []

        for step in range(self.curr_step, self.total_steps + 1):
            # allow repeated iteration over entire dataset
            try:
                x, y = next(data_iter)
            except StopIteration:
                # dataloader *is* reshuffled
                data_iter = iter(self.train_loader)
                x, y = next(data_iter)

            self.curr_step = step
            x = x.to(self.device)
            y = y.to(self.device)

            # forward the model, calculate loss
            y_pred, loss, lr = self.train_step(x, y)

            # compute relevant metrics
            y_pred_digit = y_pred.argmax(dim=1, keepdim=True)
            metric1 = self.cfg.metric1(y_pred_digit, y)

            # append losses and metrics to running lists
            losses.append(loss.item())
            metric1s.append(metric1.item())

            # log train quantities (losses, metrics, batch of images)
            if step % self.cfg.log.train_freq == 0 or step == self.total_steps:
                mean_train_loss = float(np.mean(losses))
                mean_metric1 = float(np.mean(metric1s))
                losses = []
                train_progress = (
                    f"TRAIN STEP {step}/{self.total_steps}: "
                    + f"loss {mean_train_loss:.6e} lr {lr:.2e}"
                )
                logger.info(train_progress) if self.tune else print(train_progress)
                train_metrics = {
                    "lr": lr,
                    "loss_train": mean_train_loss,
                    self.cfg.metric1.name + "_train": mean_metric1,
                }
                self.recorder.log_metrics(train_metrics, step)
                if self.cfg.log.images:
                    self.recorder.log_image_grid(x.detach().cpu(), name=f"digits_train")

            # evaluate test set
            if step % self.cfg.log.test_freq == 0 or step == self.total_steps:
                self.evaluate()
