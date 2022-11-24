"""
Implements the execution of train/val/test steps and logs metrics and artifacts.
Optimizers, lr schedules, saving/loading, and dataloading is handled in inherited
TrainerBase.
"""
import logging

import numpy as np
import torch
from tqdm import tqdm

from dl_schema.base.trainer_base import TrainerBase

logger = logging.getLogger(__name__)


class Trainer(TrainerBase):
    """train over n steps and evaluate over val/test set"""

    def __init__(
        self,
        model,
        cfg,
        train_dataset,
        val_dataset=None,
        test_dataset=None,
        recorder=None,
        verbose=True,
    ):
        super(Trainer, self).__init__(
            model, cfg, train_dataset, val_dataset, test_dataset, recorder, verbose
        )

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

    def evaluate(self, split="val"):
        """evaluation routine, iterating over val/test set"""
        self.model.eval()
        losses, metric1s = [], []
        loader = self.test_loader if split == "test" else self.val_loader
        if loader is None:
            return
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

            pbar.set_description(f"{split.upper()} loss {np.mean(losses):.6e}")

        # log val/test quantities (losses, metrics, batch of images)
        mean_loss = float(np.mean(losses))
        mean_metric1 = float(np.mean(metric1s))
        eval_metrics = {
            f"loss_{split}": mean_loss,
            f"{self.cfg.metric1.name}_{split}": mean_metric1,
        }
        self.recorder.log_metrics(eval_metrics, self.curr_step)
        self.recorder.log_image_grid(
            x.detach().cpu(), name=f"digits_{split}", normalize=True
        )

        # model checkpointing
        if self.curr_step % self.cfg.log.save_freq == 0 and split == "val":
            # update best loss, possibly save best model state
            if mean_loss < self.best_loss:
                if self.cfg.log.save_best:
                    self.save_model("best.pt", loss=self.best_loss)
            # save latest model
            if self.cfg.log.save_last:
                self.save_model("last.pt", loss=mean_loss)

        # ray tune
        if self.tune and split == "val":
            self.tune_hook()

    def run(self):
        """iterate over train set and evaluate on val/test set"""
        # evaluate test set before first train step (get baseline)
        if self.cfg.log.evaluate_init:
            self.evaluate("val")
            self.evaluate("test")

        if self.train_loader is None:
            return
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
            self.recorder.curr_step = step
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
                self.recorder.log_weights_and_grad_histograms(
                    last_step=step == self.total_steps
                )
                mean_train_loss = float(np.mean(losses))
                mean_metric1 = float(np.mean(metric1s))
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
                    self.recorder.log_image_grid(
                        x.detach().cpu(), name=f"digits_train", normalize=True
                    )

                losses, metric1s = [], []

            # evaluate test set
            if (
                step > 1
                and step % self.cfg.log.test_freq == 0
                or step == self.total_steps
            ):
                self.evaluate("val")
                self.evaluate("test")
