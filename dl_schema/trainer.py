"""Sample training loop."""
import logging
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm

from dl_schema.utils import image_grid

logger = logging.getLogger(__name__)


class Trainer:
    """train or evaluate a dataset over n epochs"""

    def __init__(self, model, cfg, train_dataset=None, test_dataset=None, verbose=True):
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_steps = self.get_dataset_size(self.train_dataset)
        self.test_steps = self.get_dataset_size(self.test_dataset)
        self.test_only = self.train_dataset is None
        self.verbose = verbose
        self.curr_epoch = 0
        self.scheduler = None

        # set mlflow paths for model/optim saving
        mlflow_artifact_path = mlflow.active_run().info.artifact_uri[7:]
        self.ckpt_root = Path(mlflow_artifact_path) / "checkpoints"
        (self.ckpt_root).mkdir(parents=True, exist_ok=True)

        # configure optimizer
        if self.test_only:
            self.optimizer = None
            self.cfg.load_optim_pth = None
        else:
            self.set_scheduler(self.train_steps)
            self.optimizer = tfa.optimizers.AdamW(
                weight_decay=self.cfg.weight_decay,
                learning_rate=self.scheduler,
                beta_1=self.cfg.adam_beta1,
                beta_2=self.cfg.adam_beta2,
            )

        # initialize best loss for ckpt saving
        self.best_epoch_loss = float("inf")

        # pass a dummy data sample through model to create weights (oh keras..)
        if not self.test_only:
            x, _ = next(iter(self.train_dataset))
            self.model(x)

    def get_dataset_size(self, dataset):
        """iterate over dataset to obtain number of batches (steps)"""
        if dataset is None:
            return 0
        return sum(1 for _ in dataset)

    def set_scheduler(self, total_steps=None):
        """create lr scheduler; total steps argument required for cyclic lr"""
        if self.cfg.lr_method.name == "cyclic":
            self.scheduler = self.cfg.lr_method(
                initial_learning_rate=self.cfg.cyclic_lr_initial,
                maximal_learning_rate=self.cfg.lr,
                scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
                step_size=total_steps // self.cfg.cyclic_n_cycles,
            )
        else:
            self.scheduler = self.cfg.lr_method(lr=self.cfg.lr)

    def save_model(self, filename="last"):
        """save model weights and optimizer"""
        # NOTE: saving current epoch and best loss supported in torch branch
        weight_path = self.ckpt_root / f"{filename}_weights.h5"
        optim_path = self.ckpt_root / f"{filename}_optim.npy"
        if self.verbose:
            logger.info(f"saving weights to {weight_path}")
            logger.info(f"saving optimizer to {optim_path}")
        self.model.save_weights(self.ckpt_root / f"{filename}_weights.h5")
        np.save(
            optim_path,
            np.array(self.optimizer.get_weights(), dtype=object),
            allow_pickle=True,
        )

    def load_model(self):
        """load model weights and (optionally) optimizer"""
        # NOTE: resuming from last or best epoch supported in torch branch
        # load weights
        weights_path = Path(self.cfg.load_weights_pth).expanduser().resolve()
        logger.info(f"loading weights from {weights_path}")
        self.model.load_weights(weights_path)

        # load optimizer (optional)
        if self.cfg.load_optim_pth is not None:
            optim_path = Path(self.cfg.load_optim_pth).expanduser().resolve()
            logger.info(f"loading optimizer from {optim_path}")
            # nonsense to load optimizer
            optim_weights = np.load(optim_path, allow_pickle=True)
            grad_vars = self.model.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            self.optimizer.apply_gradients(zip(zero_grads, grad_vars))
            self.optimizer.set_weights(optim_weights)

    def run_epoch(self, split: Literal["train", "test"] = "train"):
        """train or evalauate on a single epoch, returning mean epoch loss"""
        is_train = split == "train"
        epoch = self.curr_epoch
        dataset = self.train_dataset if is_train else self.test_dataset
        steps = self.train_steps if is_train else self.test_steps

        # initialize running lists of quantities to be logged
        losses, metric1s = [], []

        # train/test loop
        pbar = tqdm(enumerate(dataset), total=steps)
        for it, (x, y) in pbar:

            # get current learning rate before optim step (for logging)
            curr_lr = self.optimizer.lr(self.optimizer.iterations).numpy()

            # forward the model, calculate loss
            if is_train:
                with tf.GradientTape() as tape:
                    y_pred = self.model(x, training=is_train)
                    loss = self.cfg.loss(y, y_pred)
                # backward step
                grads = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            else:
                y_pred = self.model(x, training=is_train)
                loss = self.cfg.loss(y, y_pred)

            # calculate relevant metrics
            y_pred_digit = tf.math.argmax(y_pred, axis=-1, output_type=tf.int32)
            metric1 = self.cfg.metric1(y_pred_digit, y)

            # append losses and metrics to running lists
            losses.append(loss.numpy())
            metric1s.append(metric1.numpy())

            # report progress bar
            lr_str = f"lr {curr_lr:.2e}" if curr_lr is not None else ""
            pbar.set_description(
                f"({split}) epoch {epoch} iter {it}: {split} loss {loss.numpy():.6e} "
                + lr_str
            )

            # log batch quantities
            step = it + epoch * steps
            if step % self.cfg.log.batch_freq == 0:
                suffix = f"_{split}_batch"
                if is_train:
                    mlflow.log_metric("lr" + suffix, curr_lr, step)
                mlflow.log_metric("loss" + suffix, loss.numpy(), step)
                mlflow.log_metric(self.cfg.metric1.name + suffix, metric1.numpy(), step)
                # log grid of batch images
                if self.cfg.log.image_grid:
                    grid = image_grid(x.numpy())
                    mlflow.log_image(grid, f"digits{suffix}.png")

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
                        self.save_model("best")

                # save latest model
                if cfg.save_last:
                    self.save_model("last")

        return self.best_epoch_loss
