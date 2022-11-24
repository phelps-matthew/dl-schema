"""
Training executer base class. Abstracts the handling of lr schedulers, optimizers,
model saving/loading, and datasets/generators at one higher level.
"""
import logging
from pathlib import Path

from ray import tune
import torch
from torch.utils.data.dataloader import DataLoader

from dl_schema.utils.utils import configure_adamw

logger = logging.getLogger(__name__)


class TrainerBase:
    """Setup dataloaders, optimizers, schedules, saving/loading, etc."""

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
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
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
        self.train_loader = self.create_dataloader(self.train_dataset, train=True)
        self.val_loader = self.create_dataloader(self.val_dataset, train=False)
        self.test_loader = self.create_dataloader(self.test_dataset, train=False)

        # configure optimizer
        self.infer = self.train_dataset is None
        if self.infer:
            self.optimizer = None
            self.cfg.load_optimizer = False
        else:
            self.optimizer = configure_adamw(self.model, self.cfg)
            self.set_scheduler()

        # initialize best loss for ckpt saving
        self.best_loss = float("inf")

    def create_dataloader(self, dataset, train=True):
        if dataset is None:
            return None
        loader = DataLoader(
            dataset,
            shuffle=self.cfg.data.shuffle if train else False,
            pin_memory=True,
            batch_size=self.cfg.bs,
            num_workers=self.cfg.num_workers,
            drop_last=self.cfg.data.drop_last,
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
        elif self.cfg.lr_method.name in {
            "linear_warmup_cosine_decay",
            "linear_warmup_linear_decay",
        }:
            self.scheduler = self.cfg.lr_method(
                self.optimizer,
                num_warmup_steps=int(self.total_steps * self.cfg.warmup_pct / 100),
                num_training_steps=self.total_steps + 1,
            )
        elif self.cfg.lr_method.name == "linear_warmup_cosine_decay_hard_restart":
            self.scheduler = self.cfg.lr_method(
                self.optimizer,
                num_warmup_steps=int(self.total_steps * self.cfg.warmup_pct / 100),
                num_training_steps=self.total_steps + 1,
                num_cycles=self.cfg.restart_cycles,
            )
        else:
            self.scheduler = self.cfg.lr_method(self.optimizer, lr_lambda=lambda step: 1)

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

    def tune_hook(self):
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

    def train_step(self, x, y):
        """single train step (weight update)"""
        raise NotImplementedError

    def evaluate(self, split="val"):
        """evaluation routine, iterating over val/test set"""
        raise NotImplementedError

    def run(self):
        """iterate over train set and evaluate on val/test set"""
        raise NotImplementedError
