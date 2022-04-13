import os
import logging
import random
import pprint
from dataclasses import asdict
from dl_schema.dataset import MyDataset
from dl_schema.models import build_model
from dl_schema.cfg import TrainConfig
from dl_schema.trainer import Trainer
from dl_schema.utils import set_seed, flatten
import numpy as np
import pyrallis
import mlflow
from ray.tune.schedulers import ASHAScheduler

from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback, mlflow_mixin


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] (%(levelname)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


@mlflow_mixin
def train_fn(config):
    # Pass tune configs into cfg dataclass
    cfg = TrainConfig()
    cfg.lr = config["lr"]
    cfg.weight_decay = config["weight_decay"]
    cfg.bs = config["batch_size"]

    # Build model
    logger.info(f"loading model: {cfg.model.name}")
    model = build_model(asdict(cfg.model))

    # Initialize trainer
    logger.info("train begin")
    trainer = Trainer(model, cfg, train_dataset, val_dataset, verbose=False)
    cfg_dict = {
        "model": asdict(cfg.model),
        "train": asdict(cfg.train),
        "data": asdict(cfg.data),
    }
    logger.info(pprint.pformat(cfg_dict))
    best_loss = trainer.train()


scheduler = ASHAScheduler(metric="loss", mode="min", max_t=10, grace_period=4)


def tune_function(exp_name):
    mlflow.create_experiment(exp_name)
    tune.run(
        train_fn,
        name="tune_run_name1",
        num_samples=1,
        # metric="best_loss",
        config={
            "mlflow": {
                "experiment_name": exp_name,
                "tracking_uri": mlflow.get_tracking_uri(),
            },
            "lr": tune.loguniform(1e-5, 1e-1),
            "batch_size": tune.choice([2, 4, 8, 16]),
            "weight_decay": tune.loguniform(1e-5, 1e-1),
        },
        resources_per_trial={"cpu": 1, "gpu": 0.5},
        scheduler=scheduler,
        local_dir="~/ray_results",
    )


def main():

    mlflow_tracking_uri = mlflow.get_tracking_uri()

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] (%(levelname)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Create config.
    cfg = TrainConfig()

    # Sample usage, 80%/20% split.
    logger.info("loading datasets")
    train_dataset = MyDataset(split="train", **asdict(cfg.data))
    val_dataset = MyDataset(split="val", **asdict(cfg.data))
    n = 10
    exp_name = "tune_exp_mixin_" + str(n)
    while mlflow.get_experiment_by_name(exp_name) is not None:
        n += 1
        exp_name = "tune_exp_mixin_" + str(n)
    tune_function(exp_name)


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=TrainConfig)
    print(pyrallis.encode(cfg))
    main()
