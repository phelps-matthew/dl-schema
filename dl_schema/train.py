"""Sample training run"""
import os
import logging
from pathlib import Path
import pprint
from dataclasses import asdict
import mlflow
import pyrallis
from dl_schema.dataset import MyDataset
from dl_schema.model import build_model
from dl_schema.cfg import TrainConfig
from dl_schema.trainer import Trainer
from dl_schema.utils import set_seed, flatten

# checkpoint = torch.load('cifar10_model.pt')
# model.load_state_dict(checkpoint)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] (%(levelname)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Make deterministic
set_seed(42)


def main():
    cfg = pyrallis.parse(config_class=TrainConfig)

    # Set GPU
    gpus = ",".join([str(i) for i in cfg.gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    logger.info(f"setting gpus: {gpus}")

    # Sample usage, 80%/20% split.
    logger.info("loading datasets")
    train_dataset = MyDataset(split="train", **asdict(cfg.data))
    val_dataset = MyDataset(split="val", **asdict(cfg.data))

    # Create experiment
    exp_id = None
    if cfg.exp_name is not None:
        logger.info(f"creating mlflow experiment: {cfg.exp_name}")
        exp_id = mlflow.create_experiment(cfg.exp_name)

    # Train as mlflow run
    with mlflow.start_run(experiment_id=exp_id, run_name=cfg.run_name):
        logger.info(f"mlflow exp_id: {mlflow.active_run().info.experiment_id}")

        # Log the config
        cfg_dict = pyrallis.encode(cfg)  # cfg as dict, encoded for yaml
        logger.info("\n" + pprint.pformat(cfg_dict))

        # Build model
        logger.info(f"loading model: {cfg.model.name}")
        model = build_model(asdict(cfg.model))

        # Initialize Trainer
        logger.info("initializing trainer")
        trainer = Trainer(model, cfg, train_dataset, val_dataset)

        # Log params, state dicts, train script to mlflow
        mlflow.log_dict(cfg_dict, "cfg.yaml")
        mlflow.log_artifact(Path(__file__))
        mlflow.log_params(flatten(cfg_dict))

        # Train
        trainer.save_model("init.pt")
        trainer.save_optimizer("init_optim.pt")
        trainer.train()


if __name__ == "__main__":
    main()
