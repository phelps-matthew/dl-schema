"""Sample training run"""
import logging
import os
from pathlib import Path

import mlflow
import pyrallis

from dl_schema.cfg import TrainConfig
from dl_schema.dataset import MNISTDataset
from dl_schema.models import build_model
from dl_schema.trainer import Trainer
from dl_schema.utils import flatten, set_seed

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] (%(levelname)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    cfg = pyrallis.parse(config_class=TrainConfig)

    # make deterministic
    set_seed(cfg.seed)

    # set GPU
    gpus = ",".join([str(i) for i in cfg.gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    logger.info(f"setting gpus: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # create datasets
    logger.info("loading datasets")
    train_dataset, test_dataset = None, None
    if (
        cfg.data.train_root is not None
        and Path(cfg.data.train_root).expanduser().exists()
    ):
        train_dataset = MNISTDataset(split="train", cfg=cfg)
    if (
        cfg.data.test_root is not None
        and Path(cfg.data.test_root).expanduser().exists()
    ):
        test_dataset = MNISTDataset(split="test", cfg=cfg)

    # create experiment
    # if experiment does not exist, create one
    if mlflow.get_experiment_by_name(cfg.exp_name) is None:
        logger.info(f"creating mlflow experiment: {cfg.exp_name}")
        exp_id = mlflow.create_experiment(cfg.exp_name)
    # otherwise, return exp_id of existing experiment
    else:
        exp_id = mlflow.get_experiment_by_name(cfg.exp_name).experiment_id

    # train as mlflow run
    with mlflow.start_run(experiment_id=exp_id, run_name=cfg.run_name):
        mlflow_artifact_path = mlflow.active_run().info.artifact_uri[7:]
        logger.info(f"starting mlflow run: {Path(mlflow_artifact_path).parent}")

        # build model
        logger.info(f"initializing model: {cfg.model.model_class}")
        model = build_model(model_class=cfg.model.model_class, cfg=cfg.model)

        # initialize Trainer
        logger.info("initializing trainer")
        if train_dataset is None and test_dataset is None:
            logger.info("no datasets found, check that MNIST data exists")
        trainer = Trainer(model, cfg, train_dataset, test_dataset)

        # log params, state dicts, and relevant training scripts to mlflow
        cfg_dict = pyrallis.encode(cfg)  # cfg as dict, encoded for yaml
        script_dir = Path(__file__).parent
        mlflow.log_artifact(script_dir / "train.py", "archive")
        mlflow.log_artifact(script_dir / "trainer.py", "archive")
        mlflow.log_artifact(script_dir / "dataset.py", "archive")
        mlflow.log_artifact(script_dir / "cfg.py", "archive")
        mlflow.log_dict(cfg_dict, "archive/cfg.yaml")
        mlflow.log_params(flatten(cfg_dict))

        # train
        if cfg.load_ckpt_pth:
            trainer.load_model()
        if cfg.save_init:
            trainer.save_model("init.pt")
        trainer.train()


if __name__ == "__main__":
    main()
