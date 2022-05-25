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
from dl_schema.recorder import Recorder

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] (%(levelname)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    # parse CLI args or yaml config
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

    # create recorder and start mlflow run
    recorder = Recorder(cfg)
    recorder.create_experiment()
    with recorder.start_run():
        # build model
        logger.info(f"initializing model: {cfg.model.model_class}")
        model = build_model(model_class=cfg.model.model_class, cfg=cfg.model)

        # initialize Trainer
        logger.info("initializing trainer")
        if train_dataset is None and test_dataset is None:
            logger.info("no datasets found, check that MNIST data exists")
        trainer = Trainer(model, cfg, train_dataset, test_dataset, recorder)

        # log params, state dicts, and relevant training scripts to mlflow
        script_dir = Path(__file__).parent
        cfg_dict = pyrallis.encode(cfg)  # cfg as dict, encoded for yaml
        recorder.log_artifact(script_dir / "train.py", "archive")
        recorder.log_artifact(script_dir / "trainer.py", "archive")
        recorder.log_artifact(script_dir / "dataset.py", "archive")
        recorder.log_artifact(script_dir / "cfg.py", "archive")
        recorder.log_artifact(script_dir / "models/babycnn.py", "archive")
        recorder.log_dict(cfg_dict, "archive/cfg.yaml")
        recorder.log_params(flatten(cfg_dict))

        # train
        if cfg.load_ckpt_pth:
            trainer.load_model()
        if cfg.save_init:
            trainer.save_model("init.pt")
        trainer.train()

        # stop mlflow run
        recorder.end_run()


if __name__ == "__main__":
    main()
