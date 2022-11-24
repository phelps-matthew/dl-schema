"""Sample training run"""
import logging
import os
from pathlib import Path

import pyrallis

from dl_schema.cfg import TrainConfig
from dl_schema.dataset import MNISTDataset
from dl_schema.models import build_model
from dl_schema.recorder import Recorder
from dl_schema.trainer import Trainer
from dl_schema.utils.utils import flatten, set_seed

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
        logger.info(f"initializing model: {cfg.model_class}")
        if "resnet" in cfg.model_class:
            model_cfg = cfg.resnet
        elif "VGG" in cfg.model_class:
            model_cfg = cfg.vgg11
        else:
            model_cfg = cfg.babycnn
        model = build_model(cfg.model_class, model_cfg)

        # add parameter and gradient logging (if specified in cfg)
        recorder.add_weights_and_grads_hooks(model)

        # initialize Trainer
        logger.info("initializing trainer")
        if train_dataset is None and test_dataset is None:
            logger.info("no datasets found, check that MNIST data exists")
        trainer = Trainer(model, cfg, train_dataset, test_dataset, recorder)

        # log config as params and yaml
        cfg_dict = pyrallis.encode(cfg)  # cfg as dict, encoded for yaml
        recorder.log_dict(cfg_dict, "archive/cfg.yaml")
        recorder.log_params(flatten(cfg_dict))

        # log relevant source files
        script_dir = Path(__file__).parent
        src_files = [
            "cfg.py",
            "dataset.py",
            "recorder.py",
            "train.py",
            "trainer.py",
            "tune.py",
            "models/babycnn.py",
            "utils/recorder_base.py",
            "utils/utils.py",
        ]
        for relpath in src_files:
            recorder.log_artifact(script_dir / relpath, "archive")

        # train
        if cfg.load_ckpt_pth:
            trainer.load_model()
        if cfg.log.save_init:
            trainer.save_model("init.pt")
        trainer.run()

        # stop mlflow run, exit gracefully
        recorder.end_run()


if __name__ == "__main__":
    main()
