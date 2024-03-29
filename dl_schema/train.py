"""Entrypoint to train models."""
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
    train_dataset, val_dataset, test_dataset = None, None, None
    if (
        cfg.data.train_root is not None
        and Path(cfg.data.train_root).expanduser().exists()
    ):
        train_dataset = MNISTDataset(split="train", cfg=cfg)
    if cfg.data.val_root is not None and Path(cfg.data.val_root).expanduser().exists():
        val_dataset = MNISTDataset(split="val", cfg=cfg)
    if cfg.data.test_root is not None and Path(cfg.data.test_root).expanduser().exists():
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

        # log dataset size
        recorder.log_params(
            {
                "n_train": len(train_dataset) if train_dataset is not None else 0,
                "n_val": len(val_dataset) if val_dataset is not None else 0,
                "n_test": len(test_dataset) if test_dataset is not None else 0,
            }
        )

        # add parameter and gradient logging (if specified in cfg)
        recorder.add_weights_and_grads_hooks(model)

        # initialize Trainer
        logger.info("initializing trainer")
        trainer = Trainer(
            model,
            cfg,
            train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            recorder=recorder,
        )

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
            "base/recorder_base.py",
            "base/trainer_base.py",
            "models/babycnn.py",
            "models/vgg11.py",
            "models/resnet.py",
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
