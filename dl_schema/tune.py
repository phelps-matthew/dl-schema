"""ray.tune cannot serialize dataclasses with enum fields; thus any object depending on 
cfg must be constructed entirely within the primary run function."""
import os
import logging
import importlib.util
from pathlib import Path
import pprint
import collections

import pyrallis
import mlflow

from ray import tune
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.schedulers import ASHAScheduler

from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import CLIReporter

from spec21.keypoint_regression.dataset import MyDataset
from spec21.keypoint_regression.hil_dataset import HILDataset
from spec21.keypoint_regression.models import build_model
from spec21.keypoint_regression.cfg.train import TrainConfig
from spec21.keypoint_regression.trainer import Trainer
from spec21.utils import set_seed, flatten
from spec21.keypoint_regression.recorder import Recorder

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] (%(levelname)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def _run(cfg, checkpoint_dir=None):
    """execute training, testing, or inference run based on cfg"""
    # instantiate recorder and set to current experiment/run
    recorder = Recorder(cfg)
    recorder.set_experiment()
    recorder.set_run()

    # create datasets
    logger.info("loading datasets")
    train_dataset, test_dataset = None, None
    if cfg.data.train_root is not None and Path(cfg.data.train_root).is_dir():
        train_dataset = MyDataset(split="train", cfg=cfg)
    if cfg.data.test_root is not None and Path(cfg.data.test_root).is_dir():
        if cfg.infer:
            test_dataset = HILDataset(cfg=cfg)
        else:
            test_dataset = MyDataset(split="val", cfg=cfg)

    # cfg as dict, encoded and ready for yaml
    cfg_dict = pyrallis.encode(cfg)

    # build model
    logger.info(f"initializing model: {cfg.model.cfg.model_class}")
    model = build_model(model_class=cfg.model.cfg.model_class, cfg=cfg.model.cfg)

    # initialize Trainer
    logger.info("initializing trainer")
    trainer = Trainer(model, cfg, train_dataset, test_dataset, recorder, verbose=False)

    # log params, state dicts, and relevant training scripts to mlflow
    script_dir = Path(__file__).parent
    recorder.log_artifact(script_dir / "tune.py", "archive")
    recorder.log_artifact(script_dir / "train.py", "archive")
    recorder.log_artifact(script_dir / "trainer.py", "archive")
    recorder.log_artifact(script_dir / "dataset.py", "archive")
    recorder.log_artifact(script_dir / "recorder.py", "archive")
    recorder.log_artifact(script_dir / "cfg/train.py", "archive/cfg")
    recorder.log_dict(cfg_dict, "archive/cfg/cfg.yaml")
    if cfg.pnp:
        recorder.log_artifact(script_dir / "models/pnp.py", "archive")
    if cfg.py_cfg is not None and (script_dir / Path(cfg.py_cfg)).is_file():
        recorder.log_artifact(script_dir / Path(cfg.py_cfg), "archive/cfg")
    recorder.log_params(flatten(cfg_dict))
    recorder.log_params(
        {
            "n_train": len(train_dataset) if train_dataset is not None else 0,
            "n_val": len(test_dataset) if test_dataset is not None else 0,
        }
    )

    # `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint should be
    # restored
    if checkpoint_dir:
        print(f"\nLOADING CKPT DIR {checkpoint_dir}\n")
        trainer.cfg.resume = True
        trainer.load_model(Path(checkpoint_dir / "last.pt"))

    # load ckpts, save initialization
    if cfg.load_ckpt_pth:
        trainer.load_model(cfg.load_ckpt_pth)
    if cfg.save_init:
        trainer.save_model("init.pt")

    # train or infer on datasets
    if cfg.infer:
        trainer.infer()
    else:
        trainer.train()


def run(config, checkpoint_dir=None, cfg_dict=None):
    """helper function to pass in deserialized and updated cfg to _run"""
    # deserialize dictionary back into dataclass
    cfg = pyrallis.parsers.decoding.decode(TrainConfig, cfg_dict)

    # set selected hyperparamters from tune.run
    cfg.weight_decay = config["weight_decay"]
    cfg.bs = config["batch_size"]
    cfg.lr = config["lr"]

    # run the primary function
    _run(cfg)


def tune_function(cfg):
    """serializes cfg, defines run_fn, sets scheduler, and execute tune.run"""
    # must serialize cfg here before passing to tune.run
    # ray cannot pickle Enum dataclass types
    cfg_dict = pyrallis.encode(cfg)

    # rather than wrapping a partial, simply wrap defined run_fn here
    # this sets tracking uri according to config
    @mlflow_mixin
    def run_fn(config, checkpoint_dir=None, cfg=cfg_dict):
        return run(config, checkpoint_dir, cfg_dict)

    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=cfg_dict["epochs"], grace_period=4
    )

    # this reporter omits extraneous mlflow information in progress report
    # also allows tables to persist at low verbose levels
    reporter = CLIReporter(
        parameter_columns=["batch_size", "lr", "weight_decay"],
        print_intermediate_tables=True,
    )


    # setup hyperparmeter search
    mlflow_config = {
        "tracking_uri": mlflow.get_tracking_uri(),
        "experiment_name": cfg_dict["exp_name"],
    }

    hyperopt_search = HyperOptSearch(metric="loss", mode="min")

    config = {
        "mlflow": mlflow_config,
        "lr": tune.loguniform(1e-7, 1e-2),
        "batch_size": tune.choice([4, 8, 16, 32]),
        "weight_decay": tune.loguniform(1e-6, 1e-1),
    }

    # execute parallel hyperparam tuning
    tune.run(
        run_fn,
        name="tuneup",
        num_samples=2,
        config=config,
        resources_per_trial={"cpu": 4, "gpu": 0.2},
        # scheduler=scheduler,
        search_alg=hyperopt_search,
        keep_checkpoints_num=2,
        progress_reporter=reporter,
        verbose=1,
    )


def setup_experiment(cfg):
    """iniherit py_cfg args, set GPUs, and create mlflow experiment
    Args:
        cfg: pyrallis cfg parsed from CLI
    Returns:
        cfg
    """
    # check for external py cfg
    if cfg.py_cfg is not None and Path(cfg.py_cfg).is_file():
        logger.info(f"loading py cfg: {Path(cfg.py_cfg).resolve()}")
        cfg_pth = Path(cfg.py_cfg).resolve()
        # import the CFG object from path
        spec = importlib.util.spec_from_file_location("foo", str(cfg_pth))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg_new = mod.CFG
        # inherit the following CLI args
        cfg_new.py_cfg = cfg.py_cfg
        # gpu is to always be specified in CLI
        cfg_new.gpus = cfg.gpus
        cfg = cfg_new

    # make deterministic
    set_seed(cfg.seed)

    # log the config
    cfg_dict = pyrallis.encode(cfg)  # cfg as dict, encoded for yaml
    if cfg.verbose:
        logger.info("\n" + pprint.pformat(cfg_dict))

    # set GPU
    gpus = ",".join([str(i) for i in cfg.gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    logger.info(f"setting gpus: {gpus}")

    # create experiment if it does not exist
    if cfg.exp_name is not None:
        if mlflow.get_experiment_by_name(cfg.exp_name) is None:
            logger.info(f"creating mlflow experiment: {cfg.exp_name}")
            mlflow.create_experiment(cfg.exp_name)

    return cfg


def main():
    """outermost execution"""
    # parse config from CLI
    cfg = pyrallis.parse(config_class=TrainConfig)

    # iniherit py_cfg args, set GPUs, and create mlflow experiment
    cfg = setup_experiment(cfg)

    # run ray tune
    tune_function(cfg)


if __name__ == "__main__":
    main()
