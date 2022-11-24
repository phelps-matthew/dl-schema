"""
Uses ray tune to perform hyparameter search, optimization, and run scheduling

Note: ray.tune cannot serialize dataclasses with enum fields; thus any object depending on 
    cfg must be constructed entirely within the primary run function
"""
import logging
from pathlib import Path

import mlflow
import pyrallis
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from dl_schema.cfg import TrainConfig
from dl_schema.dataset import MNISTDataset
from dl_schema.models import build_model
from dl_schema.recorder import Recorder
from dl_schema.trainer import Trainer
from dl_schema.utils.utils import flatten, set_seed


def _run(cfg, checkpoint_dir=None):
    """execute training, testing, or inference run based on cfg"""

    # Set up logging (within an individual run)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] (%(levelname)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # instantiate recorder and set to current experiment/run
    recorder = Recorder(cfg)
    recorder.set_experiment()
    recorder.set_run()

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

    # build model
    logger.info(f"initializing model: {cfg.model.model_class}")
    model = build_model(model_class=cfg.model.model_class, cfg=cfg.model)

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
        "base/recorder_base.py",
        "base/trainer_base.py",
        "models/babycnn.py",
    ]
    for relpath in src_files:
        recorder.log_artifact(script_dir / relpath, "archive")

    # `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint should be
    # restored
    if checkpoint_dir:
        trainer.cfg.resume = True
        trainer.load_model(Path(checkpoint_dir / "last.pt"))

    # train
    if cfg.load_ckpt_pth:
        trainer.load_model()
    if cfg.log.save_init:
        trainer.save_model("init.pt")
    trainer.run()

    # stop mlflow run, exit gracefully
    recorder.end_run()


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
    def run_fn(config, checkpoint_dir=None):
        return run(config, checkpoint_dir, cfg_dict)

    # default trial names are obscene, lets tame it down
    def trial_dir_str(trial):
        return "{}_{}".format(trial.trainable_name, trial.trial_id)

    # this reporter omits extraneous mlflow information in progress report
    # also allows tables to persist at low verbose levels
    reporter = CLIReporter(
        parameter_columns=["batch_size", "lr", "weight_decay"],
        print_intermediate_tables=True,
    )

    # enable asynchronous successive halving algorithm scheduler
    #scheduler = ASHAScheduler(
    #   metric="loss", mode="min", max_t=cfg_dict["epochs"], grace_period=4
    #)

    # define hyperparmeter search algorithm
    hyperopt_search = HyperOptSearch(metric="loss", mode="min")

    config = {
        "mlflow": {
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment_name": cfg_dict["exp_name"],
        },
        "lr": tune.loguniform(1e-7, 1e-2),
        "batch_size": tune.choice([40, 80, 160, 320]),
        "weight_decay": tune.loguniform(1e-6, 1e-1),
    }

    # execute parallel hyperparam tuning
    tune.run(
        run_fn,
        name="tuneup",
        num_samples=3,
        config=config,
        resources_per_trial={"cpu": 2, "gpu": 1},
        #scheduler=scheduler,
        search_alg=hyperopt_search,
        keep_checkpoints_num=2,
        progress_reporter=reporter,
        verbose=1,
        local_dir="./ray_results",
        log_to_file=("stdout.log", "stderr.log"),
        trial_dirname_creator=trial_dir_str,
    )


def setup_experiment(cfg):
    """setup ray tune, cfg, and mlflow for hyperparam experiment"""
    # make deterministic
    set_seed(cfg.seed)

    # disable internal log and print statements convoluting ray cli logs
    ray.init(log_to_driver=False)

    # force async to false as this breaks ray tune; force tune
    cfg.log.enable_async = False
    cfg.tune = True

    # make dataset paths absolute
    if (
        cfg.data.train_root is not None
        and Path(cfg.data.train_root).expanduser().exists()
    ):
        cfg.data.train_root = Path(cfg.data.train_root).expanduser().absolute()
    if (
        cfg.data.test_root is not None
        and Path(cfg.data.test_root).expanduser().exists()
    ):
        cfg.data.test_root = Path(cfg.data.test_root).expanduser().absolute()

    # create experiment if it does not exist
    if cfg.exp_name is not None:
        if mlflow.get_experiment_by_name(cfg.exp_name) is None:
            print(f"creating mlflow experiment: {cfg.exp_name}")
            mlflow.create_experiment(cfg.exp_name)

    return cfg


def main():
    # parse config from CLI
    cfg = pyrallis.parse(config_class=TrainConfig)

    # prepare cfg for mlflow hyperparam experiment
    cfg = setup_experiment(cfg)

    # run ray tune
    tune_function(cfg)


if __name__ == "__main__":
    main()
