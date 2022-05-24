"""
Logging utility for training runs

https://github.com/microsoft/qlib/blob/d7d19feb4ebb0c4318ac3bfda32a34c56e28a6a0/qlib/workflow/recorder.py#L368
"""
import math
import logging
from functools import partial
from threading import Thread
from typing import Callable
from queue import Queue
from pathlib import Path
import numpy as np
import mlflow
import torch

logger = logging.getLogger(__name__)


class AsyncCaller:
    """
    This AsyncCaller tries to make it easier to async call
    Currently, it is used in MLflowRecorder to make functions like `log_params` async
    NOTE:
    - This caller didn't consider the return value
    """

    STOP_MARK = "__STOP"

    def __init__(self) -> None:
        self._q = Queue()
        self._stop = False
        self._t = Thread(target=self.run)
        self._t.start()

    def close(self):
        self._q.put(self.STOP_MARK)

    def run(self):
        while True:
            data = self._q.get()
            if data == self.STOP_MARK:
                break
            data()

    def __call__(self, func, *args, **kwargs):
        self._q.put(partial(func, *args, **kwargs))

    def wait(self, close=True):
        if close:
            self.close()
        self._t.join()

    @staticmethod
    def async_dec(ac_attr):
        def decorator_func(func):
            def wrapper(self, *args, **kwargs):
                if isinstance(getattr(self, ac_attr, None), Callable):
                    return getattr(self, ac_attr)(func, self, *args, **kwargs)
                else:
                    return func(self, *args, **kwargs)

            return wrapper

        return decorator_func


class RecorderBase:
    """assists in fine control of logging training run quantities"""

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.log.uri is not None:
            self.uri = str(Path(cfg.log.uri).expanduser())
            mlflow.set_tracking_uri(self.uri)
        else:
            self.uri = mlflow.get_tracking_uri()
        self.client = mlflow.tracking.MlflowClient(tracking_uri=self.uri)
        self.exp_id = None
        self.run = None
        self.run_id = None

    def set_experiment(self, exp_name=None):
        """set exp_id as attribute, assumes experiment already exists"""
        if self.cfg.exp_name is not None:
            self.exp_id = mlflow.get_experiment_by_name(self.cfg.exp_name).experiment_id
        else:
            self.exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
        return self.exp_id

    def set_run(self, run=None):
        """set run and run_id attributes, assuming run already exists"""
        self.run = mlflow.active_run() if run is None else run
        self.run_id = self.run.info.run_id
        self._artifact_uri = self.run.info.artifact_uri
        self.root = Path(self._artifact_uri[7:])  # cut file:/ uri

        self.async_log = AsyncCaller() if self.cfg.log.enable_async else None
        return self.run

    def create_experiment(self):
        # create experiment if specified in cfg and does not exist
        if self.cfg.exp_name is not None:
            if mlflow.get_experiment_by_name(self.cfg.exp_name) is None:
                logger.info(f"creating mlflow experiment: {self.cfg.exp_name}")
                self.exp_id = mlflow.create_experiment(self.cfg.exp_name)
            else:
                self.exp_id = mlflow.get_experiment_by_name(
                    self.cfg.exp_name
                ).experiment_id
        return self.exp_id

    def start_run(self):
        # start run
        self.run = mlflow.start_run(
            run_id=None, experiment_id=self.exp_id, run_name=self.cfg.run_name
        )

        # save the run id and artifact_uri
        self.run_id = self.run.info.run_id
        self._artifact_uri = self.run.info.artifact_uri
        if "file:/" in self._artifact_uri:
            self._artifact_uri = self.run.info.artifact_uri[7:]
        self.root = Path(self._artifact_uri)  # cut file:/ uri
        logger.info(f"starting mlflow run: {Path(self.root).parent}")

        # initialize async logging
        logger.info("initializing async logging")
        self.async_log = AsyncCaller() if self.cfg.log.enable_async else None
        return self.run

    def end_run(self):
        """end mlflow run"""
        mlflow.end_run()
        if self.async_log is not None:
            self.async_log.wait()
        self.async_log = None

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_artifact(self, local_path, artifact_path=None):
        self.client.log_artifact(self.run_id, local_path, artifact_path)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_dict(self, d, artifact_path=None):
        self.client.log_dict(self.run_id, d, artifact_path)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_metric(self, k, v, step=None):
        self.client.log_metric(self.run_id, k, v, step=step)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_metrics(self, d, step=None):
        for name, data in d.items():
            if data is not None:
                self.client.log_metric(self.run_id, name, data, step=step)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_params(self, param_dict):
        for name, data in param_dict.items():
            self.client.log_param(self.run_id, name, data)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_text(self, text, artifact_path=None):
        self.client.log_text(self.run_id, text, artifact_path)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_iter(self, inputs, step=None, split="train"):
        raise NotImplementedError

    def norm_batch(self, tensor: torch.Tensor):
        """normalize batch of images"""
        for t in tensor:  # loop over batch dimension
            self.norm_img(t)
        return tensor

    def norm_img(self, img):
        """normalize image to [0.0, 1.0] by claming in range [low, high]"""
        low = float(img.min())
        high = float(img.max())
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))
        return img
