"""
Logging base class for training runs. The philosophy is that each repo will require its
own unique analytics and thus is to inherit the RecorderBase and implement any necessary
custom logic. The base class here provides many of the common utilities and methods
shared among different ML projects.
"""
from functools import partial
import logging
import math
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Callable

import PIL
from matplotlib import pyplot as plt
import mlflow
import numpy as np
import torch
import torchvision

logger = logging.getLogger(__name__)


class AsyncCaller:
    """
    Implements asynchronous execution of class methods within an arbitrary class by
    providing a convenient decorator `async_dec`. Uses a single queue and multiplethreads
    (workers) to consume the queue (i.e. consumer/producer pattern).

    Usage:
        Decorate a class method via `@AsyncCaller.async_dec(ac_attr="async_log")`. The
        target class must possess the corresponding attribute `async_log = AsyncCaller()`.

    Note:
        In applying the decorator @AsyncCaller.async_dec(ac_attr="async_log"), the
        async_dec method will look at the class instance (e.g. Recoder or RecorderBase)
        and determine if there is an attribute named 'async_log' and if that attribute is
        callable. If true, serves as the indicator that async calling is to be applied.

        Additionally, the `wrapper` function is only called when the decorated method is
        called, otherwise the decorated methods simply point to the `wrapper` function.
    """

    STOP_MARK = "__STOP"

    def __init__(self, num_threads=4) -> None:
        self._q = Queue()
        self._stop = False
        self.num_threads = num_threads
        self.threads = []
        self.start_threads()

    def start_threads(self):
        """spin up n threads actively consuming queue via `run`"""
        for _ in range(self.num_threads):
            t = Thread(target=self.run)
            self.threads.append(t)
            t.start()

    def close(self):
        """add STOP_MARK to queue to trigger thread termination"""
        self._q.put(self.STOP_MARK)

    def run(self):
        """consume queue until STOP_MARK is reached"""
        while True:
            data = self._q.get()
            if data == self.STOP_MARK:
                break
            data()

    def __call__(self, func, *args, **kwargs):
        self._q.put(partial(func, *args, **kwargs))

    def wait(self, close=True):
        """block parent thread until async threads terminate"""
        if close:
            for t in self.threads:
                self.close()
        for i, t in enumerate(self.threads):
            logger.info(f"closing thread {i + 1}/{self.num_threads}")
            t.join()

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
    """
    To be inherited by custom user-defined Recorder class. Provides core backend methods for
    managing runs/experiments and assists in fine control of logging training run quantities.
    Includes a number of image logging helper utilities to be called within child class.
    """

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
        if "file:/" in self._artifact_uri:
            self._artifact_uri = self.run.info.artifact_uri[7:]
        self.root = Path(self._artifact_uri)  # cut file:/ uri
        logger.info(f"starting mlflow run: {Path(self.root).parent}")

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
        if self.cfg.log.enable_async:
            n_threads = self.cfg.log.num_threads
            logger.info(f"enabling async logging with {n_threads} threads")
            # note matching "async_log" in AsyncCaller method decorations
            self.async_log = AsyncCaller(num_threads=n_threads)
        else:
            self.async_log = None
        return self.run

    def end_run(self):
        mlflow.end_run()
        if self.async_log is not None:
            logger.info("waiting for recorder threads to finish")
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
    def log_image_grid(
        self,
        x: torch.Tensor,
        name: str = "x",
        NCHW: bool = True,
        normalize: bool = True,
        jpg: bool = True,
        padding: bool = 1,
    ):
        """log batch of images

        Args:
            x: torch.float32 of shape (N, C, H, W) or (N, H, W, C)
            name: filename of rendered image, not including file extension
            NCHW: if false, will convert from NHWC to NCHW
            normalize: apply per-instance normalization
            jpg: if false, convert to png
            padding: pixel padding between images in grid
        """
        N = x.shape[0]
        n_rows = math.ceil(math.sqrt(N))  # actually n_cols

        if x is not None:
            if not NCHW:
                x = x.permute(0, 3, 1, 2)
            img_fmt = "jpg" if jpg else "png"
            grid_x = torchvision.utils.make_grid(
                x, normalize=normalize, nrow=n_rows, pad_value=1.0, padding=padding
            ).permute(1, 2, 0)
            self.client.log_image(self.run_id, grid_x.numpy(), f"{name}.{img_fmt}")

    def figure_to_array(self, figure: plt.Figure) -> np.ndarray:
        """convert matplotlib figure to numpy array"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close(figure)
        buffer.seek(0)
        img = PIL.Image.open(buffer)
        data = np.array(img)
        return data

    def norm_batch(self, tensor: torch.Tensor):
        """normalize batch of images"""
        for t in tensor:  # loop over batch dimension
            self.norm_img(t)
        return tensor

    def norm_img(self, img: torch.Tensor):
        """normalize image to [0.0, 1.0] by clamping in range [low, high]"""
        low = float(img.min())
        high = float(img.max())
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))
        return img
