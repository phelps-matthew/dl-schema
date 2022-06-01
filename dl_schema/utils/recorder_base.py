"""
Logging base class for training runs. The philosophy is that each repo will require its
own unique analytics and thus is to inherit the RecorderBase and implement any necessary
custom logic. The base class here provides many of the common utilities and methods
shared among different ML projects.
"""
from collections import defaultdict
from functools import partial
import io
import logging
import math
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Callable, List, Tuple, Union

import PIL
import matplotlib
from matplotlib import pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torchvision

from dl_schema.utils.ridgeplot import ridgeplot

# use Agg backend for image rendering to file
matplotlib.use("Agg")


logger = logging.getLogger(__name__)

# TODO: Optimize histogram data types


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
        self._artifact_uri = None
        self.root = None
        self.async_log = None

        # hooks
        self.hook_handles = {}
        self.curr_step = 0

        # used to compute histograms of params and grads
        self._is_cuda_histc_supported = None
        self._num_bins = 64
        self._extra_bin_frac = 0.2
        self.histograms = defaultdict(list)

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
    def log_image(self, image: Union[np.ndarray, PIL.Image.Image], artifact_path=None):
        self.client.log_image(self.run_id, image, artifact_path)

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

    def figure_to_array(self, figure: plt.Figure, png: bool = True) -> np.ndarray:
        """convert matplotlib figure to numpy array"""
        fmt = "png" if png else "jpg"
        buffer = io.BytesIO()
        plt.savefig(buffer, format=fmt, dpi=100)
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

    def log_weights_and_grad_histograms(self):
        """create histogram plots over all stored kdes"""
        for k, v in self.histograms.items():
            self._generate_histogram(v, name=k)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def _generate_histogram(self, data: List, name):
        """generate histogram of single parameter or gradient"""
        counts, bins, steps = zip(*data)
        # find global min/max range
        bin_min, bin_max = (float("inf"), float("-inf"))
        for bin_ in bins:
            curr_bin_min = bin_[0][0]
            curr_bin_max = bin_[0][-1]
            if curr_bin_min < bin_min:
                bin_min = curr_bin_min
            if curr_bin_max > bin_max:
                bin_max = curr_bin_max
        x_range = [bin_min, bin_max]
        # plot no more than 11 labels for large total steps
        if len(steps) > 20:
            idxs = np.linspace(0, len(steps), 11, endpoint=False, dtype=int)
            labels = [s[0] if i in idxs else None for i, s in enumerate(steps)]
        else:
            labels = [s[0] for s in steps]
        # create ridgeline plot
        fig, axes = ridgeplot(
            counts,
            bins,
            x_range,
            figsize=(7, 5),
            linecolor="w",
            linewidth=0.5,
            overlap=1,
            labels=labels,
            fade=True,
            title=name,
        )
        # render figure to numpy array
        img = self.figure_to_array(fig)
        img_path = name + ".png"
        ## log as artifact (do not open new thread)
        self.client.log_image(self.run_id, img, img_path)

    def add_weights_and_grads_hooks(
        self,
        module: nn.Module,
        name=None,
        prefix="",
    ):
        """Add forward posthooks to log trainable parameters and backward posthooks to
        log gradients. We will have one paramater hook and multiple gradient hooks."""
        if name is not None:
            prefix = prefix + name

        if self.cfg.log.params:

            # this callable will be called after every forward pass
            def parameter_hook(module, input_, output):
                if not module.training:
                    return
                if self.curr_step % self.cfg.log.train_freq != 0:
                    return
                # iterate through layers to find trainable params to process
                for name, parameter in module.named_parameters():
                    self._process_hooked_tensor(
                        tensor=parameter.cpu(), name="parameters/" + prefix + name
                    )

            # register the forward posthook callable
            hook = module.register_forward_hook(parameter_hook)
            # add parameter hook handle to dict
            self.hook_handles["parameters/" + prefix] = hook

        if self.cfg.log.gradients:
            # only register hooks that require gradients
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    self._add_grad_hook(
                        parameter,
                        "gradients/" + prefix + name,
                    )

    def _add_grad_hook(self, var, name):
        """Register backward posthook on gradient variables"""
        if not isinstance(var, torch.autograd.Variable):
            cls = type(var)
            raise TypeError(
                "Expected torch.Variable, not {}.{}".format(
                    cls.__module__, cls.__name__
                )
            )
        handle = self.hook_handles.get(name, None)
        if handle is not None and self._torch_hook_handle_is_valid(handle):
            raise ValueError(f'A hook has already been set under name "{name}"')

        # this callable will be called when the associated tensor's gradients
        # are to be evaluated
        def grad_hook(grad):
            if self.curr_step % self.cfg.log.train_freq != 0:
                return
            self._process_hooked_tensor(grad.data, name)

        # register the backward posthook callable
        handle = var.register_hook(grad_hook)
        # add parameter hook handle to dict
        self.hook_handles[name] = handle
        return handle

    def _process_hooked_tensor(self, tensor, name):
        """Standardize tensor format, flatten, and send to processing function."""

        # unpack and flatten possible nested tensor tuple
        if isinstance(tensor, tuple) or isinstance(tensor, list):
            while (isinstance(tensor, tuple) or isinstance(tensor, list)) and (
                isinstance(tensor[0], tuple) or isinstance(tensor[0], list)
            ):
                tensor = [item for sublist in tensor for item in sublist]
            tensor = torch.cat([t.reshape(-1) for t in tensor])

        # ensure tensor has shape attribute, flag sparse tensors (not currently handled)
        # half precision tensors on cpu do not support view(), upconvert to 32bit
        if not hasattr(tensor, "shape"):
            cls = type(tensor)
            raise TypeError(f"Expected Tensor, not {cls.__module__}.{cls.__name__}")
        if isinstance(tensor, torch.HalfTensor):
            tensor = tensor.clone().type(torch.float).detach()
        if tensor.is_sparse:
            cls = type(tensor)
            raise TypeError(
                f"Encountered sparse tensor {cls.__module__} {cls.__name__}"
            )

        # flatten tensor
        flat = tensor.reshape(-1)

        # if gpu available, compute using cuda_histc
        if flat.is_cuda:
            if self._is_cuda_histc_supported is None:
                self._is_cuda_histc_supported = True
                check = torch.cuda.FloatTensor(1).fill_(0)
                try:
                    check = flat.histc(bins=self._num_bins)
                except RuntimeError as e:
                    self._is_cuda_histc_supported = False

            if not self._is_cuda_histc_supported:
                flat = flat.cpu().clone().detach()

            # As of torch 1.0.1.post2+nightly, float16 cuda summary ops are not supported
            # (convert to float32)
            if isinstance(flat, torch.cuda.HalfTensor):
                flat = flat.clone().type(torch.cuda.FloatTensor).detach()

        # skip logging if all values are nan or inf or the tensor is empty
        if self._no_finite_values(flat):
            return

        # remove nans and infs if present
        flat = self._remove_infs_nans(flat)

        # compute bin range, adding some extra bin padding
        tmin = flat.min().item()
        tmax = flat.max().item()
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        dt = tmax - tmin
        tmin = tmin - self._extra_bin_frac * dt
        tmax = tmax + self._extra_bin_frac * dt

        # compute histogram
        counts = flat.histc(bins=self._num_bins, min=tmin, max=tmax)
        counts = counts.cpu().clone().detach()
        bins = torch.linspace(tmin, tmax, steps=self._num_bins + 1)

        # store histogram (counts, bins, current step)
        self.histograms[name].append(
            [[counts.numpy()], [bins.numpy()], [self.curr_step]]
        )

    def _torch_hook_handle_is_valid(self, handle):
        """flag hooks that share same name"""
        d = handle.hooks_dict_ref()
        if d is None:
            return False
        else:
            return handle.id in d

    def _no_finite_values(self, tensor: "torch.Tensor") -> bool:
        """determine if all values are nan or inf"""
        q1 = tensor.shape == torch.Size([0])
        q2 = bool((~torch.isfinite(tensor)).all().item())
        return q1 or q2

    def _remove_infs_nans(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """remove nans and infs if present"""
        if not torch.isfinite(tensor).all():
            tensor = tensor[torch.isfinite(tensor)]
        return tensor


if __name__ == "__main__":
    """Test the model"""
    from torch.utils.data import DataLoader
    from dl_schema.dataset import MNISTDataset
    from dl_schema.cfg import TrainConfig
    from dl_schema.models import BabyCNN
    from dl_schema.recorder import Recorder

    cfg = TrainConfig()
    recorder = Recorder(cfg)
    train_data = MNISTDataset(split="train", cfg=cfg)
    train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True)
    x, y = next(iter(train_dataloader))

    model = BabyCNN(cfg.model)
    recorder.add_weights_and_grads_hooks(model)
    y_pred = model(x)
    loss = y_pred.log().mean()
    loss.backward()
    y_pred = model(x)
    loss = y_pred.log().mean()
    loss.backward()
