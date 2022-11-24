"""User defined logging and analytics utility for training runs"""
import logging

from dl_schema.base.recorder_base import RecorderBase

logger = logging.getLogger(__name__)


class Recorder(RecorderBase):
    """Artifact, metric, parameter, and image logger. Define custom analytic logic here."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def custom_figure_creation(self):
        raise NotImplementedError

    def custom_metric_computations(self):
        raise NotImplementedError
