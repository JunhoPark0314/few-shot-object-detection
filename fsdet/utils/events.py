# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional
import torch
from fvcore.common.history_buffer import HistoryBuffer
from detectron2.utils.events import EventWriter, get_event_storage

from detectron2.utils.file_io import PathManager

__all__ = [
    "FasterRCNNWriter"
]
class FasterRCNNWriter(EventWriter):
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    It also applies smoothing using a window of 20 elements.

    It's meant to print common metrics in common ways.
    To print something in more customized ways, please implement a similar printer by yourself.
    """

    def __init__(self, max_iter: Optional[int] = None, window_size: int = 20):
        """
        Args:
            max_iter: the maximum number of iterations to train.
                Used to compute ETA. If not given, ETA will not be printed.
            window_size (int): the losses will be median-smoothed by this window size
        """
        self.logger = logging.getLogger(__name__)
        self._max_iter = max_iter
        self._window_size = window_size
        self._last_write = None  # (step, time) of last call to write(). Used to compute ETA

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        common = ["lr", "loss", "eta_seconds", "time"]

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            "{logs}".format(
                logs="  ".join(
                    [
                        "{}: {:.4g}".format(k, v.median(self._window_size))
                        for k, v in storage.histories().items()
                        if not any(x in k for x in common)
                    ]
                ),
            )
        )