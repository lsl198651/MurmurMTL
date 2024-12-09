import builtins
import copy
import datetime
import json
import os
import sys
from collections import OrderedDict
from time import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from numpy.typing import NDArray

from src.datasets.dataset_utils import DataGenerator
from src.models.basic.features import DenseFeature, SparseFeature
from src.models.nas import SuperNet
from src.utils.utils import (
    get_loss_func, get_metric_func, get_instance, get_local_time, create_dirs, init_seed,
    GradualWarmupScheduler, SaveType, TensorboardWriter,
)

class MurmurMTL:

    def __init__(self, config, rank=0):
        self.rank = rank
        self.config = config
        self.config["rank"] = rank
        self.distribute = config["n_gpu"] > 1
        self.task_num = len(config["task_types"])
        self.loss_fns = [get_loss_func(task_type) for task_type in config["task_types"]]
        self.evaluate_fns = [get_metric_func(task_type) for task_type in config["task_types"]]
        self.early_stop_patience = config["earlystop_patience"]
        self.early_stop_counter = 0
        self.val_per_epoch = config["val_per_epoch"]
        self.full_arch_train_epoch_idx = config["warmup_epochs"]
        self.fine_tune_epoch_idx = config["warmup_epochs"] + config["full_arch_train_epochs"]
        (
            self.result_path,
            self.log_path,
            self.ckpt_path,
            self.viz_path,  # tensorboard path
        ) = self._init_files(config)
        (
            self.device,
            self.list_ids,
        ) = self._init_device(rank, config)
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.features,
        ) = self._init_dataloader(config)
        (
            self.net,
            self.best_model_weights,
        ) = self._init_model(config)
        (
            self.optimizer,
            self.arch_optimizer,
            self.scheduler,
            self.from_epoch,
            self.best_val_auc,
            # self.best_test_auc,
        ) = self._init_optim(config)
        self.logger = self._init_logger()
        print(config)

    def fit(self, ):
        """ The normal train loop: train loop: train-val and save model val-cc increases
        """
        print(
            f"#alpha_params: {len(list(self.net.alpha_parameters()))}\t"
            f"#beta_params: {len(list(self.net.beta_parameters()))}\t"
            f"#weight_params: {len(list(self.net.weight_parameters()))}\t"
        )
        experiment_begin = time()
        for epoch_idx in range(self.from_epoch + 1, self.config["epoch"]):
            print("============ Train on the train set ============")
            print("{}, learning rate: {}".format(
                "Warm up" if epoch_idx < self.full_arch_train_epoch_idx
                else "{}".format(
                    "Full arch train" if epoch_idx < self.fine_tune_epoch_idx
                    else "Fine tune"
                ),
                self.scheduler.get_last_lr()
            ))
            self._train_one_epoch(epoch_idx)  # train one epoch

            # print current network architecture
            self.net.print_arch(epoch_idx)
            # discretize
            if epoch_idx >= self.full_arch_train_epoch_idx:
                for _ in range(self.config["discretize_ops"]):
                    self.net.discretize_one_op()
            if epoch_idx == self.fine_tune_epoch_idx:
                self.net.export_architecture()
                # net re-init
                # self.net.init_model()
                # self.optimizer = get_instance(
                #     torch.optim, "optimizer", self.config, params=self.net.parameters(),
                # )
                self.best_val_auc = np.zeros(self.task_num)
                self.best_model_weights = copy.deepcopy(self.net.state_dict())

            if ((epoch_idx + 1) % self.val_per_epoch) == 0:
                print("============ Validation on the val set ============")
                val_auc = self._validate(epoch_idx=epoch_idx, is_test=False)

                if epoch_idx >= self.full_arch_train_epoch_idx:  # early stop for fine tune
                    if self._compute_improvement(val_auc, self.best_val_auc) > 0:
                        self.best_val_auc = val_auc
                        self.early_stop_counter = 0
                        self.best_model_weights = copy.deepcopy(self.net.state_dict())
                        self._save_model(epoch_idx, SaveType.BEST)
                    elif self.early_stop_counter < self.early_stop_patience:
                        self.early_stop_counter += 1
                    else:  # early stop
                        print(" * Early stopping, best_val_auc: {}".format(
                            " ".join([
                                "Task#{}-({:.5f})".format(i, self.best_val_auc[i])
                                for i in range(self.task_num)
                            ])))
                        self.net.load_state_dict(self.best_model_weights)
                        self._save_model(epoch_idx, SaveType.LAST)
                        break
                self._save_model(epoch_idx, SaveType.LAST)
                print(" * Best Auc: {}".format(
                    " ".join([
                        "Task#{}-({:.5f})".format(i, self.best_val_auc[i])
                        for i in range(self.task_num)
                    ])))

            time_scheduler = self._cal_time_scheduler(experiment_begin, epoch_idx)
            print(" * Time: {}".format(time_scheduler))
            self.scheduler.step()

        print(
            "End of experiment, took {}".format(
                str(datetime.timedelta(seconds=int(time() - experiment_begin)))
            ))
        print("Result DIR: {}".format(self.result_path))