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

    def __init__(self,config,x):
        self.config = config
        self.x = x
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ModelMTL(self.config.get_config_dict())
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get_config_dict()['learning_rate'])
        self.loss_func = nn.BCELoss()
        self.train_loader = DataLoader(self.x, batch_size=self.config.get_config_dict()['batch_size'], shuffle=True)
        self.test_loader = DataLoader(self.x, batch_size=self.config.get_config_dict()['batch_size'], shuffle=False)
        self.train_loss = []
        self.test_loss = []
        self.train_auc = []
        self.test_auc = []

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