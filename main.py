# -*- coding: utf-8 -*-
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch.profiler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from models import supernet
from run_MTL import VAR_DICT
from trainAndTest.train_eval import train_val
from utils.util_dataloader import fold5_dataloader
from utils.util_datasetClass import DatasetMTL
from utils.util_train import logger_init

sys.dont_write_bytecode = True
PROJ_PATH = Path(__file__).parent.parent.as_posix()
sys.path.append(PROJ_PATH)
# print(sys.path)

from config.Config import Config

if __name__ == '__main__':
    # ========================/ 函数入口 /========================== #
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=512, help="args.batch_size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="num_epochs")
    parser.add_argument("--num_layers", type=int, default=3, help="layers number")
    parser.add_argument("--freqm_value", type=int, default=0, help="frequency mask max length")
    parser.add_argument("--timem_value", type=int, default=0, help="time mask max length")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="learning_rate for training")
    parser.add_argument("--ap_ratio", type=float, default=1.0, help="ratio of absent and present")
    parser.add_argument("--beta", type=float, default=(0.9, 0.98), help="beta")
    parser.add_argument("--loss_type", type=str, default="FocalLoss", help="loss function", choices=["CE", "FocalLoss"])
    parser.add_argument("--scheduler_flag", type=str, default=None, help="the dataset used",
                        choices=["cos", "MultiStepLR", "step"])
    parser.add_argument("--mask", type=bool, default=False, help="number of classes", choices=[True, False])
    parser.add_argument("--train_set_balance", type=bool, default=False, help="balance absent and present in testset",
                        choices=[True, False])
    parser.add_argument("--data_augmentation", type=bool, default=False, help="Add data augmentation",
                        choices=[True, False])
    parser.add_argument("--train_total", type=bool, default=True, help="use grad_no_required", choices=[True, False], )
    parser.add_argument("--samplerWeight", type=bool, default=True, help="use balanced sampler",
                        choices=[True, False])
    parser.add_argument("--cross_eevalue", type=bool, default=False)
    parser.add_argument("--set_path", type=str, default=r"E:\Shilong\02_dataset")
    parser.add_argument("--train_fold", default=['0', '1', '2', '3'])
    parser.add_argument("--test_fold", default=['4'])
    parser.add_argument("--fold_res", default=[])
    parser.add_argument("--fold_best_ACC", default=0)
    parser.add_argument("--set_name", type=str, default=r"\01_5s_4k_txt")
    parser.add_argument("--model_folder", type=str, default=r"E:\Shilong\01_Code\MurmurMTL\models\MyModels")
    parser.add_argument("--isTensorboard", type=bool, default=False)
    parser.add_argument("--isSegments", type=bool, default=True)
    parser.add_argument("--saveModel", type=bool, default=True)
    parser.add_argument("--isTry", type=bool, default=True)
    # TODO 改模型名字
    parser.add_argument("--desperation", type=str, default="SuperNet_5s_4k_5fold MTL debug")
    args = parser.parse_args()
    config = Config(r"E:\Shilong\01_Code\MurmurMTL\config\default_nas.yaml",VAR_DICT ).get_config_dict()

    all_list = ['0', '1', '2', '3', '4']
    # ========================/ 加载数据集 /========================== #
    train_mel, train_label, train_index, train_tag, val_mel, value_label, val_index, val_tag = fold5_dataloader(
        args.set_path, args.train_fold, args.test_fold, args.data_augmentation, args.set_name)
    # todo 补充加入的嵌入数据，train_embs=[],test_embs=[]
    train_embs = 0
    test_embs = 0
    # ========================/ setup loader /========================== #
    if args.samplerWeight:
        weights = [5 if label == 1 else 1 for label in train_label]
        Data_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(DatasetMTL(features=train_mel,
                                             wav_label=train_label,
                                             wav_tag=train_tag,
                                             wav_index=train_index),
                                  sampler=Data_sampler,
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=4)
    else:
        train_loader = DataLoader(DatasetMTL(features=train_mel,
                                             wav_label=train_label,
                                             wav_tag=train_tag,
                                             wav_index=train_index
                                             ),
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=4)

    val_loader = DataLoader(DatasetMTL(wav_label=value_label,
                                       features=val_mel,
                                       wav_tag=val_tag,
                                       wav_index=val_index),
                            batch_size=args.batch_size // 4,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=4)

    # ========================/ 选择模型 /========================== #
    MyModel = supernet.SuperNet(embedding_dim=config["embedding_dim"],
                                task_types=config["task_types"], n_experts=config["models"]["kwargs"]["n_experts"],
                                n_expert_layers=config["models"]["kwargs"]["n_expert_layers"],
                                n_layers=config["models"]["kwargs"]["expert_module"]["n_layers"],
                                in_features=config["models"]["kwargs"]["expert_module"]["in_features"],
                                out_features=config["models"]["kwargs"]["expert_module"]["out_features"],
                                tower_layers=config["models"]["kwargs"]["tower_layers"],
                                dropout=config["models"]["kwargs"]["dropout"],
                                expert_candidate_ops=config["models"]["kwargs"]["expert_module"]["ops"])

    # ========================/ 打印日志 /========================== #
    if args.isTry:
        args.num_epochs = 4
    else:
        logger_init()
    logging.info(f"# {args.desperation}")
    logging.info(f"# Batch_size = {args.batch_size}")
    logging.info(f"# Num_epochs = {args.num_epochs}")
    logging.info(f"# Learning_rate = {args.learning_rate:.1e}")
    logging.info(f"# lr_scheduler = {args.scheduler_flag}")
    logging.info(f"# Loss_fn = {args.loss_type}")
    logging.info(f"# Set name = {args.set_name}")

    # ========================/ 检测分折重复 /========================== #
    for fold in all_list:
        if args.isTry:
            train_fold = '0'
            test_fold = '1'
        else:
            all_list = ['0', '1', '2', '3', '4']
            all_list.remove(fold)
            args.train_fold = all_list
            args.test_fold = list(fold)
        # ========================/ 设置优化器 /========================== #
        if not args.train_total:
            for param in MyModel.BEATs.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, MyModel.parameters()),
                                          lr=args.learning_rate)
        else:
            optimizer = torch.optim.AdamW(MyModel.parameters(),
                                          lr=args.learning_rate)

        # ========================/ 计算数据集大小 /========================== #
        train_present_size = np.sum(train_label == 1)
        train_absent_size = np.sum(train_label == 0)
        train_set_size = train_label.shape[0]
        test_present_size = np.sum(value_label == 1)
        test_absent_size = np.sum(value_label == 0)
        test_set_size = value_label.shape[0]

        # ========================/ 打印每折日志 /========================== #
        logging.info("\n**************************************")
        logging.info(f"# Train_a/p = {train_absent_size}/{train_present_size}")
        logging.info(f"# Test_a/p = {test_absent_size}/{test_present_size}")
        logging.info(f"# Train set size = {train_set_size}")
        logging.info(f"# Testnet size = {test_set_size}")
        logging.info(f"# Train_fold = {args.train_fold}")
        logging.info(f"# Test_fold = {args.test_fold}")
        logging.info("# Optimizer = " + str(optimizer))

        # ========================/ 开始训练 /========================== #
        train_val(model=MyModel,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  optimizer=optimizer,
                  args=args)
        # ========================/ 五折均值 /========================== #
        args.fold_res.append(args.fold_best_ACC)
        args.fold_best_ACC = 0
    logging.info(fr"# Average Fold ACC = {np.mean(args.fold_res):.2%}")
