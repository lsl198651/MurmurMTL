import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import binary_confusion_matrix, \
    binary_accuracy
from transformers import optimization

from utils.util_loss import FocalLoss
from utils.util_train import new_segment_cluster, new_duration_cluster


def train_val(model,
              train_loader,
              val_loader,
              optimizer=None,
              args=None):
    global lr_now, scheduler
    # ========================/ 声明 /========================== #
    error_index_path = r"./error_index/" + str(datetime.now().strftime("%Y-%m%d %H%M"))
    patient_error_index_path = r"./patient_error_index/" + str(datetime.now().strftime("%Y-%m%d %H%M"))
    if not os.path.exists(error_index_path):
        os.makedirs(error_index_path)
    if not os.path.exists(patient_error_index_path):
        os.makedirs(patient_error_index_path)
    tb_writer = SummaryWriter(r"./tensorboard/" + str(datetime.now().strftime("%Y-%m%d %H%M")))
    confusion_matrix_path = r"./confusion_matrix/" + str(datetime.now().strftime("%Y-%m%d %H%M"))
    lr = []
    max_test_acc = []
    max_train_acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = model.to(device)  # 放到设备中
    # for amp
    # ========================/ 学习率设置 /========================== #
    # scaler = GradScaler()
    warm_up_ratio = 0.1
    total_steps = len(train_loader) * args.num_epochs
    if args.scheduler_flag == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    elif args.scheduler_flag == "cos_warmup":
        scheduler = optimization.get_cosine_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=warm_up_ratio * total_steps,
                                                                 num_training_steps=total_steps)
    elif args.scheduler_flag == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler_flag == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90], gamma=0.1)
    # ========================/ 损失函数 /========================== #

    if args.loss_type == "FocalLoss":
        loss_fn = FocalLoss()
    elif args.loss_type == "CE_weighted":
        normed_weights = [1, 5]
        normed_weights = torch.FloatTensor(normed_weights).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=normed_weights)  # 内部会自动加上Softmax层,weight=normedWeights
    else:
        loss_fn = nn.CrossEntropyLoss()
    # ========================/ 训练网络 /========================== #
    for epochs in range(args.num_epochs):
        # train models
        model.train()
        train_loss = 0
        correct_t = 0
        train_len = 0
        input_train = []
        target_train = []
        for input_train, tag_item, label_train, index_train in train_loader:
            input_train, label_train, index_train, tag_item = \
                input_train.to(device), label_train.to(device), index_train.to(device), tag_item.to(device)
            output_train = model(input_train)
            loss_train = loss_fn(output_train, label_train.long())
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            train_loss += loss_train.item()
            # get the index of the max log-probability
            pred_t = output_train.max(1, keepdim=True)[1]
            pred_t = pred_t.squeeze(1)
            input_train.extend(pred_t.cpu().tolist())
            target_train.extend(label_train.cpu().tolist())
            correct_t += pred_t.eq(label_train).sum().item()
            train_len += len(pred_t)
        # ========================/ 调库计算指标  /========================== #
        train_input, train_target = torch.as_tensor(input_train), torch.as_tensor(target_train)
        train_acc = binary_accuracy(train_input, train_target)
        # print(f"train_acc:{train_acc:.2%}")
        # ========================/ 验证网络 /========================== #
        model.eval()
        label = []
        pred = []
        error_index = []
        result_list_present = []
        test_loss = 0
        correct_v = 0
        with (torch.no_grad()):
            for input_val, tag_val, label_val, index_val in val_loader:
                input_val, label_val, index_val, tag_val = \
                    input_val.to(device), label_val.to(device), index_val.to(device), tag_val.to(device)
                optimizer.zero_grad()
                output_val = model(input_val)
                loss_val = loss_fn(output_val, label_val.long())
                # get the index of the max log-probability
                pred_val = output_val.max(1, keepdim=True)[1]
                test_loss += loss_val.item()
                pred_val = pred_val.squeeze(1)
                correct_v += pred_val.eq(label_val).sum().item()
                idx_v = index_val[pred_val.ne(label_val)]
                result_list_present.extend(index_val[pred_val.eq(1)].cpu().tolist())
                try:
                    error_index.extend(idx_v.cpu().tolist())
                except TypeError:
                    logging.ERROR("TypeError: 'int' object is not iterable")
                    # print("TypeError: 'int' object is not iterable")
                pred.extend(pred_val.cpu().tolist())
                label.extend(label_val.cpu().tolist())
        if args.scheduler_flag is not None:
            scheduler.step()
        # ========================/ 调库计算指标  /========================== #
        test_input, test_target = torch.as_tensor(pred), torch.as_tensor(label)
        test_acc = binary_accuracy(test_input, test_target)
        segments_CM = binary_confusion_matrix(test_input, test_target)
        # --------------------------------------------------------
        # pd.DataFrame(error_index).to_csv(error_index_path + "/epoch" + str(epochs + 1) + ".csv",
        #                                  index=False,
        #                                  header=False)
        if args.isSegments:
            result, target = new_segment_cluster(result_list_present, args.test_fold, args.set_name)
        else:
            result, target = new_duration_cluster(result_list_present, args.test_fold, args.set_name)
        test_patient_input, test_patient_target = torch.as_tensor(result), torch.as_tensor(target)
        patient_ACC = binary_accuracy(test_patient_input, test_patient_target)
        patient_CM = binary_confusion_matrix(test_patient_input, test_patient_target)
        # 这两个算出来的都是present的

        "保存最好的模型"
        if patient_ACC > args.fold_best_ACC:
            args.fold_best_ACC = patient_ACC

        for group in optimizer.param_groups:
            lr_now = group["lr"]
        lr.append(lr_now)
        # "更新权值"
        test_loss /= len(pred)
        train_loss /= train_len
        max_train_acc.append(train_acc)
        max_test_acc.append(test_acc)
        max_train_acc_value = max(max_train_acc)
        max_test_acc_value = max_test_acc[max_train_acc.index(max_train_acc_value)]
        # ========================/ tensorboard绘制曲线  /========================== #
        if args.isTensorboard:
            tb_writer.add_scalar("train_acc", train_acc, epochs)
            tb_writer.add_scalar("test_acc", test_acc, epochs)
            tb_writer.add_scalar("train_loss", train_loss, epochs)
            tb_writer.add_scalar("test_loss", test_loss, epochs)
            tb_writer.add_scalar("learning_rate", lr_now, epochs)
            # tb_writer.add_scalar("patient_acc", test_patient_acc, epochs)
        # ========================/ 日志  /========================== #
        logging.info(f"============================")
        logging.info(f"EPOCH: {epochs + 1}/{args.num_epochs}")
        logging.info(f"Learning rate: {lr_now:.1e}")
        logging.info(f"LOSS: train:{train_loss:.2e} verify:{test_loss:.2e}")
        logging.info(f"segments_CM:{segments_CM.numpy()}")
        logging.info(f"ACC: train:{train_acc:.2%} verify:{test_acc:.2%}")
        # logging.info(f"segments_TPR:{segments_TPR:.2%}")
        # logging.info(f"segments_PPV:{segments_PPV:.2%}")
        # logging.info(f"segments_F1:{segments_F1:.2%}")
        # logging.info(f"segments_AUROC:{segments_AUROC:.2%}")
        # logging.info(f"segments_AUPRC:{segments_AUPRC:.2%}")
        logging.info(f"----------------------------")
        logging.info(f"patient_CM:{patient_CM.numpy()}")
        logging.info(f"patient_ACC:{patient_ACC:.2%}")
        # logging.info(f"patient_TPR:{patient_TPR:.2%}")
        # logging.info(f"patient_PPV:{patient_PPV:.2%}")
        # logging.info(f"patient_F1:{patient_F1:.2%}")
        # logging.info(f"patient_AUROC:{patient_AUROC:.2%}")
        # logging.info(f"patient_AUPRC:{patient_AUPRC:.2%}")
        logging.info(f"LR max:{max(lr):.1e} min:{min(lr):.1e}")
        logging.info(f"ACC MAX train:{max_train_acc_value:.2%} verify:{max_test_acc_value:.2%}")
        logging.info(f"best ACC:{args.fold_best_ACC:.2%}")

        # ========================/ 混淆矩阵 /========================== #
        """draw_confusion_matrix(
            test_cm.numpy(),
            ["Absent", "Present"],
            "epoch" + str(epochs + 1) + ",testacc: {:.3%}".format(test_acc),
            pdf_save_path=confusion_matrix_path,
            epoch=epochs + 1
        )"""
