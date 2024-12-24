import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
from numpy import dtype
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import binary_accuracy
from transformers import optimization

from utils.util_loss import FocalLoss
from utils.util_train import new_segment_cluster, new_duration_cluster


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

    confusion_matrix_path = r"./confusion_matrix/" + str(datetime.now().strftime("%Y-%m%d %H%M"))
    lr = []
    max_test_acc = []
    max_train_acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 保证CuDNN的确定性
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    model = model.to(device)  # 放到设备中
    alpha = args.aplha
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
    loss_fn_murmur = FocalLoss()
    loss_fn_segment = FocalLoss()
    if args.loss_type == "FocalLoss":
        pass
    elif args.loss_type == "CE_weighted":
        normed_weights = [1, 5]
        normed_weights = torch.FloatTensor(normed_weights).to(device)
        loss_fn_murmur = nn.CrossEntropyLoss(weight=normed_weights)  # 内部会自动加上Softmax层,weight=normedWeights
        loss_fn_segment = nn.CrossEntropyLoss(weight=normed_weights)  # 内部会自动加上Softmax层,weight=normedWeights
    else:
        loss_fn_murmur = nn.CrossEntropyLoss()
        loss_fn_segment = nn.CrossEntropyLoss()
    # ========================/ 训练网络 /========================== #
    for epochs in range(args.num_epochs):
        # train models
        model.train()
        train_total_loss_murmur = 0
        train_acc_seg = 0
        train_acc_murmur = 0

        correct_t = 0
        train_len = 0

        for input_train, tag_train, label_train, index_train in train_loader:
            input_train, label_train, index_train, tag_train = \
                input_train.to(device), label_train.to(device), index_train.to(device), tag_train.to(device)
            output_train = model(input_train)


            train_loss_murmur = loss_fn_murmur(output_train[0], label_train.long())  # TODO tag_train 要float32
            train_loss_segment = loss_fn_segment(output_train[1], tag_train.long())  # TODO tag_train 要float32

            train_loss = alpha * train_loss_murmur + (1 - alpha) * train_loss_segment
            train_total_loss_murmur += train_loss_murmur

            # optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # train_loss += train_loss.item()
            output_murmur = torch.argmax(output_train[0], dim=1)
            output_segment = torch.argmax(output_train[1], dim=1)
            # # get the index of the max log-probability

            # ========================/ 计算ACC指标  /========================== #
            train_acc_murmur = torch.mean((label_train == output_murmur).float())
            train_acc_seg = torch.mean((tag_train == output_segment).float())

            correct_t += label_train.eq(output_murmur).sum().item()
            train_len += len(label_train)
        # ========================/ 验证模型 /========================== #
        model.eval()
        label = []
        pred = []
        error_index = []
        result_list_present = []
        val_loss = 0
        correct_v = 0
        with (torch.no_grad()):
            for input_val, tag_val, label_val, index_val in val_loader:
                input_val, label_val, index_val, tag_val = \
                    input_val.to(device), label_val.to(device), index_val.to(device), tag_val.to(device)
                optimizer.zero_grad()
                output_val = model(input_val)

                loss_val_murmur = loss_fn_murmur(output_val[0], label_val.long())
                loss_val_segment = loss_fn_segment(output_val[1], tag_val.long())

                val_loss = alpha * loss_val_murmur + (1 - alpha) * loss_val_segment

                # get the index of the max log-probability

                output_val_murmur = torch.argmax(output_val[0], dim=1)
                output_val_segment = torch.argmax(output_val[1], dim=1)

                val_acc_murmur = torch.mean((label_val == output_val_murmur).float())
                val_acc_seg = torch.mean((tag_val == output_val_segment).float())

                correct_v += val_acc_murmur.eq(label_val).sum().item()
                idx_v = index_val[label_val.ne(output_val_murmur)]
                result_list_present.extend(index_val[output_val_murmur.eq(1)].cpu().tolist())
                try:
                    error_index.extend(idx_v.cpu().tolist())
                except TypeError:
                    logging.ERROR("TypeError: 'int' object is not iterable")
                    # print("TypeError: 'int' object is not iterable")
                pred.extend(output_val_murmur.cpu().tolist())
                label.extend(label_val.cpu().tolist())
        if args.scheduler_flag is not None:
            scheduler.step()
        # ========================/ 调库计算指标  /========================== #
        segments_CM = confusion_matrix(label, pred)
        test_input, test_target = torch.as_tensor(pred), torch.as_tensor(label)
        # test_acc = binary_accuracy(test_input, test_target)
        # segments_CM = binary_confusion_matrix(test_input, test_target)
        # --------------------------------------------------------
        # pd.DataFrame(error_index).to_csv(error_index_path + "/epoch" + str(epochs + 1) + ".csv",
        #                                  index=False,
        #                                  header=False)
        if args.isSegments:
            result, target = new_segment_cluster(result_list_present, args.test_fold, args.set_path + args.set_name)
        else:
            result, target = new_duration_cluster(result_list_present, args.test_fold, args.set_path + args.set_name)
        test_patient_input, test_patient_target = torch.as_tensor(result), torch.as_tensor(target)
        patient_ACC = binary_accuracy(test_patient_input, test_patient_target)
        patient_CM = confusion_matrix(target, result)
        # 这两个算出来的都是present的

        "保存最好的模型"
        if patient_ACC > args.fold_best_ACC:
            args.fold_best_ACC = patient_ACC

        for group in optimizer.param_groups:
            lr_now = group["lr"]
        lr.append(lr_now)
        # "更新权值"
        val_loss /= len(pred)
        train_total_loss_murmur /= train_len
        max_train_acc.append(train_acc_murmur)
        max_test_acc.append(val_acc_murmur)
        max_train_acc_value = max(max_train_acc)
        max_test_acc_value = max_test_acc[max_train_acc.index(max_train_acc_value)]
        # ========================/ tensorboard绘制曲线  /========================== #
        if args.isTensorboard:
            tb_writer = SummaryWriter(r"./tensorboard/" + str(datetime.now().strftime("%Y-%m%d %H%M")))
            tb_writer.add_scalar("train_acc", train_acc_murmur, epochs)
            tb_writer.add_scalar("test_acc", val_acc_murmur, epochs)
            tb_writer.add_scalar("train_loss", train_total_loss_murmur, epochs)
            tb_writer.add_scalar("test_loss", val_loss, epochs)
            tb_writer.add_scalar("learning_rate", lr_now, epochs)
            # tb_writer.add_scalar("patient_acc", test_patient_acc, epochs)
        # ========================/ 日志  /========================== #
        logging.info(f"============================")
        logging.info(f"EPOCH: {epochs + 1}/{args.num_epochs}")
        logging.info(f"Learning rate: {lr_now:.1e}")
        logging.info(f"LOSS: train:{train_total_loss_murmur:.2e} verify:{val_loss:.2e}")
        logging.info(f"murmur_CM:{segments_CM[0]}")
        logging.info(f"            {segments_CM[1]}")
        logging.info(f"murmur_ACC: train:{train_acc_murmur:.2%} verify:{val_acc_murmur:.2%}")
        logging.info(f"cut_ACC:    train:{train_acc_seg:.2%} verify:{val_acc_seg:.2%}")
        # logging.info(f"segments_TPR:{segments_TPR:.2%}")
        # logging.info(f"segments_PPV:{segments_PPV:.2%}")
        # logging.info(f"segments_F1:{segments_F1:.2%}")
        # logging.info(f"segments_AUROC:{segments_AUROC:.2%}")
        # logging.info(f"segments_AUPRC:{segments_AUPRC:.2%}")
        logging.info(f"----------------------------")
        logging.info(f"murmur_patient_CM:{patient_CM[0]}")
        logging.info(f"           {patient_CM[1]}")
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
