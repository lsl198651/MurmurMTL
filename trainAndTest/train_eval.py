import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = model.to(device)  # 放到设备中
    alpha = 0.5
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

        correct_t = 0
        train_len = 0
        output_train_list = []
        target_train_list = []
        for input_train, tag_train, label_train, index_train in train_loader:
            input_train, label_train, index_train, tag_train = \
                input_train.to(device), label_train.to(device), index_train.to(device), tag_train.to(device)
            output_train = model(input_train)
            output_murmur = output_train[0]
            output_segment = output_train[1]
            train_loss_murmur = loss_fn_murmur(output_murmur, label_train.float())  # TODO tag_train 要float32
            train_loss_segment = loss_fn_segment(output_segment, tag_train)  # TODO tag_train 要float32
            train_loss = alpha * train_loss_murmur + (1 - alpha) * train_loss_segment
            train_total_loss_murmur += train_loss_murmur

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # train_loss += train_loss.item()

            # get the index of the max log-probability
            pred_train = [1 if murmur > 0.5 else 0 for murmur in output_murmur]

            # pred_t = output_murmur.max(1, keepdim=True)[1]
            pred_t = torch.tensor(pred_train).to(device)
            output_train_list.extend(pred_train)
            target_train_list.extend(label_train.cpu().tolist())
            correct_t += pred_t.eq(label_train).sum().item()
            train_len += len(pred_train)
        # ========================/ 调库计算指标  /========================== #
        train_input, train_target = torch.as_tensor(output_train_list), torch.as_tensor(target_train_list)
        train_acc = binary_accuracy(train_input, train_target)
        # print(f"train_acc:{train_acc:.2%}")
        # ========================/ 验证网络 /========================== #
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

                output_val_murmur = output_val[0]
                output_val_segment = output_val[1]
                loss_val_murmur = loss_fn_murmur(output_val_murmur, label_val.float())
                loss_val_segment = loss_fn_segment(output_val_segment, tag_val)
                val_loss = alpha * loss_val_murmur + (1 - alpha) * loss_val_segment

                # get the index of the max log-probability
                pred_val = [1 if murmur > 0.5 else 0 for murmur in output_val_murmur]
                # pred_val = output_val_murmur.max(1, keepdim=True)[1]
                # val_loss += (loss_val_murmur + loss_val_segment).item()
                # pred_val = pred_val.squeeze(1)
                pred_v = torch.tensor(pred_val).to(device)
                correct_v += pred_v.eq(label_val).sum().item()
                idx_v = index_val[pred_v.ne(label_val)]
                result_list_present.extend(pred_v[pred_v.eq(1)].cpu().tolist())
                try:
                    error_index.extend(idx_v.cpu().tolist())
                except TypeError:
                    logging.ERROR("TypeError: 'int' object is not iterable")
                    # print("TypeError: 'int' object is not iterable")
                pred.extend(pred_val)
                label.extend(label_val.cpu().tolist())
        if args.scheduler_flag is not None:
            scheduler.step()
        # ========================/ 调库计算指标  /========================== #
        segments_CM = confusion_matrix(label, pred)
        test_input, test_target = torch.as_tensor(pred), torch.as_tensor(label)
        test_acc = binary_accuracy(test_input, test_target)
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
        max_train_acc.append(train_acc)
        max_test_acc.append(test_acc)
        max_train_acc_value = max(max_train_acc)
        max_test_acc_value = max_test_acc[max_train_acc.index(max_train_acc_value)]
        # ========================/ tensorboard绘制曲线  /========================== #
        if args.isTensorboard:
            tb_writer = SummaryWriter(r"./tensorboard/" + str(datetime.now().strftime("%Y-%m%d %H%M")))
            tb_writer.add_scalar("train_acc", train_acc, epochs)
            tb_writer.add_scalar("test_acc", test_acc, epochs)
            tb_writer.add_scalar("train_loss", train_total_loss_murmur, epochs)
            tb_writer.add_scalar("test_loss", val_loss, epochs)
            tb_writer.add_scalar("learning_rate", lr_now, epochs)
            # tb_writer.add_scalar("patient_acc", test_patient_acc, epochs)
        # ========================/ 日志  /========================== #
        logging.info(f"============================")
        logging.info(f"EPOCH: {epochs + 1}/{args.num_epochs}")
        logging.info(f"Learning rate: {lr_now:.1e}")
        logging.info(f"LOSS: train:{train_total_loss_murmur:.2e} verify:{val_loss:.2e}")
        logging.info(f"segments_CM:{segments_CM[0]}")
        logging.info(f"            {segments_CM[1]}")
        logging.info(f"ACC: train:{train_acc:.2%} verify:{test_acc:.2%}")
        # logging.info(f"segments_TPR:{segments_TPR:.2%}")
        # logging.info(f"segments_PPV:{segments_PPV:.2%}")
        # logging.info(f"segments_F1:{segments_F1:.2%}")
        # logging.info(f"segments_AUROC:{segments_AUROC:.2%}")
        # logging.info(f"segments_AUPRC:{segments_AUPRC:.2%}")
        logging.info(f"----------------------------")
        logging.info(f"patient_CM:{patient_CM[0]}")
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
