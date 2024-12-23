import torch
import torch.nn as nn
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """Focal Loss"""

    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = torch.log(torch.softmax(input, dim=1))  #
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.features.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.features.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BCEFocalLoss(nn.Module):
    """BiFocal Loss"""

    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)  # sigmoide获取概率
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        self.gamma = self.gamma.view(target.size)
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (
                1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class FocalLoss_VGG(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True, weight=None):
        super(FocalLoss_VGG, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.weight = weight
        # if isinstance(alpha, list):
        #     assert len(alpha) == num_classes
        #     self.alpha = torch.Tensor(alpha)
        # else:
        #     assert alpha < 1
        #     self.alpha = torch.zeros(num_classes)
        #     self.alpha[0] += alpha
        #     self.alpha[1:] += (1 - alpha)

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                torch.argmax(inputs, dim=1).float(), targets, weight=self.weight, reduce=False)
        else:
            CE_loss = nn.CrossEntropyLoss(inputs, targets, reduce=True)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
