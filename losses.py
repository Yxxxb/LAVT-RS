from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(input, target):
    # input: (batch, 2, h, w)
    # target: (batch, h, w)
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return F.cross_entropy(input, target, weight=weight)


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 3:
        raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                         .format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)


# based on: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
class MultiClassDiceLoss(nn.Module):
    def __init__(self) -> None:
        super(MultiClassDiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: (batch, 2, h, w)
        # target: (batch, h, w)
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}, {}" .format(
                    input.device, target.device))

        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)  # (b, c)
        cardinality = torch.sum(input_soft * input_soft + target_one_hot, dims)  # (b, c)

        dice_score = 2. * intersection / (cardinality + self.eps)  # (b, c)
        dice_loss = 1. - dice_score  # (b, c)
        dice_loss = torch.mean(dice_loss, dim=0)  # (c); c == 2
        # [1] is foreground scores; [0] is background scores
        dice_loss = (dice_loss[1] + dice_loss[0]) / 2

        return dice_loss
    

class DiceFocalLoss(nn.Module):
    def __init__(self, focal_rate=3, dice_rate=1) -> None:
        super(DiceFocalLoss, self).__init__()
        self.eps: float = 1e-6
        self.focal_rate = focal_rate
        self.dice_rate = dice_rate

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: (batch, 2, h, w)
        # target: (batch, h, w)
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}, {}" .format(
                    input.device, target.device))

        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)  # (b, c)
        cardinality = torch.sum(input_soft * input_soft + target_one_hot, dims)  # (b, c)

        dice_score = 2. * intersection / (cardinality + self.eps)  # (b, c)
        dice_loss = 1. - dice_score  # (b, c)
        dice_loss = torch.mean(dice_loss, dim=0)  # (c); c == 2
        # [1] is foreground scores; [0] is background scores
        dice_loss = (dice_loss[1] + dice_loss[0]) / 2

        alpha = 0.25
        gamma = 2.0
        eps = 1e-5

        # Compute Binary Mask Focal Loss
        pt = (input_soft * target_one_hot + (1 - input_soft) * (1 - target_one_hot))
        # pt = target_one_hot[:, 1] * input_soft[:, 1] + target_one_hot[:, 0] * input_soft[:, 0]

        focal_weight = alpha * (1 - pt).pow(gamma)
        # pt: 16 80 80; tgt: 16 2 80 80
        binary_mask_focal_loss = -focal_weight * (target_one_hot * torch.log(pt + eps) + (1 - target_one_hot) * torch.log(1 - pt + eps))
        binary_mask_focal_loss = binary_mask_focal_loss.mean()

        # print("Dice: ", dice_loss)
        # print("Focal: ", binary_mask_focal_loss * self.focal_rate)
        # Calculate the average of Dice Loss and Binary Mask Focal Loss
        combined_loss = (dice_loss * self.dice_rate + binary_mask_focal_loss * self.focal_rate)

        return combined_loss
    

class DiceBoundaryLoss(nn.Module):
    def __init__(self, boundary_rate=0.05, dice_rate=1) -> None:
        super(DiceBoundaryLoss, self).__init__()
        self.eps: float = 1e-6
        self.BoundaryLoss = BoundaryLoss()
        self.boundary_rate = boundary_rate
        self.dice_rate = dice_rate

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: (batch, 2, h, w)
        # target: (batch, h, w)
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}, {}" .format(
                    input.device, target.device))

        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)  # (b, c)
        cardinality = torch.sum(input_soft * input_soft + target_one_hot, dims)  # (b, c)

        dice_score = 2. * intersection / (cardinality + self.eps)  # (b, c)
        dice_loss = 1. - dice_score  # (b, c)
        dice_loss = torch.mean(dice_loss, dim=0)  # (c); c == 2
        dice_loss = (dice_loss[1] + dice_loss[0]) / 2

        # Boundary loss 
        boundary_loss = self.BoundaryLoss(input_soft, target_one_hot)

        combined_loss = (dice_loss * self.dice_rate + boundary_loss * self.boundary_rate)

        return combined_loss


class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape
        one_hot_gt = gt

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss