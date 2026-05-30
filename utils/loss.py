import torch
import torch.nn as nn
import torch.nn.functional as F

# 仅计算毛孔类别的分割loss
class PoreDiceLoss(nn.Module):
    def __init__(self, pore_class_idx, smooth=1):
        super(PoreDiceLoss, self).__init__()
        self.pore_class_idx = pore_class_idx  # 毛孔类别索引
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = F.softmax(preds, dim=1) # 二分类时使用 Sigmoid
        
        # 只计算毛孔类别的 Dice Loss
        pred_flat = preds[:, self.pore_class_idx].contiguous().view(-1)
        target_flat = (targets == self.pore_class_idx).float().contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice
        
class CombinedLoss(nn.Module):
    def __init__(self, pore_class_idx, alpha=1.0, beta=1.0):
        super(CombinedLoss, self).__init__()
        # self.ce_loss = nn.CrossEntropyLoss()  
        self.dice_loss = PoreDiceLoss(pore_class_idx)  # 只针对毛孔类别
        self.cls_loss = nn.CrossEntropyLoss() # 分类用交叉熵
        self.alpha = alpha
        self.beta = beta

    def forward(self, seg_preds, seg_targets, cls_preds, cls_targets):
        # **只计算毛孔类别的 Dice 损失**
        dice_loss = self.dice_loss(seg_preds, seg_targets)
        
        # 分类损失（交叉熵）
        cls_loss = self.cls_loss(cls_preds, cls_targets)

        # 总损失 = α * 分割损失 + β * 分类损失
        # print(f"cls loss:{cls_loss}, seg pore loss:{dice_loss}")
        return self.alpha * dice_loss + self.beta * cls_loss