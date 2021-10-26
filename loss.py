import torch
from torch import nn
import torch.nn.functional as F

class DistillLoss(nn.Module):
    def __init__(self, temperature, distillation_weight):
        super().__init__()

        self.temperature = temperature
        self.distillation_weight = distillation_weight
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, labels, outputs_teacher):
        """Compute distillation loss given outputs, labels, and outputs of teacher model

        Arguments:
            outputs {[type]} -- [description]
            labels {[type]} -- [description]
            output_teacher {[type]} -- [description]
        """
        soft_target_loss = 0
        if outputs_teacher is not None and self.distillation_weight > 0:
            soft_target_loss = self.kldiv(
                F.softmax(outputs/self.temperature, dim=1),
                F.softmax(outputs_teacher/self.temperature, dim=1)
                ) * (self.temperature ** 2)
            
        hard_target_loss = F.cross_entropy(outputs, labels, reduction="mean")
        
        total_loss = (soft_target_loss * self.distillation_weight + 
                    hard_target_loss * (1 - self.distillation_weight))

        return soft_target_loss, hard_target_loss, total_loss
