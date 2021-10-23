import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import datasets, transforms 
import pytorch_lightning as plt
import wandb
from loss import DistillLoss

class TeacherModel(ResNet):
    def __init__(self):
        """Resnet34 acts as a teacher model
        """
        super(TeacherModel, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) #ResNet34

    def forward(self, batch):
        logits = super(TeacherModel, self).forward(batch)
        return logits

class StudentModel(ResNet):
    def __init__(self):
        """Resnet18 acts as a student model
        """
        super(StudentModel, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) #ResNet18
        
    def forward(self, batch):
        logits = super(StudentModel, self).forward(batch)
        return logits

class TeacherTrainingModule(plt.LightningModule):
    def __init__(self, lr=1e-3, num_classes=10):
        super(TeacherTrainingModule, self).__init__()
        self.save_hyperparameters()

        self.model = TeacherModel()
        self.num_classes = num_classes
        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, images):
        return self.model(images)
        
    def training_step(self, batch, batch_idx):
        images, labels = batch 
        outputs = self(images)
        loss = self.loss_function(outputs, labels).mean()
        
        logits = torch.sigmoid(outputs)
        preds = torch.argmax(logits, 1)

        train_accuracy = self.train_accuracy_metric(preds, labels)
        self.log('train/loss', loss, prog_bar=True, on_epoch=True)
        self.log('train/acc', train_accuracy, prog_bar=True, on_epoch=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        images, labels = batch 
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels).mean()
        
        logits = torch.sigmoid(outputs)
        preds = torch.argmax(logits, 1)
        #metrics
        valid_accurracy = self.val_accuracy_metric(preds, labels)
        f1_score = self.f1_metric(preds, labels)

        #logging metrics
        self.log('valid/loss', loss.item(), prog_bar=True, on_step=True)
        self.log('valid/acc', valid_accurracy, prog_bar=True, on_epoch=True)
        self.log('valid/f1_score', f1_score, prog_bar=True, on_epoch=True)

        return {"labels": labels, "logits": logits}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)

        # Confusion matrix plotting using inbuilt W&B method
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.cpu().numpy(), y_true=labels.cpu().numpy()
                )
            }
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

class DistillTrainingModule(TeacherTrainingModule):
    def __init__(self, teacher_model, 
                temperature=0.6, distillation_weight=0.5,
                lr=0.01, num_classes=10):
        super(DistillTrainingModule, self).__init__(lr=lr, num_classes=num_classes)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.model = StudentModel()        
        self.loss_function = DistillLoss(temperature=temperature, 
                                         distillation_weight=distillation_weight)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs_student = self.forward(images)
        with torch.no_grad():
            outputs_teacher = self.teacher_model(images)
        soft_target_loss, hard_target_loss, loss = self.loss_function(
            outputs_student, labels, outputs_teacher)

        logits = torch.sigmoid(outputs_student)
        preds = torch.argmax(logits, 1)

        train_accuracy = self.train_accuracy_metric(preds, labels)

        self.log('distill_train/total_loss', loss, prog_bar=True, on_epoch=True)
        self.log('distill_train/soft_loss', soft_target_loss, prog_bar=True, on_epoch=True)
        self.log('distill_train/hard_loss', hard_target_loss, prog_bar=True, on_epoch=True)
        self.log('distill_train/acc', train_accuracy, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch 
        outputs = self.forward(images)
        loss = F.cross_entropy(outputs, labels, reduction='mean')
        
        logits = torch.sigmoid(outputs)
        preds = torch.argmax(logits, 1)
        #metrics
        valid_accurracy = self.val_accuracy_metric(preds, labels)
        f1_score = self.f1_metric(preds, labels)

        #logging metrics
        self.log('distill_valid/loss', loss.item(), prog_bar=True, on_step=True)
        self.log('distill_valid/acc', valid_accurracy, prog_bar=True, on_epoch=True)
        self.log('distill_valid/f1_score', f1_score, prog_bar=True, on_epoch=True)

        return {"labels": labels, "logits": logits} 
