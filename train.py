import os
import torch
import hydra
import wandb
import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import DistillTrainingModule, TeacherTrainingModule

logger = logging.getLogger(__name__)

class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    # def on_train_start(self, trainer, pl_module):
    #     data_split_at = wandb.Artifact("data_split", type='balanced_data')
    #     preview_data_table = wandb.Table(columns=['image', 'label', 'split'])
    #     split_data = {
    #         'train' : self.datamodule.train_dataset, 
    #         'valid' : self.datamodule.val_dataset
    #     }
    #     index = 0
    #     for split, dataset in split_data.items():
    #         for image, label in zip(dataset.data, dataset.targets):
    #             if index < 10:
    #                 data_split_at.add(
    #                     wandb.Image(image), 
    #                     name=os.path.join(split, dataset.classes[label], f"{index}.jpg")
    #                     )
    #             preview_data_table.add_data(
    #                 wandb.Image(image), 
    #                 dataset.classes[label], 
    #                 split)
    #             index += 1
    #     data_split_at.add(preview_data_table, 'data_split')
    #     trainer.logger.experiment.log_artifact(data_split_at)

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        images, labels = val_batch

        outputs = pl_module(images.to(pl_module.device))
        logits = torch.sigmoid(outputs)
        preds = torch.argmax(logits, 1)

        columns = ["image", "preds", "ground_truth"]
        for class_name in self.datamodule.train_dataset.classes:
            columns.append('score_' + class_name)
        predictions_table = wandb.Table(columns=columns)

        for img, pred, scores, ground_truth in zip(images, preds, logits, labels):
            row = [wandb.Image(img), self.datamodule.train_dataset.classes[pred], 
                   self.datamodule.train_dataset.classes[ground_truth]]
            for score in scores.tolist():
                row.append(np.round(score, 4))
            predictions_table.add_data(*row)

        trainer.logger.experiment.log({'predictions_results' : predictions_table})

@hydra.main(config_path='./configs', config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    os.makedirs(cfg.data.path, exist_ok=True)
    data = DataModule(data_name=cfg.data.name, 
                      data_path=cfg.data.path,
                      download=cfg.data.download)
    data.prepare_dataset()
    logger.info(f"Downloaded data was saved: {data.train_dataset.root}")

    
    if cfg.model.name == "teacher":
        model = TeacherTrainingModule(
            lr=cfg.training.lr, 
            num_classes=cfg.data.num_classes
            )
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.training.checkpoint_folder,
            filename="best-checkpoint-teacher",
            monitor="valid/acc",
            mode="max",
        )

        early_stopping_callback = EarlyStopping(
            monitor="valid/acc", patience=5, verbose=True, mode="max"
        )

        wandb_logger = WandbLogger(project="knowledge_distillation", 
                                   name="teacher_model")

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            logger=wandb_logger,
            gpus=1,
            callbacks=[checkpoint_callback, 
                       SamplesVisualisationLogger(data), 
                       early_stopping_callback],
            log_every_n_steps=cfg.training.log_every_n_steps,
            deterministic=cfg.training.deterministic,
            limit_train_batches=cfg.training.limit_train_batches,
            limit_val_batches=cfg.training.limit_val_batches,
        )
        trainer.fit(model, data)
        wandb.finish()

    elif cfg.model.name == "distill":
        teacher_model = TeacherTrainingModule.load_from_checkpoint(
            os.path.join(
                cfg.training.checkpoint_folder, 
                'best-checkpoint-teacher.ckpt'
                )
        )
        model = DistillTrainingModule(
            teacher_model,
            temperature=cfg.model.distill_model.temperature,
            distillation_weight=cfg.model.distill_model.distillation_weight, 
            lr=cfg.training.lr,
            num_classes=cfg.data.num_classes
            )
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.training.checkpoint_folder,
            filename="best-checkpoint-distill",
            monitor="distill_valid/acc",
            mode="max",
        )

        early_stopping_callback = EarlyStopping(
            monitor="distill_valid/acc", patience=3, verbose=True, mode="max"
        )

        wandb_logger = WandbLogger(project="knowledge_distillation", 
                                   name="student model")

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            logger=wandb_logger,
            gpus=1,
            callbacks=[checkpoint_callback, 
                       SamplesVisualisationLogger(data), 
                       early_stopping_callback],
            log_every_n_steps=cfg.training.log_every_n_steps,
            deterministic=cfg.training.deterministic,
            limit_train_batches=cfg.training.limit_train_batches,
            limit_val_batches=cfg.training.limit_val_batches,
        )
        trainer.fit(model, data)
        wandb.finish()

if __name__ == "__main__":
    main()