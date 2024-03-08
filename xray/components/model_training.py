from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from xray.entity.config_entity import DataIngestionConfig, DataTransformationConfig,ModelTrainerConfig
from xray.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, ModelTrainerArtifact
from xray.exception import XRayException
from xray.logger import logging
from torchvision.datasets import ImageFolder
from xray.ml.model.arch import Net
import tqdm
from torch import DEVICE
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler


import sys

class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig) -> None:
        self.model_trainer_config:ModelTrainerConfig=model_trainer_config
        self.data_transfromation_artifact:DataTransformationArtifact=(data_transformation_artifact)
        self.model:Module=Net()

    
    def train(self, optimizer: Optimizer) -> None:
        """
        Description: To train the model

        input: model,device,train_loader,optimizer,epoch

        output: loss, batch id and accuracy
        """
        logging.info("Entered the train method of Model trainer class")

        try:
            self.model.train()

            pbar = tqdm(self.data_transformation_artifact.transformed_train_object)

            correct: int = 0

            processed = 0

            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(DEVICE), target.to(DEVICE)

                # Initialization of gradient
                optimizer.zero_grad()

                # In PyTorch, gradient is accumulated over backprop and even though thats used in RNN generally not used in CNN
                # or specific requirements
                ## prediction on data

                y_pred = self.model(data)

                # Calculating loss given the prediction
                loss = F.nll_loss(y_pred, target)

                # Backprop
                loss.backward()

                optimizer.step()

                # get the index of the log-probability corresponding to the max value
                pred = y_pred.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

                processed += len(data)

                pbar.set_description(
                    desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
                )

            logging.info("Exited the train method of Model trainer class")

        except Exception as e:
            raise XRayException(e, sys)
        