import copy
import logging
from typing import Dict
from sklearn import metrics
import numpy as np
from tqdm import tqdm

import torch
from src.cube.ai_backend.pytorch.utils import Monitor, Terminator

LOGGER = logging.getLogger(__name__)


# cool tricks: https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/
class Trainer:
    """Trainer with training and validation loops"""
    def __init__(self, model, dataloaders: Dict, num_classes: int, input_channels: int, criterion, optimizer, scheduler,
                 num_epochs: int, device, monitor: Monitor, terminator: Terminator, finetune: bool, folds: int,
                 parallel_networks: int, nok_threshold: float):
        """
            Args:
                model : PyTorch model
                dataloaders (dict) : Dict containing train and val dataloaders
                num_classes (int) : Number of classes to one hot targets
                input_channels (int) : Number of channels of input
                criterion : pytorch loss function
                optimizer : pytorch optimizer function
                scheduler : pytorch scheduler function
                num_epochs (int) : Number of epochs to train the model
                device : torch.device indicating whether device is cpu or gpu
                monitor (Monitor) : Keep track of current batch and epoch
                terminator (Terminator) : Check if there is any termination signal
                finetune (bool) : If model will further be finetuned
                                if true, scale the progress by 2
                folds (int) : If folds!=-1, scale the progress by number of folds
                parallel_networks (int) : If folds!=-1, parallel_networks is used for aggregating predictions
                nok_threshold (float) : If folds!=-1, it is used for thresholding the aggregated predictions
        """
        self.model = model
        self.train_data = dataloaders['train']
        self.valid_data = dataloaders['val']
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.monitor = monitor
        self.terminator = terminator
        self.finetune = finetune
        self.folds = folds
        self.parallel_networks = parallel_networks
        self.nok_threshold = nok_threshold
        self.best_acc = 0.0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        # scaling of two models (pretrained + finetune)
        if self.finetune:
            self.monitor.n_epochs = self.num_epochs * 2
        # scaling of two models * number of folds
        if self.folds != -1 and self.finetune:
            self.monitor.n_epochs = self.num_epochs * 2 * self.folds
        if not finetune:
            self.monitor.n_epochs = self.num_epochs

    def train_one_epoch(self):
        self.model.train()  # Set model to training mode
        self.monitor.batches_per_epoch = len(self.train_data)
        running_loss = 0.0
        running_corrects = 0
        f1s, recalls, precisions = [], [], []
        if self.terminator.terminate_flag:
            self.terminator.reset()
            raise KeyboardInterrupt
        stream = tqdm(self.train_data, position=0, leave=True)
        # Iterate over batch of data.
        for _, (inputs, labels) in enumerate(stream):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            # zero the parameter gradients
            self.optimizer.zero_grad(set_to_none=True)
            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                # backward + optimize only if in training phase
                outputs = self.model(inputs)
                onehot_labels = torch.nn.functional.one_hot(labels, self.num_classes)
                onehot_labels = onehot_labels.type_as(outputs)
                loss = self.criterion(outputs, onehot_labels)
                stream.set_description('train_loss: {:.2f}'.format(loss.item()))
                loss.backward()
                self.optimizer.step()
            # statistics
            _, preds = torch.max(outputs, 1)
            f1s.append(
                metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
            )
            precisions.append(
                metrics.precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
            )
            recalls.append(
                metrics.recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
            )
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            # monitor only training dataset
            self.monitor.current_batch += 1
            if self.terminator.terminate_flag:
                self.terminator.reset()
                raise KeyboardInterrupt
        self.scheduler.step()
        # monitor only training dataset
        self.monitor.current_batch = 0
        self.monitor.current_epoch += 1

        epoch_loss = running_loss / (len(self.train_data.dataset))
        epoch_acc = (running_corrects.double() / (len(self.train_data.dataset))).item()
        epoch_f1 = np.mean(f1s)
        epoch_precision = np.mean(precisions)
        epoch_recall = np.mean(recalls)
        return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall

    def valid_one_epoch(self):
        self.model.eval()  # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
        f1s, recalls, precisions = [], [], []
        agg_preds, agg_labels = [], []
        if self.terminator.terminate_flag:
            self.terminator.reset()
            raise KeyboardInterrupt
        stream = tqdm(self.valid_data, position=0, leave=True)
        # Iterate over batch of data.
        for _, (inputs, labels) in enumerate(stream):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                onehot_labels = torch.nn.functional.one_hot(labels, self.num_classes)
                onehot_labels = onehot_labels.type_as(outputs)
                loss = self.criterion(outputs, onehot_labels)
                stream.set_description('val_loss: {:.2f}'.format(loss.item()))
            # aggregate validation prediction results for kfold training
            if self.folds != -1:
                lbl = labels.cpu().detach().numpy()
                prob = torch.sigmoid(outputs).cpu().detach().numpy()
                # aggregate the predictions in groups of self.parallel_networks
                # select only the mean predictions corresponding to first column
                for i in range(0, len(prob), self.parallel_networks):
                    agg_preds.append(np.mean(prob[i: i + self.parallel_networks], axis=0)[1])
                    agg_labels.append(lbl[i])
            # statistics
            _, preds = torch.max(outputs, 1)
            f1s.append(
                metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
            )
            precisions.append(
                metrics.precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
            )
            recalls.append(
                metrics.recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
            )
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            if self.terminator.terminate_flag:
                self.terminator.reset()
                raise KeyboardInterrupt

        epoch_loss = running_loss / (len(self.valid_data.dataset))
        epoch_acc = (running_corrects.double() / (len(self.valid_data.dataset))).item()
        epoch_f1 = np.mean(f1s)
        epoch_precision = np.mean(precisions)
        epoch_recall = np.mean(recalls)
        # deep copy the model
        if epoch_acc > self.best_acc:
            if self.folds != -1:
                # calculate aggregated accuracy
                agg_preds = [0 if x < self.nok_threshold else 1 for x in agg_preds]
                assert len(agg_labels) == len(agg_preds)
                agg_acc = np.sum(np.array(agg_labels) == np.array(agg_preds)) / len(agg_preds)
                LOGGER.info(f"Val acc improved from {self.best_acc} to {agg_acc}.")
                self.best_acc = agg_acc
            else:
                LOGGER.info(f"Val acc improved from {self.best_acc} to {epoch_acc}.")
                self.best_acc = epoch_acc
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
        return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall
