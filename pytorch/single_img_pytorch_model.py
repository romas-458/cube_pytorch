import os
from typing import Dict, List
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
import cv2
from PIL import Image
from sklearn import metrics

import wandb

import torch
import torch.nn as nn

from cube_pytorch.pytorch.dataset import CubeDataset
from cube_pytorch.pytorch.model import build_models
from cube_pytorch.pytorch.train_loop import Trainer
from cube_pytorch.pytorch.utils import Monitor, Terminator, get_train_transforms, get_val_transforms, \
    set_global_seeds, EvaluationMonitor

# LOGGER = logging.getLogger(__name__)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def take_filename(str):
    return str.split('\\')[-1]

def prepare_df_from_json(path_to_datajson, needed_classes):
    df = pd.read_json(path_to_datajson)
    subdf = df[["image_urls", "class_type", "class_name"]]
    training_subset = subdf[(subdf["class_name"] == needed_classes[0]) | (subdf["class_name"] == needed_classes[1]) | (
                subdf["class_name"] == needed_classes[2]) | (subdf["class_name"] == needed_classes[3])]
    data = []
    for urls, label, subclass in training_subset[["image_urls", "class_type", "class_name"]].values:
        for i in range(4):
            if label == 'OK':
                label01 = 0
            else:
                label01 = 1
            # data.append([take_filename(urls[i]), label, subclass])
            data.append([take_filename(urls[i]), label01, subclass])

    return pd.DataFrame(data, columns=["file", "label", "subclass"])


class ClassifierModel:
    def __init__(
            self,
            seed: int = 42,
            num_workers: int = 2,
            learning_rate: float = 1e-3,
            batch_size: int = 8,
            epochs: int = 20,
            split_ratio: float = 0.1,
            num_classes: int = 2,
            embed_dim: int = 256,
            input_channels: int = 3,
            folds: int = 4,
            width: int = 512,
            height: int = 512,
            means: list = [0.485, 0.456, 0.406],
            stds: list = [0.229, 0.224, 0.225],
            train_path: str = "",
            save_model_path: str = "",
            trained_model_path: str = "",
            base_model_path: str = "",
            model_name: str = "resnext101",
            feature_extract: bool = True,
            use_pretrain: bool = False,  # don't download imagenet weights
            finetune_layer: int = 270,
            nok_threshold: float = 0.5,
            images_per_sample: int = 4,
            #wandb
            finetune_lr_multiplier: float = 0.1,
            finetune_max_lr_multiplier: int = 10,
            finetune_epochs: int = 20,
            finetune_embed_dim: int = 256,
            #
            eval_examples = None,
            path_to_datajson = None,

    ):
        set_global_seeds()
        # hyperparameters BEGIN
        self.random_seed = seed
        self.num_workers = num_workers
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.split_ratio = split_ratio
        self.num_classes = num_classes
        self.embed_size = embed_dim
        self.input_channels = input_channels
        self.folds = folds
        self.width = width
        self.height = height
        self.means = means
        self.stds = stds
        self.train_path = train_path
        self.base_model_path = base_model_path
        self.save_model_path = save_model_path
        self.trained_model_path = trained_model_path
        self.model_name = model_name
        self.feature_extract = feature_extract
        self.use_pretrain = use_pretrain
        self.nok_threshold = nok_threshold
        # hyperparameters END
        self.parallel_networks = images_per_sample
        self.net = self._init_model()

        ##wandb
        #previous
        self.finetune_layer = finetune_layer
        #new
        self.finetune_embed_size = finetune_embed_dim
        self.finetune_lr_multiplier = finetune_lr_multiplier
        self.finetune_max_lr_multiplier = finetune_max_lr_multiplier
        self.finetune_epochs = finetune_epochs
        #
        self.eval_examples = eval_examples
        self.path_to_datajson = path_to_datajson

    def _init_model(self):
        LOGGER.info("Initializing model from given weights")
        if os.path.exists(self.save_model_path):
            LOGGER.info(f"Found a model at {self.save_model_path}")
            classifier_model = torch.load(self.save_model_path, map_location=device)
        else:
            LOGGER.info(
                f"Saved model not found. Rebuilding from base model, using base weights found at {self.base_model_path}"
            )
            classifier_model = self._build_pretrain_model()
        return classifier_model

    # build a pretrained model with imagenet weights
    def _build_pretrain_model(self):
        """
        Create a pretrained model
        """
        if not os.path.exists(self.base_model_path):
            raise FileNotFoundError(f"Base model file not found: {self.base_model_path}")
        # set best weights and finetune layers to defaults of pretraining
        classifier_model = build_models(
            model_name=self.model_name,
            num_classes=self.num_classes,
            in_channels=self.input_channels,
            embedding_size=self.embed_size,
            feature_extract=self.feature_extract,
            use_pretrained=self.use_pretrain,
            base_model_path=self.base_model_path,
            num_ft_layers=-1,
            bst_model_weights=None
        )
        return classifier_model.to(device)

    def _build_finetune_model(self):
        """
        Create a finetuned model with pretrained weights
        """
        classifier_model = build_models(
            model_name=self.model_name,
            num_classes=self.num_classes,
            in_channels=self.input_channels,
            # embedding_size=self.embed_size,
            embedding_size = self.finetune_embed_size,
            feature_extract=self.feature_extract,
            use_pretrained=self.use_pretrain,
            base_model_path=self.base_model_path,
            num_ft_layers=self.finetune_layer,
            bst_model_weights=self.trained_model_path
        )
        return classifier_model.to(device)

    def _prepare_training_generators(self, train_df: pd.DataFrame, train_root_dir: str, is_kfold: bool = False,
                                     fold: int = -1) -> tuple:
        """
        Prepare training and validation dataloaders

        Args:
            train_df (pd.DataFrame) : Dataframe containing 2 columns file and labels
            train_root_dir (str) : Base path to images
            is_kfold (bool): whether to create kfold dataloaders
            fold (int) : integer corresponding to current fold number
        Returns:
            train_loader, val_loader (tuple) : A tuple of train and val dataloaders
        """

        if is_kfold:
            df_train = train_df[train_df.kfold != fold].reset_index(drop=True)
            df_val = train_df[train_df.kfold == fold].reset_index(drop=True)
            train_x, train_y = df_train["file"], df_train["label"]
            val_x, val_y = df_val["file"], df_val["label"]
        else:
            x = train_df["file"]
            y = train_df["label"]

            train_x, val_x, train_y, val_y = train_test_split(
                x,
                y,
                test_size=self.split_ratio,
                random_state=self.random_seed,
                shuffle=True,
                stratify=y
            )
        LOGGER.info(
            f"Training shape: {train_x.shape}, {train_y.shape}, {np.unique(train_y, return_counts=True)}"
        )
        LOGGER.info(
            f"Validation shape: {val_x.shape}, {val_y.shape}, {np.unique(val_y, return_counts=True)}"
        )

        trn_transforms = get_train_transforms(
            self.height, self.width, self.means, self.stds
        )
        val_transforms = get_val_transforms(
            self.height, self.width, self.means, self.stds
        )
        train_dataset = CubeDataset(
            x=list(train_x), y=list(train_y), root_dir=train_root_dir, transform=trn_transforms
        )
        val_dataset = CubeDataset(
            x=list(val_x), y=list(val_y), root_dir=train_root_dir, transform=val_transforms
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=False
        )
        return train_loader, val_loader

    def _prepare_df(self, dataframe: pd.DataFrame, is_kfold: bool = False) -> pd.DataFrame:
        """
        Prepare dataframe from raw dataset

        Args:
            dataframe (pd.DataFrame) : Pandas dataframe of the original dataset
            is_kfold (bool) : Boolean to indicate if preparing is done for k-fold dataset
        Returns:
            pd.DataFrame : A dataframe containing 3 columns (file, label, subclass)
        """
        data = []
        dataframe = dataframe.copy()
        if is_kfold:
            for prefix, label, subclass, kfold in dataframe[["prefix", "label", "subclass", "kfold"]].values:
                for i in range(4):
                    data.append([f"{prefix}_{i}.jpg", label, subclass, kfold])
            return pd.DataFrame(data, columns=["file", "label", "subclass", "kfold"])
        else:
            dataframe["prefix"] = dataframe.index
            for prefix, label, subclass in dataframe[["prefix", "label", "subclass"]].values:
                for i in range(4):
                    data.append([f"{prefix}_{i}.jpg", label, subclass])
            return pd.DataFrame(data, columns=["file", "label", "subclass"])

    def preprocess_pil_image(self, image: Image) -> np.ndarray:
        img_array = np.array(image)
        img_array = cv2.resize(img_array, (self.height, self.width))
        img_array = img_array / 255.0
        img_array = (img_array - np.array(self.means)) / np.array(self.stds)
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array

    def transform_and_preprocess_batch(self, images: list):
        return np.array(list(map(self.preprocess_pil_image, images)))

    def predict(self, images: list) -> float:
        """
        Prediction

        Args:
            images (list): images_per_set number of pil images
        Returns:
            mean_pred (float): Prediction score after aggregating all predictions
        """
        pred_batch = self.transform_and_preprocess_batch(images)
        pred_batch = torch.from_numpy(pred_batch).to(device).float()
        self.net.eval()
        # batch prediction
        with torch.no_grad():
            output = self.net(pred_batch)
            prob = torch.sigmoid(output)
        mean_pred = np.mean(prob.cpu().numpy(), axis=0)
        return mean_pred[1]

    def evaluate(self, eval_df: pd.DataFrame,  local_storage_dir: str, eval_monitor: EvaluationMonitor,
                 eval_terminator: Terminator) -> List:
        """
        Evaluation

        Args:
            eval_df (pd.DataFrame): Dataframe containing image file names
            local_storage_dir (str): Path to eval images
            eval_monitor (EvaluationMonitor): Monitor the progress of evaluating dataset
            eval_terminator (Terminator) : Check if there is a termination signal
        Returns:
            preds (List): List of aggregated predictions
        """
        eval_df = self._prepare_df(eval_df)
        val_transforms = get_val_transforms(
            self.height, self.width, self.means, self.stds
        )
        eval_batch = CubeDataset(
            x=list(eval_df["file"]),
            y=None,
            root_dir=local_storage_dir,
            transform=val_transforms
        )
        # NOTE: batch size should be a multiple of self.parallel_networks
        eval_loader = torch.utils.data.DataLoader(
            eval_batch,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=False
        )
        eval_monitor.total_images = len(eval_loader) * self.batch_size
        all_preds = []
        preds = []
        self.net.eval()
        try:
            with torch.no_grad():
                for _, (data) in tqdm(
                        enumerate(eval_loader), total=len(eval_loader), position=0
                ):
                    data = data.to(device, non_blocking=True)
                    output = self.net(data)
                    prob = torch.sigmoid(output).cpu().numpy()
                    # aggregate the predictions in groups of self.parallel_networks
                    for i in range(0, len(prob), self.parallel_networks):
                        all_preds.append(
                            list(np.mean(prob[i: i + self.parallel_networks], axis=0))
                        )
                    eval_monitor.evaluated_images += self.batch_size
                    if eval_terminator.terminate_flag:
                        eval_terminator.reset()
                        raise KeyboardInterrupt
            # instead of storing max prediction store column 1 prediction and use it for thresholding
            for x in all_preds:
                preds.append(x[1])
        except KeyboardInterrupt:
            LOGGER.info("Evaluation interrupted")
        return preds

    def evaluate_from_csv(self, path, costume_classes, eval_monitor: EvaluationMonitor,
                 eval_terminator: Terminator) -> List:
        """
        Evaluation

        Args:
            eval_df (pd.DataFrame): Dataframe containing image file names
            local_storage_dir (str): Path to eval images
            eval_monitor (EvaluationMonitor): Monitor the progress of evaluating dataset
            eval_terminator (Terminator) : Check if there is a termination signal
        Returns:
            preds (List): List of aggregated predictions
        """

        eval_df = prepare_df_from_json(path_to_datajson=path, needed_classes=costume_classes)
        LOGGER.info("size of DF" + str(len(eval_df)))
        LOGGER.info("DF head")
        LOGGER.info(eval_df.head())
        # eval_df = self._prepare_df(eval_df)
        val_transforms = get_val_transforms(
            self.height, self.width, self.means, self.stds
        )
        eval_batch = CubeDataset(
            x=list(eval_df["file"]),
            y=None,
            root_dir=self.train_path,
            transform=val_transforms
        )
        # NOTE: batch size should be a multiple of self.parallel_networks
        eval_loader = torch.utils.data.DataLoader(
            eval_batch,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=False
        )
        eval_monitor.total_images = len(eval_loader) * self.batch_size
        all_preds = []
        preds = []
        self.net.eval()
        try:
            with torch.no_grad():
                for _, (data) in tqdm(
                        enumerate(eval_loader), total=len(eval_loader), position=0
                ):
                    data = data.to(device, non_blocking=True)
                    output = self.net(data)
                    prob = torch.sigmoid(output).cpu().numpy()
                    # aggregate the predictions in groups of self.parallel_networks
                    # for i in range(0, len(prob), self.parallel_networks):
                    #     all_preds.append(
                    #         list(np.mean(prob[i: i + self.parallel_networks], axis=0))
                    #     )

                    prob = prob.tolist()
                    all_preds.extend(prob)

                    eval_monitor.evaluated_images += self.batch_size
                    if eval_terminator.terminate_flag:
                        eval_terminator.reset()
                        raise KeyboardInterrupt
            # instead of storing max prediction store column 1 prediction and use it for thresholding
            for x in all_preds:
                preds.append(x[1])
        except KeyboardInterrupt:
            LOGGER.info("Evaluation interrupted")
        return eval_df, preds

    def evaluate_from_csv_each_epoch(self, path, costume_classes, model, eval_monitor: EvaluationMonitor,
                 eval_terminator: Terminator) -> List:
        """
        Evaluation

        Args:
            eval_df (pd.DataFrame): Dataframe containing image file names
            local_storage_dir (str): Path to eval images
            eval_monitor (EvaluationMonitor): Monitor the progress of evaluating dataset
            eval_terminator (Terminator) : Check if there is a termination signal
        Returns:
            preds (List): List of aggregated predictions
        """

        eval_df = prepare_df_from_json(path_to_datajson=path, needed_classes=costume_classes)
        LOGGER.info("size of DF" + str(len(eval_df)))
        LOGGER.info("DF head")
        LOGGER.info(eval_df.head())
        # eval_df = self._prepare_df(eval_df)
        val_transforms = get_val_transforms(
            self.height, self.width, self.means, self.stds
        )
        eval_batch = CubeDataset(
            x=list(eval_df["file"]),
            y=None,
            root_dir=self.train_path,
            transform=val_transforms
        )
        # NOTE: batch size should be a multiple of self.parallel_networks
        eval_loader = torch.utils.data.DataLoader(
            eval_batch,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=False
        )
        # eval_monitor.total_images = len(eval_loader) * self.batch_size
        all_preds = []
        preds = []
        model.eval()
        try:
            with torch.no_grad():
                for _, (data) in tqdm(
                        enumerate(eval_loader), total=len(eval_loader), position=0
                ):
                    data = data.to(device, non_blocking=True)
                    output = model(data)
                    prob = torch.sigmoid(output).cpu().numpy()
                    # aggregate the predictions in groups of self.parallel_networks
                    # for i in range(0, len(prob), self.parallel_networks):
                    #     all_preds.append(
                    #         list(np.mean(prob[i: i + self.parallel_networks], axis=0))
                    #     )

                    prob = prob.tolist()
                    all_preds.extend(prob)

                    # eval_monitor.evaluated_images += self.batch_size
                    if eval_terminator.terminate_flag:
                        eval_terminator.reset()
                        raise KeyboardInterrupt
            # instead of storing max prediction store column 1 prediction and use it for thresholding
            for x in all_preds:
                preds.append(x[1])
        except KeyboardInterrupt:
            LOGGER.info("Evaluation interrupted")
        return eval_df, preds

    def train_pretrain_wandb(self, cws: np.ndarray, train_loader_length: int, dataloaders_dict: Dict, monitor: Monitor,
                       terminator: Terminator, is_kfold: bool = False, f: int = -1, base_model_path: str = "",
                       folds: int = -1) -> float:
        """
        Train a pretrained model

        Args:
            cws (np.ndarray): A numpy array of class weights corresponding to each label
            train_loader_length (int): Length of train dataloader
            dataloaders_dict (Dict): A dict containing train and val dataloader with keys train and val
            monitor (Monitor) : monitor the current batch and epoch of the training loop
            terminator (Terminator) : check if there is any termination request
            is_kfold (bool): whether training is using kfold dataset
            f (int): integer corresponding to current fold number
            base_model_path (str) : a base path with only model name
            folds (int) : Number of kfolds used for scaling progress
        Returns:
            best_val_acc (float): Best validation accuracy
        """
        self.net = self._build_pretrain_model()
        t_parameters = [p for p in self.net.parameters() if p.requires_grad]
        criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(cws)).to(device)
        optimizer = torch.optim.AdamW(t_parameters, lr=self.lr, amsgrad=True)
        # one cycle scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 100,
            steps_per_epoch=train_loader_length,
            epochs=self.epochs
        )
        if is_kfold:
            self.save_model_path = base_model_path.split(".pth")[0] + f"_{f}_fold.pth"
        trainer = Trainer(model=self.net, dataloaders=dataloaders_dict, num_classes=self.num_classes,
                          input_channels=self.input_channels, criterion=criterion, optimizer=optimizer,
                          scheduler=scheduler, num_epochs=self.epochs, device=device, monitor=monitor,
                          terminator=terminator, finetune=True, folds=folds, parallel_networks=self.parallel_networks,
                          nok_threshold=self.nok_threshold)
        since = time.time()
        for epoch in range(1, self.epochs+1):
            LOGGER.info(f"\n{'--'*5} EPOCH: {epoch} | {self.epochs} {'--'*5}\n")
            epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall = trainer.train_one_epoch()
            LOGGER.info(
                "\nPhase: {} | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(
                    'train',
                    epoch_loss,
                    epoch_acc,
                    epoch_f1,
                    epoch_precision,
                    epoch_recall
                )
            )

            wandb.log({'train_acc': epoch_acc, 'train_loss': epoch_loss})
            epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall = trainer.valid_one_epoch()
            LOGGER.info(
                "\nPhase: {} | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(
                    'valid',
                    epoch_loss,
                    epoch_acc,
                    epoch_f1,
                    epoch_precision,
                    epoch_recall
                )
            )

            wandb.log({'train_epoch': epoch, 'train_val_acc': epoch_acc, 'train_val_loss': epoch_loss})

        time_elapsed = time.time() - since
        LOGGER.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        LOGGER.info("Best val accuracy: {:4f}".format(trainer.best_acc))
        # load best model weights
        self.net.load_state_dict(trainer.best_model_wts)
        LOGGER.info(f"Saving best pretrained model at {self.save_model_path}")
        torch.save(self.net, self.save_model_path)
        return trainer.best_acc

    def train_pretrain_wandb_each_epoch(self, cws: np.ndarray, train_loader_length: int, dataloaders_dict: Dict, monitor: Monitor,
                       terminator: Terminator, is_kfold: bool = False, f: int = -1, base_model_path: str = "",
                       folds: int = -1) -> float:
        """
        Train a pretrained model

        Args:
            cws (np.ndarray): A numpy array of class weights corresponding to each label
            train_loader_length (int): Length of train dataloader
            dataloaders_dict (Dict): A dict containing train and val dataloader with keys train and val
            monitor (Monitor) : monitor the current batch and epoch of the training loop
            terminator (Terminator) : check if there is any termination request
            is_kfold (bool): whether training is using kfold dataset
            f (int): integer corresponding to current fold number
            base_model_path (str) : a base path with only model name
            folds (int) : Number of kfolds used for scaling progress
        Returns:
            best_val_acc (float): Best validation accuracy
        """
        self.net = self._build_pretrain_model()
        t_parameters = [p for p in self.net.parameters() if p.requires_grad]
        criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(cws)).to(device)
        optimizer = torch.optim.AdamW(t_parameters, lr=self.lr, amsgrad=True)
        # one cycle scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 100,
            steps_per_epoch=train_loader_length,
            epochs=self.epochs
        )
        if is_kfold:
            self.save_model_path = base_model_path.split(".pth")[0] + f"_{f}_fold.pth"
        trainer = Trainer(model=self.net, dataloaders=dataloaders_dict, num_classes=self.num_classes,
                          input_channels=self.input_channels, criterion=criterion, optimizer=optimizer,
                          scheduler=scheduler, num_epochs=self.epochs, device=device, monitor=monitor,
                          terminator=terminator, finetune=True, folds=folds, parallel_networks=self.parallel_networks,
                          nok_threshold=self.nok_threshold)
        since = time.time()

        best_tntp = -1
        best_epoch = -1
        for epoch in range(1, self.epochs+1):
            LOGGER.info(f"\n{'--'*5} EPOCH: {epoch} | {self.epochs} {'--'*5}\n")
            epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall = trainer.train_one_epoch()
            LOGGER.info(
                "\nPhase: {} | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(
                    'train',
                    epoch_loss,
                    epoch_acc,
                    epoch_f1,
                    epoch_precision,
                    epoch_recall
                )
            )

            wandb.log({'train_acc': epoch_acc, 'train_loss': epoch_loss})

            epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall = trainer.valid_one_epoch()
            LOGGER.info(
                "\nPhase: {} | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(
                    'valid',
                    epoch_loss,
                    epoch_acc,
                    epoch_f1,
                    epoch_precision,
                    epoch_recall
                )
            )

            wandb.log({'train_epoch': epoch, 'train_val_acc': epoch_acc, 'train_val_loss': epoch_loss})

            eval_df, preds = self.evaluate_from_csv_each_epoch(self.path_to_datajson, self.eval_examples, trainer.model, monitor,
                                              terminator)

            predictions = [0 if x < self.nok_threshold else 1 for x in preds]
            LOGGER.info("predictions len = " + str(len(predictions)) + 'eval len = ' + str(len(eval_df)))
            assert len(predictions) == len(eval_df)
            # NOTE works only for binary case
            tn, fp, fn, tp = metrics.confusion_matrix(eval_df["label"].values, predictions).ravel()
            print('tn= ' + str(tn) + 'fp= ' + str(fp) + 'fn= ' + str(fn) + 'tp= ' + str(tp))
            wandb.log({'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})

            tntp = tn + tp

            if best_tntp < tntp:
                best_tntp = tntp
                best_epoch = epoch

                wandb.log({'best_tntp': best_tntp, 'best_epoch': best_epoch})

        time_elapsed = time.time() - since
        LOGGER.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        LOGGER.info("Best val accuracy: {:4f}".format(trainer.best_acc))
        # load best model weights
        self.net.load_state_dict(trainer.best_model_wts)
        LOGGER.info(f"Saving best pretrained model at {self.save_model_path}")
        torch.save(self.net, self.save_model_path)
        return trainer.best_acc

    def train_pretrain(self, cws: np.ndarray, train_loader_length: int, dataloaders_dict: Dict, monitor: Monitor,
                       terminator: Terminator, is_kfold: bool = False, f: int = -1, base_model_path: str = "",
                       folds: int = -1) -> float:
        """
        Train a pretrained model

        Args:
            cws (np.ndarray): A numpy array of class weights corresponding to each label
            train_loader_length (int): Length of train dataloader
            dataloaders_dict (Dict): A dict containing train and val dataloader with keys train and val
            monitor (Monitor) : monitor the current batch and epoch of the training loop
            terminator (Terminator) : check if there is any termination request
            is_kfold (bool): whether training is using kfold dataset
            f (int): integer corresponding to current fold number
            base_model_path (str) : a base path with only model name
            folds (int) : Number of kfolds used for scaling progress
        Returns:
            best_val_acc (float): Best validation accuracy
        """
        self.net = self._build_pretrain_model()
        t_parameters = [p for p in self.net.parameters() if p.requires_grad]
        criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(cws)).to(device)
        optimizer = torch.optim.AdamW(t_parameters, lr=self.lr, amsgrad=True)
        # one cycle scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 100,
            steps_per_epoch=train_loader_length,
            epochs=self.epochs
        )
        if is_kfold:
            self.save_model_path = base_model_path.split(".pth")[0] + f"_{f}_fold.pth"
        trainer = Trainer(model=self.net, dataloaders=dataloaders_dict, num_classes=self.num_classes,
                          input_channels=self.input_channels, criterion=criterion, optimizer=optimizer,
                          scheduler=scheduler, num_epochs=self.epochs, device=device, monitor=monitor,
                          terminator=terminator, finetune=True, folds=folds, parallel_networks=self.parallel_networks,
                          nok_threshold=self.nok_threshold)
        since = time.time()
        for epoch in range(1, self.epochs+1):
            LOGGER.info(f"\n{'--'*5} EPOCH: {epoch} | {self.epochs} {'--'*5}\n")
            epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall = trainer.train_one_epoch()
            LOGGER.info(
                "\nPhase: {} | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(
                    'train',
                    epoch_loss,
                    epoch_acc,
                    epoch_f1,
                    epoch_precision,
                    epoch_recall
                )
            )

            epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall = trainer.valid_one_epoch()
            LOGGER.info(
                "\nPhase: {} | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(
                    'valid',
                    epoch_loss,
                    epoch_acc,
                    epoch_f1,
                    epoch_precision,
                    epoch_recall
                )
            )

        time_elapsed = time.time() - since
        LOGGER.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        LOGGER.info("Best val accuracy: {:4f}".format(trainer.best_acc))
        # load best model weights
        self.net.load_state_dict(trainer.best_model_wts)
        LOGGER.info(f"Saving best pretrained model at {self.save_model_path}")
        torch.save(self.net, self.save_model_path)
        return trainer.best_acc

    def train_finetune_wandb(self, cws: np.ndarray, train_loader_length: int, dataloaders_dict: Dict, monitor: Monitor,
                       terminator: Terminator, is_kfold: bool = False, folds: int = -1) -> float:
        """
        Finetune a pretrained model

        Args:
            cws (np.ndarray): A numpy array of class weights corresponding to each label
            train_loader_length (int): Length of train dataloader
            dataloaders_dict (Dict): A dict containing train and val dataloader with keys train and val
            monitor (Monitor) : monitor the current batch and epoch of the training loop
            terminator (Terminator) : check if there is any termination request
            is_kfold (bool): whether training is using kfold dataset
            folds (int) : Number of kfolds used for scaling progress
        Returns:
            best_val_acc (float): Best validation accuracy
        """
        self.trained_model_path = self.save_model_path
        self.net = self._build_finetune_model()
        t_parameters = [p for p in self.net.parameters() if p.requires_grad]
        criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(cws)).to(device)
        optimizer = torch.optim.AdamW(t_parameters, lr=self.lr * self.finetune_lr_multiplier, amsgrad=True)
        # one cycle scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * self.finetune_max_lr_multiplier,
            steps_per_epoch=train_loader_length,
            # epochs=self.epochs
            epochs = self.finetune_epochs
        )
        trainer = Trainer(model=self.net, dataloaders=dataloaders_dict, num_classes=self.num_classes,
                          input_channels=self.input_channels, criterion=criterion, optimizer=optimizer,
                          scheduler=scheduler, num_epochs=self.finetune_epochs, device=device, monitor=monitor,
                          terminator=terminator, finetune=True, folds=folds, parallel_networks=self.parallel_networks,
                          nok_threshold=self.nok_threshold)
        since = time.time()
        for epoch in range(1, self.finetune_epochs+1):
            LOGGER.info(f"\n{'--'*5} EPOCH: {epoch} | {self.finetune_epochs} {'--'*5}\n")
            epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall = trainer.train_one_epoch()
            LOGGER.info(
                "\nPhase: {} | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(
                    'train',
                    epoch_loss,
                    epoch_acc,
                    epoch_f1,
                    epoch_precision,
                    epoch_recall
                )
            )
            wandb.log({'fine_acc': epoch_acc, 'fine_loss': epoch_loss})
            epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall = trainer.valid_one_epoch()
            LOGGER.info(
                "\nPhase: {} | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(
                    'valid',
                    epoch_loss,
                    epoch_acc,
                    epoch_f1,
                    epoch_precision,
                    epoch_recall
                )
            )
            wandb.log({'fine_epoch': epoch, 'fine_loss_val': epoch_loss, 'fine_acc_val': epoch_acc})

        time_elapsed = time.time() - since
        LOGGER.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        LOGGER.info("Best val accuracy: {:4f}".format(trainer.best_acc))
        wandb.log({'fine_best_val_acc': trainer.best_acc})
        # load best model weights
        self.net.load_state_dict(trainer.best_model_wts)
        # don't save the finetuned model for any folds
        if not is_kfold:
            torch.save(self.net, self.save_model_path)
            LOGGER.info(f"Saving best finetuned model at {self.save_model_path}")
        return trainer.best_acc

    def train_finetune_wandb_each_epoch(self, cws: np.ndarray, train_loader_length: int, dataloaders_dict: Dict, monitor: Monitor,
                       terminator: Terminator, is_kfold: bool = False, folds: int = -1) -> float:
        """
        Finetune a pretrained model

        Args:
            cws (np.ndarray): A numpy array of class weights corresponding to each label
            train_loader_length (int): Length of train dataloader
            dataloaders_dict (Dict): A dict containing train and val dataloader with keys train and val
            monitor (Monitor) : monitor the current batch and epoch of the training loop
            terminator (Terminator) : check if there is any termination request
            is_kfold (bool): whether training is using kfold dataset
            folds (int) : Number of kfolds used for scaling progress
        Returns:
            best_val_acc (float): Best validation accuracy
        """
        self.trained_model_path = self.save_model_path
        self.net = self._build_finetune_model()
        t_parameters = [p for p in self.net.parameters() if p.requires_grad]
        criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(cws)).to(device)
        optimizer = torch.optim.AdamW(t_parameters, lr=self.lr * self.finetune_lr_multiplier, amsgrad=True)
        # one cycle scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * self.finetune_max_lr_multiplier,
            steps_per_epoch=train_loader_length,
            # epochs=self.epochs
            epochs = self.finetune_epochs
        )
        trainer = Trainer(model=self.net, dataloaders=dataloaders_dict, num_classes=self.num_classes,
                          input_channels=self.input_channels, criterion=criterion, optimizer=optimizer,
                          scheduler=scheduler, num_epochs=self.finetune_epochs, device=device, monitor=monitor,
                          terminator=terminator, finetune=True, folds=folds, parallel_networks=self.parallel_networks,
                          nok_threshold=self.nok_threshold)
        since = time.time()

        best_tntp = -1
        best_epoch = -1

        for epoch in range(1, self.finetune_epochs+1):
            LOGGER.info(f"\n{'--'*5} EPOCH: {epoch} | {self.finetune_epochs} {'--'*5}\n")
            epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall = trainer.train_one_epoch()
            LOGGER.info(
                "\nPhase: {} | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(
                    'train',
                    epoch_loss,
                    epoch_acc,
                    epoch_f1,
                    epoch_precision,
                    epoch_recall
                )
            )
            wandb.log({'fine_acc': epoch_acc, 'fine_loss': epoch_loss})
            epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall = trainer.valid_one_epoch()
            LOGGER.info(
                "\nPhase: {} | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(
                    'valid',
                    epoch_loss,
                    epoch_acc,
                    epoch_f1,
                    epoch_precision,
                    epoch_recall
                )
            )
            wandb.log({'fine_epoch': epoch, 'fine_loss_val': epoch_loss, 'fine_acc_val': epoch_acc})

            eval_df, preds = self.evaluate_from_csv_each_epoch(self.path_to_datajson, self.eval_examples, trainer.model,
                                                               monitor,
                                                               terminator)

            predictions = [0 if x < self.nok_threshold else 1 for x in preds]
            LOGGER.info("predictions len = " + str(len(predictions)) + 'eval len = ' + str(len(eval_df)))
            assert len(predictions) == len(eval_df)
            # NOTE works only for binary case
            tn, fp, fn, tp = metrics.confusion_matrix(eval_df["label"].values, predictions).ravel()
            print('tn= ' + str(tn) + 'fp= ' + str(fp) + 'fn= ' + str(fn) + 'tp= ' + str(tp))
            wandb.log({'tn_fine': tn, 'fp_fine': fp, 'fn_fine': fn, 'tp_fine': tp})

            tntp = tn + tp

            if best_tntp < tntp:
                best_tntp = tntp
                best_epoch = epoch

                wandb.log({'best_tntp_fine': best_tntp, 'best_epoch_fine': best_epoch})

        time_elapsed = time.time() - since
        LOGGER.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        LOGGER.info("Best val accuracy: {:4f}".format(trainer.best_acc))
        wandb.log({'fine_best_val_acc': trainer.best_acc})
        # load best model weights
        self.net.load_state_dict(trainer.best_model_wts)
        # don't save the finetuned model for any folds
        if not is_kfold:
            torch.save(self.net, self.save_model_path)
            LOGGER.info(f"Saving best finetuned model at {self.save_model_path}")
        return trainer.best_acc

    def train_finetune(self, cws: np.ndarray, train_loader_length: int, dataloaders_dict: Dict, monitor: Monitor,
                       terminator: Terminator, is_kfold: bool = False, folds: int = -1) -> float:
        """
        Finetune a pretrained model

        Args:
            cws (np.ndarray): A numpy array of class weights corresponding to each label
            train_loader_length (int): Length of train dataloader
            dataloaders_dict (Dict): A dict containing train and val dataloader with keys train and val
            monitor (Monitor) : monitor the current batch and epoch of the training loop
            terminator (Terminator) : check if there is any termination request
            is_kfold (bool): whether training is using kfold dataset
            folds (int) : Number of kfolds used for scaling progress
        Returns:
            best_val_acc (float): Best validation accuracy
        """
        self.trained_model_path = self.save_model_path
        self.net = self._build_finetune_model()
        t_parameters = [p for p in self.net.parameters() if p.requires_grad]
        criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(cws)).to(device)
        optimizer = torch.optim.AdamW(t_parameters, lr=self.lr * self.finetune_lr_multiplier, amsgrad=True)
        # one cycle scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * self.finetune_max_lr_multiplier,
            steps_per_epoch=train_loader_length,
            # epochs=self.epochs
            epochs = self.finetune_epochs
        )
        trainer = Trainer(model=self.net, dataloaders=dataloaders_dict, num_classes=self.num_classes,
                          input_channels=self.input_channels, criterion=criterion, optimizer=optimizer,
                          scheduler=scheduler, num_epochs=self.finetune_epochs, device=device, monitor=monitor,
                          terminator=terminator, finetune=True, folds=folds, parallel_networks=self.parallel_networks,
                          nok_threshold=self.nok_threshold)
        since = time.time()
        for epoch in range(1, self.finetune_epochs+1):
            LOGGER.info(f"\n{'--'*5} EPOCH: {epoch} | {self.finetune_epochs} {'--'*5}\n")
            epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall = trainer.train_one_epoch()
            LOGGER.info(
                "\nPhase: {} | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(
                    'train',
                    epoch_loss,
                    epoch_acc,
                    epoch_f1,
                    epoch_precision,
                    epoch_recall
                )
            )

            epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall = trainer.valid_one_epoch()
            LOGGER.info(
                "\nPhase: {} | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(
                    'valid',
                    epoch_loss,
                    epoch_acc,
                    epoch_f1,
                    epoch_precision,
                    epoch_recall
                )
            )

        time_elapsed = time.time() - since
        LOGGER.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        LOGGER.info("Best val accuracy: {:4f}".format(trainer.best_acc))
        # load best model weights
        self.net.load_state_dict(trainer.best_model_wts)
        # don't save the finetuned model for any folds
        if not is_kfold:
            torch.save(self.net, self.save_model_path)
            LOGGER.info(f"Saving best finetuned model at {self.save_model_path}")
        return trainer.best_acc

    def train(self, train_df: pd.DataFrame, monitor: Monitor, terminator: Terminator):
        """
        Training
        First perform a training of pretrained model.
        If arguments of finetuning are passed, perform a finetuning using weights of above pretrained model.

        Args:
            train_df (pd.DataFrame) : Dataframe containing image file names, labels and sublabel
            monitor (Monitor) : monitor the current batch and epoch of the training loop
            terminator (Terminator) : check if there is any termination request
        """
        train_df = self._prepare_df(train_df)
        train_y = train_df["label"]
        cws = class_weight.compute_class_weight("balanced", np.unique(train_y), train_y)
        LOGGER.info(f"Class weights for labels: {cws}")

        LOGGER.info("Loading data")
        train_loader, val_loader = self._prepare_training_generators(
            train_df, self.train_path
        )
        train_loader_length = len(train_loader)
        # Create training and validation dataloaders
        dataloaders_dict = {"train": train_loader, "val": val_loader}
        try:
            # train a pretrained model
            if self.feature_extract:
                LOGGER.info("Start training pretrained models")
                _ = self.train_pretrain(cws, train_loader_length, dataloaders_dict, monitor, terminator, False)
            # finetune a pretrained model
            if self.finetune_layer != -1:
                LOGGER.info("Start finetuning pretrained model")
                _ = self.train_finetune(cws, train_loader_length, dataloaders_dict, monitor, terminator)
        except KeyboardInterrupt:
            LOGGER.info("Training interrupted")
            self.net = self._init_model()

    def train_from_csv_wandb(self, path, costume_classes, monitor: Monitor, terminator: Terminator):
        """
        Training
        First perform a training of pretrained model.
        If arguments of finetuning are passed, perform a finetuning using weights of above pretrained model.

        Args:
            train_df (pd.DataFrame) : Dataframe containing image file names, labels and sublabel
            monitor (Monitor) : monitor the current batch and epoch of the training loop
            terminator (Terminator) : check if there is any termination request
        """
        # train_df = self._prepare_df(train_df)

        train_df = prepare_df_from_json(path_to_datajson=path, needed_classes=costume_classes)

        train_y = train_df["label"]

        # cws = class_weight.compute_class_weight("balanced", np.unique(train_y), train_y)

        cws = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_y),
            y=train_y
        )
        # cws = dict(zip(np.unique(train_y), cws))

        LOGGER.info(f"Class weights for labels: {cws}")

        LOGGER.info("Loading data")
        train_loader, val_loader = self._prepare_training_generators(
            train_df, self.train_path
        )
        train_loader_length = len(train_loader)
        # Create training and validation dataloaders
        dataloaders_dict = {"train": train_loader, "val": val_loader}
        try:
            # train a pretrained model
            if self.feature_extract:
                LOGGER.info("Start training pretrained models")
                _ = self.train_pretrain_wandb(cws, train_loader_length, dataloaders_dict, monitor, terminator, False)
            # finetune a pretrained model
            if self.finetune_layer != -1:
                LOGGER.info("Start finetuning pretrained model")
                _ = self.train_finetune_wandb(cws, train_loader_length, dataloaders_dict, monitor, terminator)
        except KeyboardInterrupt:
            LOGGER.info("Training interrupted")
            self.net = self._init_model()

    def train_from_csv_wandb_eval_each_epoch(self, path, costume_classes, monitor: Monitor, terminator: Terminator):
        """
        Training
        First perform a training of pretrained model.
        If arguments of finetuning are passed, perform a finetuning using weights of above pretrained model.

        Args:
            train_df (pd.DataFrame) : Dataframe containing image file names, labels and sublabel
            monitor (Monitor) : monitor the current batch and epoch of the training loop
            terminator (Terminator) : check if there is any termination request
        """
        # train_df = self._prepare_df(train_df)

        train_df = prepare_df_from_json(path_to_datajson=path, needed_classes=costume_classes)

        train_y = train_df["label"]

        # cws = class_weight.compute_class_weight("balanced", np.unique(train_y), train_y)

        cws = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_y),
            y=train_y
        )
        # cws = dict(zip(np.unique(train_y), cws))

        LOGGER.info(f"Class weights for labels: {cws}")

        LOGGER.info("Loading data")
        train_loader, val_loader = self._prepare_training_generators(
            train_df, self.train_path
        )
        train_loader_length = len(train_loader)
        # Create training and validation dataloaders
        dataloaders_dict = {"train": train_loader, "val": val_loader}
        try:
            # train a pretrained model
            if self.feature_extract:
                LOGGER.info("Start training pretrained models")
                _ = self.train_pretrain_wandb_each_epoch(cws, train_loader_length, dataloaders_dict, monitor, terminator, False)
            # finetune a pretrained model
            if self.finetune_layer != -1:
                LOGGER.info("Start finetuning pretrained model")
                _ = self.train_finetune_wandb_each_epoch(cws, train_loader_length, dataloaders_dict, monitor, terminator)
        except KeyboardInterrupt:
            LOGGER.info("Training interrupted")
            self.net = self._init_model()

    def train_from_csv(self, path, costume_classes, monitor: Monitor, terminator: Terminator):
        """
        Training
        First perform a training of pretrained model.
        If arguments of finetuning are passed, perform a finetuning using weights of above pretrained model.

        Args:
            train_df (pd.DataFrame) : Dataframe containing image file names, labels and sublabel
            monitor (Monitor) : monitor the current batch and epoch of the training loop
            terminator (Terminator) : check if there is any termination request
        """
        # train_df = self._prepare_df(train_df)

        train_df = prepare_df_from_json(path_to_datajson=path, needed_classes=costume_classes)

        train_y = train_df["label"]

        # cws = class_weight.compute_class_weight("balanced", np.unique(train_y), train_y)

        cws = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_y),
            y=train_y
        )
        # cws = dict(zip(np.unique(train_y), cws))

        LOGGER.info(f"Class weights for labels: {cws}")

        LOGGER.info("Loading data")
        train_loader, val_loader = self._prepare_training_generators(
            train_df, self.train_path
        )
        train_loader_length = len(train_loader)
        # Create training and validation dataloaders
        dataloaders_dict = {"train": train_loader, "val": val_loader}
        try:
            # train a pretrained model
            if self.feature_extract:
                LOGGER.info("Start training pretrained models")
                _ = self.train_pretrain(cws, train_loader_length, dataloaders_dict, monitor, terminator, False)
            # finetune a pretrained model
            if self.finetune_layer != -1:
                LOGGER.info("Start finetuning pretrained model")
                _ = self.train_finetune(cws, train_loader_length, dataloaders_dict, monitor, terminator)
        except KeyboardInterrupt:
            LOGGER.info("Training interrupted")
            self.net = self._init_model()

    def train_k_folds(self, train_df: pd.DataFrame, monitor: Monitor, terminator: Terminator):
        """
        Training on stratified k-folds
        First perform a training of pretrained model.
        If arguments of finetuning are passed, perform a finetuning using weights of above pretrained model.

        Args:
            train_df (pd.DataFrame) : Dataframe containing image file names, labels and sublabel
            monitor (Monitor) : monitor the current batch and epoch of the training loop
            terminator (Terminator) : check if there is any termination request
        Returns:
            best_accs (List) : A list of best val accuracy across all folds
        """
        train_df["prefix"] = train_df.index
        train_df["kfold"] = -1
        # shuffle dataset
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        skf = StratifiedKFold(n_splits=self.folds)
        # perform stratification on subclasses
        for f, (t_, v_) in enumerate(skf.split(X=train_df, y=train_df["subclass"].values)):
            train_df.loc[v_, 'kfold'] = f

        train_df = self._prepare_df(train_df, is_kfold=True)
        train_y = train_df["label"]
        cws = class_weight.compute_class_weight("balanced", np.unique(train_y), train_y)
        LOGGER.info(f"Class weights for labels: {cws}")

        pretrain_acc, finetune_acc, base_path = [], [], self.save_model_path
        for f in range(self.folds):
            LOGGER.info("Loading data")
            train_loader, val_loader = self._prepare_training_generators(
                train_df, self.train_path, True, f
            )
            train_loader_length = len(train_loader)
            # Create training and validation dataloaders
            dataloaders_dict = {"train": train_loader, "val": val_loader}
            LOGGER.info(f"Training for {f} Fold")
            try:
                # train a pretrained model
                if self.feature_extract:
                    LOGGER.info("Start training pretrained models")
                    pretrain_acc.append(
                        self.train_pretrain(cws, train_loader_length, dataloaders_dict, monitor, terminator,
                                            True, f, base_path, self.folds))
                # finetune a pretrained model
                if self.finetune_layer != -1:
                    LOGGER.info("Start finetuning pretrained model")
                    finetune_acc.append(
                        self.train_finetune(cws, train_loader_length, dataloaders_dict, monitor, terminator, True,
                                            self.folds))
            except KeyboardInterrupt:
                LOGGER.info(f"kfold training interrupted at fold {f}")
                self.net = self._init_model()
                return
        if self.feature_extract:
            LOGGER.info(
                f"Pretrained Model => Average validation accuracy across {self.folds} folds : {np.mean(pretrain_acc)}"
            )
            return pretrain_acc
        if self.finetune_layer != -1:
            LOGGER.info(
                f"Finetune Model => Average validation accuracy across {self.folds} folds : {np.mean(finetune_acc)}"
            )
            return finetune_acc
