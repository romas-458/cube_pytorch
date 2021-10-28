import copy
import os
import torch
import torch.nn as nn
from torchvision import models

from efficientnet_pytorch import EfficientNet


def set_parameter_requires_grad(model, feature_extracting: bool, num_ft_layers: int):
    """
    Freeze the weights of the model is feature_extracting=True
    Fine tune layers >= num_ft_layers
    Batch Normalization: https://keras.io/guides/transfer_learning/

    Args:
        model: PyTorch model
        feature_extracting (bool): A bool to set all parameters to be trainable or not
        num_ft_layers (int): Number of layers to freeze and unfreezing the rest
    """
    if feature_extracting:
        if num_ft_layers != -1:
            for i, module in enumerate(model.modules()):
                if i >= num_ft_layers:
                    if not isinstance(module, nn.BatchNorm2d):
                        module.requires_grad_(True)
                else:
                    module.requires_grad_(False)
        else:
            for param in model.parameters():
                param.requires_grad = False


def build_models(
        model_name: str,
        num_classes: int,
        in_channels: int,
        embedding_size: int,
        feature_extract: bool = True,
        use_pretrained: bool = True,
        base_model_path: str = None,
        num_ft_layers: int = -1,
        bst_model_weights=None
):
    """
    Build various architectures to either train from scratch, finetune or as feature extractor.

    Args:
        model_name (str) : Name of model from [enet-b3, enet-b7, resnext101, resnet18, resnet50, densenet121, mobilenetv2]
        num_classes (int) : Number of output classes added as final layer
        in_channels (int) : Number of input channels
        embedding_size (int): Size of intermediate features
        feature_extract (bool): Flag for feature extracting.
                               False = finetune the whole model,
                               True = only update the new added layers params
        use_pretrained (bool): Pretraining parameter to pass to the model or if base_model_path is given use that to
                                initialize the model weights
        base_model_path (str) : Path to imagenet trained weights of selected model (for offline systems)
        num_ft_layers (int) : Number of layers to finetune
                             Default = -1 (do not finetune any layers)
        bst_model_weights : Best weights obtained after training pretrained model
                            which will be used for further finetuning.

    Returns:
        model : A pytorch model
    """
    model_ft = None

    if model_name == "enet-b3":
        model_ft = EfficientNet.from_pretrained("efficientnet-b3")
        if in_channels == 1:
            model_ft._conv_stem = nn.Conv2d(1, 64, kernel_size=3, stride=2, bias=False)
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        num_ftrs = model_ft._fc.in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft._fc = head

    elif model_name == "enet-b7":
        model_ft = EfficientNet.from_pretrained("efficientnet-b7")
        if in_channels == 1:
            model_ft._conv_stem = nn.Conv2d(1, 64, kernel_size=3, stride=2, bias=False)
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        num_ftrs = model_ft._fc.in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft._fc = head

    elif model_name == "resnext101":
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        if base_model_path is not None:
            pretrain_model_wts = torch.load(base_model_path)
            model_ft.load_state_dict(pretrain_model_wts)
        if in_channels == 1:
            model_ft.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        num_ftrs = model_ft.fc.in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft.fc = head

    elif model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained)
        if base_model_path is not None:
            pretrain_model_wts = torch.load(base_model_path)
            model_ft.load_state_dict(pretrain_model_wts)
        if in_channels == 1:
            model_ft.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        num_ftrs = model_ft.fc.in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft.fc = head

    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        if base_model_path is not None:
            pretrain_model_wts = torch.load(base_model_path)
            model_ft.load_state_dict(pretrain_model_wts)
        if in_channels == 1:
            model_ft.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        num_ftrs = model_ft.fc.in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft.fc = head

    elif model_name == "densenet121":
        model_ft = models.densenet121(pretrained=use_pretrained)
        if base_model_path is not None:
            pretrain_model_wts = torch.load(base_model_path)
            model_ft.load_state_dict(pretrain_model_wts)
        if in_channels == 1:
            model_ft.features.conv0 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        num_ftrs = model_ft.classifier.in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft.classifier = head

    elif model_name == "mobilenetv2":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        if base_model_path is not None:
            pretrain_model_wts = torch.load(base_model_path)
            model_ft.load_state_dict(pretrain_model_wts)
        if in_channels == 1:
            model_ft.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        head = nn.Sequential(
            nn.Linear(1280, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft.classifier = head

    else:
        print("Invalid model name, exiting...")
        exit()

    # load best model dict for further finetuning
    if bst_model_weights is not None:
        pretrain_model = torch.load(bst_model_weights)
        best_model_wts = copy.deepcopy(pretrain_model.state_dict())
        if feature_extract and num_ft_layers != -1:
            model_ft.load_state_dict(best_model_wts)
            # delete the trained pretrained weights
            os.remove(bst_model_weights)
    return model_ft
