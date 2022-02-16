from cube_pytorch.pytorch.single_img_pytorch_model import ClassifierModel
from cube_pytorch.pytorch.utils import Monitor, Terminator, EvaluationMonitor

from sklearn import metrics
import os
import pandas as pd
import logging
import wandb

logging.basicConfig()

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# path_to_datajson = '/home/roman/Завантаження/Cube_project/Data 13.09.2021-20210921T124220Z-001/Data 13.09.2021/trainingData.json'
# okt_class = 'OK Top New'
# okb_class = 'OK Bottom New'
# nokt_class = 'NOK Top New'
# nokb_class = 'NOK Bottom New'

# examples = [okb_class, okt_class, nokb_class, nokt_class]

def main(path_to_datajson, examples, root_dir, local_storage_dir, epochs, width, height, save_model_path = "model.pth", ft_layers = 270):
    LOGGER.info("Initializing components")
    ROOT_DIR = root_dir #"/home/roman/Projects/PreProjects/Cube_Project/Cube/train_pytorch"
    model_path = os.path.join(ROOT_DIR, save_model_path)
    ai_default_model_path = os.path.join(ROOT_DIR, "models/cube_resnext101.pth")  # path to save model
    ai_default_base_path = os.path.join(ROOT_DIR, "models/resnext101_32x8d-8ba56ff5.pth")  # imagenet weights
    ai_nok_threshold = 0.5
    local_storage_dir = local_storage_dir # "/home/roman/Завантаження/Cube_project/Data 13.09.2021-20210921T124220Z-001/Data 13.09.2021/blob_storage"

    pytorch_model = ClassifierModel(
        save_model_path=model_path,
        base_model_path=os.path.join(ROOT_DIR, ai_default_base_path),
        train_path=os.path.join(ROOT_DIR, local_storage_dir),
        nok_threshold=ai_nok_threshold,
        epochs = epochs,
        width = width,
        height = height,
        finetune_epochs = epochs,
        finetune_layer = ft_layers,
    )

    terminator = Terminator()
    monitor = Monitor()
    pytorch_model.train_from_csv(path_to_datajson, examples, monitor, terminator)

def main_check_val_loader(path_to_datajson, examples, root_dir, local_storage_dir, epochs, width, height, save_model_path = "model.pth", ft_layers = 270):
    LOGGER.info("Initializing components")
    ROOT_DIR = root_dir #"/home/roman/Projects/PreProjects/Cube_Project/Cube/train_pytorch"
    model_path = os.path.join(ROOT_DIR, save_model_path)
    ai_default_model_path = os.path.join(ROOT_DIR, "models/cube_resnext101.pth")  # path to save model
    ai_default_base_path = os.path.join(ROOT_DIR, "models/resnext101_32x8d-8ba56ff5.pth")  # imagenet weights
    ai_nok_threshold = 0.5
    local_storage_dir = local_storage_dir # "/home/roman/Завантаження/Cube_project/Data 13.09.2021-20210921T124220Z-001/Data 13.09.2021/blob_storage"

    pytorch_model = ClassifierModel(
        save_model_path=model_path,
        base_model_path=os.path.join(ROOT_DIR, ai_default_base_path),
        train_path=os.path.join(ROOT_DIR, local_storage_dir),
        nok_threshold=ai_nok_threshold,
        epochs = epochs,
        width = width,
        height = height,
        finetune_epochs = epochs,
        finetune_layer = ft_layers,
    )

    terminator = Terminator()
    monitor = Monitor()
    dataloaders_dict = pytorch_model.train_from_csv_check_val_loader(path_to_datajson, examples, monitor, terminator)
    return dataloaders_dict

def main_wandb(path_to_datajson, examples, root_dir, local_storage_dir, epochs, width, height, config, feature_extract = True, default_base_path = "models/resnext101_32x8d-8ba56ff5.pth", default_model_path = "models/cube_resnext101.pth"):
    LOGGER.info("Initializing components")
    ROOT_DIR = root_dir #"/home/roman/Projects/PreProjects/Cube_Project/Cube/train_pytorch"
    model_path = os.path.join(ROOT_DIR, "model.pth")
    ai_default_model_path = os.path.join(ROOT_DIR, default_model_path)  # path to save model
    ai_default_base_path = os.path.join(ROOT_DIR, default_base_path)  # imagenet weights
    ai_nok_threshold = 0.5
    local_storage_dir = local_storage_dir # "/home/roman/Завантаження/Cube_project/Data 13.09.2021-20210921T124220Z-001/Data 13.09.2021/blob_storage"

    pytorch_model = ClassifierModel(
        save_model_path=model_path,
        base_model_path=os.path.join(ROOT_DIR, ai_default_base_path),
        train_path=os.path.join(ROOT_DIR, local_storage_dir),
        nok_threshold=ai_nok_threshold,
        epochs = epochs,
        width = width,
        height = height,
        #
        feature_extract = feature_extract,
        trained_model_path = ai_default_model_path,
        #wandb
        finetune_layer = config.finetune_layer,
        finetune_lr_multiplier = config.finetune_lr_multiplier,
        finetune_max_lr_multiplier = config.finetune_max_lr_multiplier,
        finetune_epochs = config.finetune_epochs,
        finetune_embed_dim = config.finetune_embed_dim,
    )

    terminator = Terminator()
    monitor = Monitor()
    pytorch_model.train_from_csv_wandb(path_to_datajson, examples, monitor, terminator)
    # pytorch_model.train_from_csv_wandb(path_to_datajson, examples_train, monitor, terminator)

def main_wandb_eval_each_epoch(path_to_datajson, examples_train, examples_eval, root_dir, local_storage_dir, epochs, width, height, config, path_to_model = "model.pth", feature_extract = True, default_base_path = "models/resnext101_32x8d-8ba56ff5.pth", default_model_path = "models/cube_resnext101.pth"):
    LOGGER.info("Initializing components")
    ROOT_DIR = root_dir #"/home/roman/Projects/PreProjects/Cube_Project/Cube/train_pytorch"
    model_path = os.path.join(ROOT_DIR, path_to_model)
    ai_default_model_path = os.path.join(ROOT_DIR, default_model_path)  # path to save model
    ai_default_base_path = os.path.join(ROOT_DIR, default_base_path)  # imagenet weights
    ai_nok_threshold = 0.5
    local_storage_dir = local_storage_dir # "/home/roman/Завантаження/Cube_project/Data 13.09.2021-20210921T124220Z-001/Data 13.09.2021/blob_storage"

    pytorch_model = ClassifierModel(
        save_model_path=model_path,
        base_model_path=os.path.join(ROOT_DIR, ai_default_base_path),
        train_path=os.path.join(ROOT_DIR, local_storage_dir),
        nok_threshold=ai_nok_threshold,
        epochs = epochs,
        width = width,
        height = height,
        #
        feature_extract = feature_extract,
        trained_model_path = ai_default_model_path,
        #wandb
        finetune_layer = config.finetune_layer,
        finetune_lr_multiplier = config.finetune_lr_multiplier,
        finetune_max_lr_multiplier = config.finetune_max_lr_multiplier,
        finetune_epochs = config.finetune_epochs,
        finetune_embed_dim = config.finetune_embed_dim,

        #
        eval_examples = examples_eval,
        path_to_datajson = path_to_datajson,
    )

    terminator = Terminator()
    monitor = Monitor()
    pytorch_model.train_from_csv_wandb_eval_each_epoch(path_to_datajson, examples_train, monitor, terminator)

def evaluation_wandb(path_to_datajson, examples, root_dir, local_storage_dir, epochs, path_to_model):
    ROOT_DIR = root_dir  # "/home/roman/Projects/PreProjects/Cube_Project/Cube/train_pytorch"
    ai_default_model_path = os.path.join(ROOT_DIR, path_to_model)  # path to save model
    model_path = os.path.join(ROOT_DIR, path_to_model)
    ai_default_base_path = os.path.join(ROOT_DIR, "models/resnext101_32x8d-8ba56ff5.pth")  # imagenet weights
    ai_nok_threshold = 0.5

    pytorch_model = ClassifierModel(
        save_model_path=ROOT_DIR + path_to_model,
        base_model_path=os.path.join(ROOT_DIR, ai_default_base_path),
        train_path=os.path.join(ROOT_DIR, local_storage_dir),
        nok_threshold=ai_nok_threshold,
        epochs=epochs,
    )
    # pytorch_model = ClassifierModel(
    #     save_model_path=model_path,
    #     # base_model_path=os.path.join(ROOT_DIR, path_to_model),
    #     base_model_path=ai_default_model_path,
    #     train_path=os.path.join(ROOT_DIR, local_storage_dir),
    #     trained_model_path =
    #     nok_threshold=ai_nok_threshold,
    #     epochs=epochs,
    # )
    terminator = Terminator()
    monitor = EvaluationMonitor()
    eval_df, preds = pytorch_model.evaluate_from_csv(path_to_datajson, examples, monitor, terminator)

    predictions = [0 if x < pytorch_model.nok_threshold else 1 for x in preds]
    LOGGER.info("predictions len = " + str(len(predictions)) + 'eval len = ' + str(len(eval_df)))
    assert len(predictions) == len(eval_df)
    # NOTE works only for binary case
    tn, fp, fn, tp = metrics.confusion_matrix(eval_df["label"].values, predictions).ravel()
    print('tn= ' + str(tn) + 'fp= ' + str(fp) + 'fn= ' + str(fn) + 'tp= ' + str(tp))
    wandb.log({'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})

def evaluation(path_to_datajson, examples, root_dir, local_storage_dir, epochs, path_to_model, width = 512, height = 512):
    ROOT_DIR = root_dir  # "/home/roman/Projects/PreProjects/Cube_Project/Cube/train_pytorch"
    ai_default_model_path = os.path.join(ROOT_DIR, path_to_model)  # path to save model
    model_path = os.path.join(ROOT_DIR, path_to_model)
    ai_default_base_path = os.path.join(ROOT_DIR, "models/resnext101_32x8d-8ba56ff5.pth")  # imagenet weights
    ai_nok_threshold = 0.5


    pytorch_model = ClassifierModel(
        save_model_path=ROOT_DIR + path_to_model,
        base_model_path=os.path.join(ROOT_DIR, ai_default_base_path),
        train_path=os.path.join(ROOT_DIR, local_storage_dir),
        nok_threshold=ai_nok_threshold,
        epochs = epochs,
        width=width,
        height=height,
    )
    # pytorch_model = ClassifierModel(
    #     save_model_path=model_path,
    #     # base_model_path=os.path.join(ROOT_DIR, path_to_model),
    #     base_model_path=ai_default_model_path,
    #     train_path=os.path.join(ROOT_DIR, local_storage_dir),
    #     trained_model_path =
    #     nok_threshold=ai_nok_threshold,
    #     epochs=epochs,
    # )
    terminator = Terminator()
    monitor = EvaluationMonitor()
    eval_df, preds = pytorch_model.evaluate_from_csv(path_to_datajson, examples, monitor, terminator)

    predictions = [0 if x < pytorch_model.nok_threshold else 1 for x in preds]
    LOGGER.info("predictions len = " + str(len(predictions)) + 'eval len = ' + str(len(eval_df)))
    assert len(predictions) == len(eval_df)
    # NOTE works only for binary case
    tn, fp, fn, tp = metrics.confusion_matrix(eval_df["label"].values, predictions).ravel()
    print('tn= ' + str(tn) + 'fp= ' + str(fp) + 'fn= '+ str(fn) + 'tp= '+ str(tp))

def evaluation_check_df(path_to_datajson, examples, root_dir, local_storage_dir, epochs, path_to_model):
    ROOT_DIR = root_dir  # "/home/roman/Projects/PreProjects/Cube_Project/Cube/train_pytorch"
    ai_default_model_path = os.path.join(ROOT_DIR, path_to_model)  # path to save model
    model_path = os.path.join(ROOT_DIR, path_to_model)
    ai_default_base_path = os.path.join(ROOT_DIR, "models/resnext101_32x8d-8ba56ff5.pth")  # imagenet weights
    ai_nok_threshold = 0.5

    pytorch_model = ClassifierModel(
        save_model_path=ROOT_DIR + path_to_model,
        base_model_path=os.path.join(ROOT_DIR, ai_default_base_path),
        train_path=os.path.join(ROOT_DIR, local_storage_dir),
        nok_threshold=ai_nok_threshold,
        epochs=epochs,
    )
    # pytorch_model = ClassifierModel(
    #     save_model_path=model_path,
    #     # base_model_path=os.path.join(ROOT_DIR, path_to_model),
    #     base_model_path=ai_default_model_path,
    #     train_path=os.path.join(ROOT_DIR, local_storage_dir),
    #     trained_model_path =
    #     nok_threshold=ai_nok_threshold,
    #     epochs=epochs,
    # )
    terminator = Terminator()
    monitor = EvaluationMonitor()
    eval_df, preds = pytorch_model.evaluate_from_csv(path_to_datajson, examples, monitor, terminator)

    return eval_df, preds

def evaluation_per_object(path_to_datajson, examples, root_dir, local_storage_dir, epochs, path_to_model):
    ROOT_DIR = root_dir  # "/home/roman/Projects/PreProjects/Cube_Project/Cube/train_pytorch"
    ai_default_model_path = os.path.join(ROOT_DIR, path_to_model)  # path to save model
    model_path = os.path.join(ROOT_DIR, path_to_model)
    ai_default_base_path = os.path.join(ROOT_DIR, "models/resnext101_32x8d-8ba56ff5.pth")  # imagenet weights
    ai_nok_threshold = 0.5


    pytorch_model = ClassifierModel(
        save_model_path=ROOT_DIR + path_to_model,
        base_model_path=os.path.join(ROOT_DIR, ai_default_base_path),
        train_path=os.path.join(ROOT_DIR, local_storage_dir),
        nok_threshold=ai_nok_threshold,
        epochs = epochs,
    )
    # pytorch_model = ClassifierModel(
    #     save_model_path=model_path,
    #     # base_model_path=os.path.join(ROOT_DIR, path_to_model),
    #     base_model_path=ai_default_model_path,
    #     train_path=os.path.join(ROOT_DIR, local_storage_dir),
    #     trained_model_path =
    #     nok_threshold=ai_nok_threshold,
    #     epochs=epochs,
    # )
    terminator = Terminator()
    monitor = EvaluationMonitor()
    eval_df, preds = pytorch_model.evaluate_from_csv_per_object(path_to_datajson, examples, monitor, terminator)

    predictions = [0 if x < pytorch_model.nok_threshold else 1 for x in preds]
    LOGGER.info("predictions len = " + str(len(predictions)) + 'eval len = ' + str(len(eval_df)))
    assert len(predictions) == len(eval_df)
    # NOTE works only for binary case
    tn, fp, fn, tp = metrics.confusion_matrix(eval_df["label"].values, predictions).ravel()
    print('tn= ' + str(tn) + 'fp= ' + str(fp) + 'fn= '+ str(fn) + 'tp= '+ str(tp) )