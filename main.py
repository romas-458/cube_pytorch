from cube_pytorch.pytorch.single_img_pytorch_model import ClassifierModel
from cube_pytorch.pytorch.utils import Monitor, Terminator, EvaluationMonitor

from sklearn import metrics
import os
import pandas as pd
import logging

logging.basicConfig()

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# path_to_datajson = '/home/roman/Завантаження/Cube_project/Data 13.09.2021-20210921T124220Z-001/Data 13.09.2021/trainingData.json'
# okt_class = 'OK Top New'
# okb_class = 'OK Bottom New'
# nokt_class = 'NOK Top New'
# nokb_class = 'NOK Bottom New'

# examples = [okb_class, okt_class, nokb_class, nokt_class]

def main(path_to_datajson, examples, root_dir, local_storage_dir, epochs):
    LOGGER.info("Initializing components")
    ROOT_DIR = root_dir #"/home/roman/Projects/PreProjects/Cube_Project/Cube/train_pytorch"
    model_path = ROOT_DIR + "models_out"
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
    )

    terminator = Terminator()
    monitor = Monitor()
    pytorch_model.train_from_csv(path_to_datajson, examples, monitor, terminator)

def evaluation(path_to_datajson, examples, root_dir, local_storage_dir, epochs, path_to_model):
    ROOT_DIR = root_dir  # "/home/roman/Projects/PreProjects/Cube_Project/Cube/train_pytorch"
    ai_default_model_path = os.path.join(ROOT_DIR, path_to_model)  # path to save model
    model_path = os.path.join(ROOT_DIR, path_to_model)
    ai_default_base_path = os.path.join(ROOT_DIR, "models/resnext101_32x8d-8ba56ff5.pth")  # imagenet weights
    ai_nok_threshold = 0.3


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
    eval_df, preds = pytorch_model.evaluate_from_csv(path_to_datajson, examples, monitor, terminator)

    predictions = [0 if x < pytorch_model.nok_threshold else 1 for x in preds]
    LOGGER.info("predictions len = " + str(len(predictions)) + 'eval len = ' + str(len(eval_df)))
    assert len(predictions) == len(eval_df)
    # NOTE works only for binary case
    tn, fp, fn, tp = metrics.confusion_matrix(eval_df["label"].values, predictions).ravel()
    print('tp= ' + str(tp) + 'fp= ' + str(fp) + 'fn= '+ str(fn) + 'tp= '+ str(tp) )