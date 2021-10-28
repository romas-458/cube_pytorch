from cube_pytorch.pytorch.single_img_pytorch_model import ClassifierModel
from cube_pytorch.pytorch.utils import Monitor, Terminator
import os
import pandas as pd

# path_to_datajson = '/home/roman/Завантаження/Cube_project/Data 13.09.2021-20210921T124220Z-001/Data 13.09.2021/trainingData.json'
# okt_class = 'OK Top New'
# okb_class = 'OK Bottom New'
# nokt_class = 'NOK Top New'
# nokb_class = 'NOK Bottom New'

# examples = [okb_class, okt_class, nokb_class, nokt_class]

def main(path_to_datajson, examples, root_dir, local_storage_dir, epochs):
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
