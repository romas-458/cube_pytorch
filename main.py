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

def main(path_to_datajson, examples, root_dir, local_storage_dir):
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
        nok_threshold=ai_nok_threshold
    )

    terminator = Terminator()
    monitor = Monitor()
    pytorch_model.train_from_csv(path_to_datajson, examples, monitor, terminator)


# df = pd.read_json(path_to_datajson)
#
# print(df.head())
# for col in df.columns:
#     print(col)
#
# subdf = df[["class_name", "class_type", "image_urls"]]
# print(subdf.head())
#
#
#
#
# newdf = read_df_from_json(path_to_datajson, examples)
#
# print(newdf.head())
# print(examples)
#
# training_subset = subdf[(subdf["class_name"] == examples[0]) | (subdf["class_name"] == examples[1]) | (subdf["class_name"] == examples[2]) | (subdf["class_name"] == examples[3])]
#
#
# # new = [subdf[subdf["class_name"] == ex] for ex in examples if subdf["class_name"] == ex]
# print(training_subset.tail())


# jf = {'_id': {'$oid': '60d966ca71747f7c1b8428b0'},
#  'class_name': '210628-OK-B',
#  'class_type': 'OK',
#  'image_urls': ['C:\\dev\\cube\\blob_storage\\0581624860361242465400_0.jpg',
#   'C:\\dev\\cube\\blob_storage\\0581624860361242465400_1.jpg',
#   'C:\\dev\\cube\\blob_storage\\0581624860361242465400_2.jpg',
#   'C:\\dev\\cube\\blob_storage\\0581624860361242465400_3.jpg'],
#  'is_correction': False,
#  'timestamp': {'$date': '2021-06-28T05:44:06.961Z'}}
#
# print(jf['class_name'])

# pytorch_model.train(loader.load(None)[0], self.monitor, self.terminator)