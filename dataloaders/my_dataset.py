import os

import cv2
import numpy as np
from torch.utils.data.dataloader import Dataset

from dataloaders.data_process import RandomGenerator


class MyDataset(Dataset):
    def __init__(self, file_path, run_type="train", val_fold="fold1"):
        self.file_path = file_path
        self.run_type = run_type
        fold_list = ["fold1", "fold2", "fold3", "fold4", "fold5"]
        data_path = []
        if run_type == "train":
            for fold_name in fold_list:
                if fold_name != val_fold:
                    data_name_list = os.listdir(f"{file_path}/{fold_name}/images")
                    for data_name in data_name_list:
                        data_path.append({"image": f"{file_path}/{fold_name}/images/{data_name}",
                                          "label": f"{file_path}/{fold_name}/scribble/{data_name}"})
        else:
            data_name_list = os.listdir(f"{file_path}/{val_fold}/images")
            for data_name in data_name_list:
                data_path.append({"image": f"{file_path}/{val_fold}/images/{data_name}",
                                  "label": f"{file_path}/{val_fold}/labels/{data_name}"})
        self.trans = RandomGenerator(run_type)
        self.data_path = data_path
        self.data_len = len(data_path)

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        d = self.data_path[item]
        image_path = d["image"]
        label_path = d["label"]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if image.shape[0] != 256:
            image = cv2.resize(image, (256, 256))
            label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
        if label.max() > 100:
            label = (label > 0).astype(np.uint8)
        image, label = self.trans(image, label)
        return image, label
